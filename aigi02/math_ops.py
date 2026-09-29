"""Explicit numerical definitions. No external pretrained models are required."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def ste_round(x: Tensor) -> Tensor:
    return x + (x.round() - x).detach()


def normalize_map(x: Tensor, eps: float = 1e-6) -> Tensor:
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(eps)


def resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def importance(s: Tensor, w_min: float = .1, eta: float = 2.) -> Tensor:
    if not (0 < w_min <= 1 and eta > 0):
        raise ValueError('w_min must be in (0,1], eta must be positive')
    return w_min + (1 - w_min) * s.clamp(0, 1).pow(eta)


def information_scores(errors_cond: Tensor, errors_null: Tensor,
                       log_snr: Tensor) -> tuple[Tensor, Tensor]:
    """[K,B,1,H,W] errors summed over latent channels, averaged over noise draws.

    Finite trapezoidal quadrature, NOT an exact mutual-information estimator.
    The signed explanatory score is deliberately not clamped.
    """
    if log_snr.ndim != 1 or len(log_snr) < 2:
        raise ValueError('At least two log-SNR nodes are required')
    if not bool(torch.all(log_snr[1:] > log_snr[:-1])):
        raise ValueError('log-SNR nodes must be strictly increasing')
    if errors_cond.shape != errors_null.shape or errors_cond.shape[0] != len(log_snr):
        raise ValueError('Error tensors and quadrature nodes are incompatible')
    c = .5 * torch.trapezoid(errors_cond, log_snr, dim=0)
    e = .5 * torch.trapezoid(errors_null - errors_cond, log_snr, dim=0)
    return c, e


def gaussian_mass(value: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
    scale = scale.clamp_min(.11)
    upper = (value - mean + .5) / (scale * math.sqrt(2.))
    lower = (value - mean - .5) / (scale * math.sqrt(2.))
    return (.5 * (torch.erf(upper) - torch.erf(lower))).clamp_min(1e-9)


def waterfill_target(w: Tensor, c: Tensor, rate_map: Tensor,
                     iterations: int = 64) -> Tensor:
    """Solve a separate tau per image, matching TOTAL bits (not mean bits)."""
    score = .5 * w.clamp_min(1e-6).log2() + c / math.log(2.)
    s = score.detach().flatten(1)
    budget = rate_map.detach().flatten(1).sum(1, keepdim=True).clamp_min(0)
    lo = s.amin(1, keepdim=True) - budget - 1
    hi = s.amax(1, keepdim=True) + 1
    for _ in range(iterations):
        mid = (lo + hi) * .5
        amount = (s - mid).clamp_min(0).sum(1, keepdim=True)
        lo = torch.where(amount > budget, mid, lo)
        hi = torch.where(amount > budget, hi, mid)
    return (s - (lo + hi) * .5).clamp_min(0).reshape_as(score)


def local_snr(mu: Tensor, sigma: Tensor, delta: float = 1.) -> Tensor:
    if delta <= 0:
        raise ValueError('delta must be positive')
    return ((mu.square() + sigma.square()) / (delta * delta / 12.)).mean(1, keepdim=True)


def snr_timestep(snr: Tensor, w_hat: Tensor, beta: Tensor | float = 0.,
                 rho: Tensor | float = 0., eps: float = 1e-4) -> Tensor:
    t = 1 / (1 + snr.clamp_min(1e-12).sqrt())
    t = t.clamp(eps, 1 - eps)
    return torch.sigmoid(torch.logit(t) - beta * (2 * w_hat - 1) + rho).clamp(eps, 1 - eps)


def weighted_mse(x: Tensor, y: Tensor, w: Tensor) -> Tensor:
    w = resize(w, x.shape[-2:]).detach()
    error = (x - y).square().mean(1, keepdim=True)
    return ((error * w).sum((1, 2, 3)) / w.sum((1, 2, 3)).clamp_min(1e-8)).mean()


def cell_loss(reencoded_y: Tensor, m: Tensor, qhat: Tensor) -> Tensor:
    """qhat is BEFORE learned residual prediction, with the actual mean offset."""
    return ((reencoded_y / m.detach() - qhat.detach()).abs() - .5).clamp_min(0).mean()


class CrossAttention2d(nn.Module):
    def __init__(self, channels: int, text_dim: int, heads: int = 4, zero_out: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.text = nn.Linear(text_dim, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        if zero_out:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: Tensor, text: Tensor, mask: Tensor, return_attention: bool = False):
        b, c, h, w = x.shape
        query = x.flatten(2).transpose(1, 2)
        keys = self.text(text.to(x.dtype))
        v, attention = self.attn(self.norm(query), keys, keys,
                         key_padding_mask=~mask.bool(), need_weights=return_attention, average_attn_weights=True)
        result = x + v.transpose(1, 2).reshape(b, c, h, w)
        return (result, attention) if return_attention else result


class MapPredictor(nn.Module):
    """Two cross-attention blocks; c >= 0, signed e, S in [0,1]."""
    def __init__(self, channels: int, text_dim: int, hidden: int = 64):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, hidden, 3, padding=1)
        self.attn = nn.ModuleList([CrossAttention2d(hidden, text_dim) for _ in range(2)])
        self.condition = nn.Linear(1, hidden)
        self.token_weights = nn.Sequential(nn.Linear(text_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.token_weights[-1].weight)
        nn.init.zeros_(self.token_weights[-1].bias)
        self.use_token_weights = True
        self.out = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, 3, 1))

    def forward(self, y: Tensor, text: Tensor, mask: Tensor, log_lambda: Tensor) -> dict:
        h = self.in_proj(y) + self.condition(log_lambda.reshape(-1, 1))[:, :, None, None]
        h = self.attn[0](h, text, mask)
        h, attention = self.attn[1](h, text, mask, return_attention=True)
        b, _, hh, ww = h.shape
        token_maps = normalize_map(attention.transpose(1, 2).reshape(b, -1, hh, ww))
        omega = (F.softplus(self.token_weights(text)).squeeze(-1) if self.use_token_weights
                 else torch.ones_like(mask, dtype=h.dtype))
        omega = omega * mask
        omega = omega / omega.sum(1, keepdim=True).clamp_min(1e-8)
        saliency = normalize_map((token_maps * omega[:, :, None, None]).sum(1, keepdim=True))
        s, c, e = self.out(h).chunk(3, 1)
        saliency = (saliency + .1 * s.tanh()).clamp(0, 1)
        return {'S': saliency, 'c': F.softplus(c), 'e': e}


class SemanticUGAQ(nn.Module):
    def __init__(self, channels: int, max_m: float = 32.):
        super().__init__()
        self.max_m = max_m
        self.texture = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(64, 1, 3, padding=1))
        self.correction = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, y: Tensor, hyper_up: Tensor, maps: dict, w: Tensor,
                log_lambda: Tensor, texture: bool = True, semantic: bool = True,
                correction: bool = True) -> Tensor:
        log_m = torch.zeros_like(w)
        if texture:
            fu = 1 + F.softplus(self.texture(y - hyper_up))
            log_m = log_m + fu.log()
        if semantic:
            log_m = log_m - .5 * w.clamp_min(1e-6).log()
        if correction:
            lam = log_lambda.reshape(-1, 1, 1, 1).expand_as(w)
            features = torch.cat((maps['S'], maps['c'].log1p(),
                                  torch.sign(maps['e']) * maps['e'].abs().log1p(), lam), 1)
            log_m = log_m + self.correction(features)
        return log_m.clamp(0, math.log(self.max_m)).exp()


class GenerationField(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.importance_head = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1),
                                             nn.SiLU(), nn.Conv2d(64, 1, 1))
        self.residual = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, hyper: Tensor, mu: Tensor, sigma: Tensor, beta: Tensor,
                log_lambda: Tensor, semantic: bool = True, residual: bool = True) -> dict:
        w_hat = .1 + .9 * self.importance_head(hyper).sigmoid()
        snr = local_snr(mu, sigma)
        b = beta.reshape(-1, 1, 1, 1).expand_as(w_hat)
        l = log_lambda.reshape(-1, 1, 1, 1).expand_as(w_hat)
        rho = self.residual(torch.cat((snr.log1p(), w_hat, b, l), 1)) if residual else 0.
        t = snr_timestep(snr, w_hat, b if semantic else 0., rho)
        return {'t': t, 'snr': snr, 'w_hat': w_hat}


class SpatialAdaLN(nn.Module):
    """Real per-token scale/shift injected after each existing norm layer."""
    def __init__(self, width: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normalized_tokens: Tensor, t: Tensor, beta: Tensor, log_lambda: Tensor,
                grid: tuple[int, int]) -> Tensor:
        field = resize(t, grid).flatten(2).transpose(1, 2)
        if field.shape[1] != normalized_tokens.shape[1]:
            raise ValueError('Spatial timestep grid does not match transformer tokens')
        b = beta.reshape(-1, 1, 1).expand_as(field)
        lam = log_lambda.reshape(-1, 1, 1).expand_as(field)
        shift, scale = self.net(torch.cat((field, b, lam), -1).to(normalized_tokens.dtype)).chunk(2, -1)
        return normalized_tokens * (1 + scale) + shift
