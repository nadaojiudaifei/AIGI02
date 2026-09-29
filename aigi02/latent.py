"""Additive semantic codec. Original DiT-IC files are imported, never edited.

Four disjoint groups are coded independently. The same stage implementation is
used by forward(), encode() and decode(). Decoder inputs contain no original
image, encoder map or quantization-scale map.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .math_ops import (CrossAttention2d, MapPredictor, SemanticUGAQ, GenerationField,
                       importance, resize, ste_round, gaussian_mass)
from .reference_coder import ReferenceBottleneck, ReferenceGaussian


def conv(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)


class TinyAnalysis(nn.Module):
    def __init__(self, latent_c, channels):
        super().__init__()
        self.net = nn.Sequential(conv(latent_c * 2, channels, 2), nn.SiLU(), conv(channels, channels))
    def forward(self, z, aux): return self.net(torch.cat((z, aux), 1))


class TinyCore(nn.Module):
    """Only for tests and mechanics demos. Not a DiT-IC performance baseline."""
    def __init__(self, channels=16, latent_c=8):
        super().__init__(); m = channels
        self.M = m
        self.g_a = TinyAnalysis(latent_c, m)
        def up(out):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), conv(m, out))
        self.g_s = up(latent_c); self.scale = up(latent_c)
        self.prompt = up(m); self.aux = up(latent_c)
        self.h_a = nn.Sequential(conv(m, m//2, 2), nn.SiLU(), conv(m//2, m//2, 2))
        self.h_s = nn.Sequential(nn.Upsample(scale_factor=4, mode='nearest'), conv(m//2, m))
        self.adapter_in = nn.ModuleList([conv(m, m*2) for _ in range(4)])
        self.g_c = nn.Sequential(nn.SiLU(), conv(m*2, m*2), nn.SiLU())
        self.adapter_out = nn.ModuleList([conv(m*2, m*2) for _ in range(4)])
        self.LRP = nn.ModuleList([conv(m*2, m) for _ in range(4)])
        self.entropy_bottleneck = ReferenceBottleneck(m//2)
        self.gaussian_conditional = ReferenceGaussian()
    def update(self, force=False): return True
    def aux_loss(self): return self.entropy_bottleneck.loss()


def masks_four(x: Tensor) -> list[Tensor]:
    b, c, h, w = x.shape
    if c % 4: raise ValueError('Four-part coding requires C divisible by 4')
    row = torch.arange(h, device=x.device).reshape(h, 1) % 2
    col = torch.arange(w, device=x.device).reshape(1, w) % 2
    positions = row * 2 + col
    permutations = ((0,1,2,3), (3,2,1,0), (2,3,0,1), (1,0,3,2))
    return [torch.cat([(positions == n).to(x).expand(b, c//4, h, w)
                        for n in perm], 1) for perm in permutations]


def squeeze(x: Tensor, mask: Tensor) -> Tensor:
    return sum(a * b for a, b in zip(x.chunk(4, 1), mask.chunk(4, 1)))


def unsqueeze(x: Tensor, mask: Tensor) -> Tensor:
    return torch.cat([x * m for m in mask.chunk(4, 1)], 1)


class SemanticLatent(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__(); self.cfg = cfg
        m = cfg['model']['channels']; d = cfg['model']['text_dim']
        if cfg['backend'] == 'tiny': self.core = TinyCore(m, cfg['model']['latent_channels'])
        else:
            from models.latent_codec import LatentCodec
            self.core = LatentCodec(ch_emd=32, channel=320, channel_out=32)
        self.predictor = MapPredictor(m, d)
        self.predictor.use_token_weights = cfg['features']['token_weights']
        self.ugaq = SemanticUGAQ(m)
        self.hyper_attention = CrossAttention2d(m, d)
        self.context_attention = nn.ModuleList([CrossAttention2d(2*m, d) for _ in range(4)])
        self.field = GenerationField(m)

    def update(self): self.core.update(force=True)

    def encode_features(self, z0, aux, text, mask, log_lambda, oracle=None):
        y = self.core.g_a(z0, aux)
        maps = self.predictor(y, text, mask, log_lambda)
        if oracle is not None:
            if set(oracle) != {'S', 'c', 'e'}: raise ValueError('Oracle maps require S,c,e')
            maps = {k: resize(v.to(y), y.shape[-2:]) for k, v in oracle.items()}
        flags = self.cfg['features']
        maps = dict(maps)
        if not flags['importance']: maps['S'] = torch.zeros_like(maps['S'])
        if not flags['conditional_score']:
            maps['c'] = torch.zeros_like(maps['c']); maps['e'] = torch.zeros_like(maps['e'])
        w = importance(maps['S']) if flags['importance'] else torch.ones_like(maps['S'])
        if self.cfg['method'] == 'no_adaptation':
            m = torch.ones_like(w)
        else:
            # Encoder-only first pass. It is NOT transmitted and not a decoder dependency.
            z_pre = self.core.h_a(y)
            offset = self.core.entropy_bottleneck._get_medians()
            z_pre_hat = ste_round(z_pre - offset) + offset
            hyper_pre = self.core.h_s(z_pre_hat)[..., :y.shape[-2], :y.shape[-1]]
            semantic = flags['importance'] and self.cfg['method'] != 'texture_only'
            m = self.ugaq(y, hyper_pre, maps, w, log_lambda,
                          texture=flags['texture'], semantic=semantic,
                          correction=flags['correction'] and self.cfg['method'] != 'texture_only')
        ybar = y / m
        z = self.core.h_a(ybar)
        return {'y': y, 'ybar': ybar, 'z': z, 'm': m, 'w': w, 'maps': maps}

    def hyper(self, zhat, size, text, mask):
        h = self.core.h_s(zhat)[..., :size[0], :size[1]]
        if tuple(h.shape[-2:]) != tuple(size): raise ValueError('Hyperprior shape inconsistent')
        if self.cfg['features']['prompt_entropy']:
            h = self.hyper_attention(h, text, mask)
        return h

    def stage_params(self, base, index, text, mask):
        h = self.core.g_c(self.core.adapter_in[index](base))
        if self.cfg['features']['prompt_entropy']:
            h = self.context_attention[index](h, text, mask)
        mean, scale = self.core.adapter_out[index](h).chunk(2, 1)
        # Match GaussianConditional's lower bound in forward AND actual coding.
        return mean, scale.clamp_min(.11)

    def groups(self, hyper, text, mask, mode, ybar=None, streams=None):
        if mode not in ('forward', 'encode', 'decode'): raise ValueError(mode)
        base = hyper; masks = masks_four(hyper)
        q_all = torch.zeros_like(hyper); post_all = torch.zeros_like(hyper)
        mu_all = torch.zeros_like(hyper); std_all = torch.zeros_like(hyper)
        likelihood = torch.zeros_like(hyper); out_streams = []
        gaussian = self.core.gaussian_conditional
        if mode == 'decode' and (streams is None or len(streams) != 4):
            raise ValueError('Exactly four main-latent groups are required')
        for index, group_mask in enumerate(masks):
            mu, std = self.stage_params(base, index, text, mask)
            means = squeeze(mu, group_mask); scales = squeeze(std, group_mask)
            if mode == 'decode':
                q = gaussian.decompress([streams[index]], gaussian.build_indexes(scales), means=means)
                prob = gaussian_mass(q, means, scales)
            else:
                ys = squeeze(ybar, group_mask)
                if mode == 'encode':
                    encoded = gaussian.compress(ys, gaussian.build_indexes(scales), means=means)
                    if len(encoded) != 1: raise ValueError('Encode one image at a time')
                    out_streams.append(encoded[0])
                    # Use exactly the values the independent decoder will obtain.
                    q = gaussian.decompress(encoded, gaussian.build_indexes(scales), means=means)
                    prob = gaussian_mass(q, means, scales)
                else:
                    _, prob = gaussian(ys, scales, means=means)
                    q = ste_round(ys - means) + means
            qfull = unsqueeze(q, group_mask)
            # Store cell center BEFORE LRP. LRP is a decoder correction, not a symbol.
            residual = .5 * torch.tanh(self.core.LRP[index](torch.cat((qfull, base), 1)) * group_mask)
            post = qfull + residual
            q_all = q_all + qfull; post_all = post_all + post
            mu_all = mu_all + mu * group_mask; std_all = std_all + std * group_mask
            likelihood = likelihood + unsqueeze(prob, group_mask)
            base = base * (1 - group_mask) + post
        return {'qhat': q_all, 'yhat': post_all, 'mu': mu_all, 'sigma': std_all,
                'likelihood_y': likelihood.clamp_min(1e-9), 'streams_y': out_streams,
                'hyper': hyper}

    def synthesis(self, result):
        yhat = result['yhat']
        result.update(mean=self.core.g_s(yhat),
                      logvar=self.core.scale(result['sigma']).clamp(-30,20),
                      res=self.core.aux(yhat), prompt_latent=self.core.prompt(yhat))
        return result

    def forward(self, z0, aux, text, mask, log_lambda, oracle=None):
        enc = self.encode_features(z0, aux, text, mask, log_lambda, oracle)
        _, pz = self.core.entropy_bottleneck(enc['z'])
        medians = self.core.entropy_bottleneck._get_medians()
        zhat = ste_round(enc['z'] - medians) + medians
        hyper = self.hyper(zhat, enc['y'].shape[-2:], text, mask)
        result = self.groups(hyper, text, mask, 'forward', ybar=enc['ybar'])
        result.update(enc); result['likelihood_z'] = pz.clamp_min(1e-9)
        return self.synthesis(result)

    @torch.no_grad()
    def compress(self, z0, aux, text, mask, log_lambda, oracle=None):
        if z0.shape[0] != 1: raise ValueError('Real entropy coding accepts batch size one')
        if next(self.parameters()).device.type != 'cpu':
            raise ValueError('Entropy coding is standardized on CPU FP32; call coding_mode()')
        self.update()
        enc = self.encode_features(z0, aux, text, mask, log_lambda, oracle)
        eb = self.core.entropy_bottleneck
        streams_z = eb.compress(enc['z'])
        zsize = tuple(enc['z'].shape[-2:]); ysize = tuple(enc['y'].shape[-2:])
        zhat = eb.decompress(streams_z, zsize)
        hyper = self.hyper(zhat, ysize, text, mask)
        result = self.groups(hyper, text, mask, 'encode', ybar=enc['ybar'])
        result.update(enc)
        return [streams_z[0], *result['streams_y']], {'z_size': list(zsize), 'y_size': list(ysize)}, self.synthesis(result)

    @torch.no_grad()
    def decompress(self, streams, shape, text, mask):
        if len(streams) != 5: raise ValueError('One hyperprior plus four groups required')
        if next(self.parameters()).device.type != 'cpu': raise ValueError('Entropy decoding must use CPU FP32')
        self.update()
        zhat = self.core.entropy_bottleneck.decompress([streams[0]], shape['z_size'])
        hyper = self.hyper(zhat, shape['y_size'], text, mask)
        return self.synthesis(self.groups(hyper, text, mask, 'decode', streams=streams[1:]))
