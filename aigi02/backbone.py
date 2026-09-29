"""Backbones and text conditioning, with an explicit tiny test implementation.

SANA's public timestep argument is scalar-per-image. Spatial fields therefore
enter real per-token normalization layers via additive hooks; no unsupported
spatial timestep tensor is passed into the stock API.
"""
from __future__ import annotations
import contextlib
from pathlib import Path
from collections import OrderedDict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .math_ops import SpatialAdaLN


def frozen(module):
    module.requires_grad_(False); module.eval(); return module


def safe_torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


class TextConditioner:
    """Frozen CPU FP32 text features; bounded cache; never part of the optimizer.

    The CPU rule is shared by encoder and decoder. Deploy with the same pinned
    software/CPU execution profile; floating neural entropy models are not a
    platform-independent integer format.
    """
    def __init__(self, cfg):
        self.cfg = cfg; self.cache = OrderedDict(); self.model = None; self.tokenizer = None
        if cfg['backend'] == 'sana':
            from transformers import AutoTokenizer, AutoModel
            source = cfg['assets']['text'] or cfg['assets']['sana']
            opts = {'local_files_only': cfg['assets']['local_files_only'], 'revision': cfg['assets']['revision']}
            self.tokenizer = AutoTokenizer.from_pretrained(source, subfolder='tokenizer', **opts)
            self.tokenizer.padding_side = 'right'
            self.model = frozen(AutoModel.from_pretrained(source, subfolder='text_encoder',
                                      torch_dtype=torch.float32, **opts)).cpu()
        else:
            # Fixed analytic byte embeddings, explicitly NOT a semantic model.
            d = cfg['model']['text_dim']
            ids = torch.arange(257).float()[:,None]; dims = torch.arange(1,d+1).float()[None,:]
            self.table = torch.sin(ids * dims * .017)

    @torch.no_grad()
    def one(self, prompt: str):
        if prompt in self.cache:
            value = self.cache.pop(prompt); self.cache[prompt] = value; return value
        length = self.cfg['model']['max_tokens']
        if self.model is None:
            tokens = [256] + list(prompt.encode('utf-8'))
            if len(tokens) > length: tokens = tokens[:length]
            valid = torch.zeros(length, dtype=torch.bool); valid[:len(tokens)] = True
            ids = tokens + [0] * (length-len(tokens))
            emb = self.table[ids].unsqueeze(0)
            content = valid.clone(); content[0] = False
            if not content.any(): content[0] = True
            value = (emb, valid.unsqueeze(0), content.unsqueeze(0))
        else:
            encoded = self.tokenizer(prompt, max_length=length, padding='max_length',
                                     truncation=True, return_tensors='pt', return_special_tokens_mask=True)
            special = encoded.pop('special_tokens_mask').bool()
            mask = encoded['attention_mask'].bool()
            # Keep the exact tokenizer result; raw lossless prompt still goes into the packet.
            out = self.model(**encoded, output_hidden_states=False, use_cache=False)
            features = out.last_hidden_state.float().cpu()
            if features.shape[-1] != self.cfg['model']['text_dim']:
                raise ValueError('Text encoder hidden dimension does not match SANA caption_channels')
            content = mask & ~special
            if not content.any(): content = mask.clone()
            value = (features, mask.cpu(), content.cpu())
        self.cache[prompt] = value
        if len(self.cache) > 128: self.cache.popitem(last=False)
        return value

    def __call__(self, prompts, device):
        values = [self.one(p) for p in prompts]
        return tuple(torch.cat([v[k] for v in values], 0).to(device) for k in range(3))


class TinyBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        c = cfg['model']['latent_channels']; m = cfg['model']['channels']; d = cfg['model']['text_dim']
        self.encoder = frozen(nn.Sequential(nn.Conv2d(3,c,3,2,1), nn.Tanh()))
        self.aux_encoder = frozen(nn.Sequential(nn.Conv2d(3,c,3,2,1), nn.Tanh()))
        self.decoder = nn.Sequential(nn.Conv2d(c,3,3,1,1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Tanh())
        self.prompter = nn.Linear(m, d)
        self.input = nn.Conv2d(c,32,3,1,1)
        self.text_proj = nn.Linear(d,32)
        self.norm = nn.LayerNorm(32)
        self.adaln = SpatialAdaLN(32)
        self.attention = nn.MultiheadAttention(32,4,batch_first=True)
        self.output = nn.Conv2d(32,c,3,1,1)
        self.last_attention = None

    def analysis(self, x): return self.encoder(x), self.aux_encoder(x)
    def decode(self, z): return self.decoder(z)
    def condition(self, latent): return self.prompter(latent.flatten(2).transpose(1,2))

    def predict(self, noisy, text, mask, t, beta, log_lambda, spatial=True, capture=False):
        h = self.input(noisy); b,c,hh,ww = h.shape
        q = self.norm(h.flatten(2).transpose(1,2))
        if spatial: q = self.adaln(q,t,beta,log_lambda,(hh,ww))
        keys = self.text_proj(text.to(q.dtype))
        value, weights = self.attention(q, keys, keys, key_padding_mask=~mask,
                                        need_weights=capture, average_attn_weights=True)
        if capture: self.last_attention = weights.detach().transpose(1,2).reshape(b,-1,hh,ww)
        return self.output((q+value).transpose(1,2).reshape(b,c,hh,ww))


class SanaBackbone(nn.Module):
    def __init__(self, cfg, latent_core, initialize=True):
        super().__init__(); self.cfg = cfg; self._field = None; self._capture = False
        self.captured = []; self._handles = []
        from diffusers import AutoencoderDC, SanaTransformer2DModel
        from ELIC.elic_official import ELIC
        # Reuse only the architecture class; do not construct upstream Codec/losses.
        from .upstream_compat import LatentConditionAlignment, filter_supported_modules
        source = cfg['assets']['sana']; offline = cfg['assets']['local_files_only']
        opts = {'local_files_only': offline, 'revision': cfg['assets']['revision']}
        initial = cfg['assets']['ditic']
        if initial or not initialize:
            self.vae = AutoencoderDC.from_config(AutoencoderDC.load_config(source, subfolder='vae', **opts))
            self.dit = SanaTransformer2DModel.from_config(SanaTransformer2DModel.load_config(source, subfolder='transformer', **opts))
        else:
            self.vae = AutoencoderDC.from_pretrained(source, subfolder='vae', torch_dtype=torch.float32, **opts)
            self.dit = SanaTransformer2DModel.from_pretrained(source, subfolder='transformer', torch_dtype=torch.float32, **opts)
        elic = ELIC(); self.aux_encoder = elic.g_a
        self.prompter = LatentConditionAlignment(in_channels=320, embed_dim=2304,
                                                num_tokens=77, mode='mlp', use_clip_contrast=False)
        if initial and initialize:
            state = safe_torch_load(initial)
            if 'model' in state or 'ema' in state:
                raise ValueError('New SANA backend warm-start requires a MERGED upstream checkpoint; merge raw LoRA first')
            for prefix, module in [('codec.',latent_core),('vae.',self.vae),('DiT.',self.dit),
                                   ('aux_codec.',self.aux_encoder),('prompter.',self.prompter)]:
                part = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
                if not part: raise ValueError(f'Merged checkpoint missing component {prefix}')
                report = module.load_state_dict(part, strict=True)
        elif initialize:
            if not cfg['assets']['elic']: raise ValueError('Training from SANA requires assets.elic checkpoint')
            elic.load_state_dict(safe_torch_load(cfg['assets']['elic']), strict=True)
        frozen(self.aux_encoder); frozen(self.vae); frozen(self.dit)
        rank = cfg['model']['lora_rank']
        if rank:
            from peft import LoraConfig
            self.dit.add_adapter(LoraConfig(r=rank,lora_alpha=rank,init_lora_weights='gaussian',
                                           target_modules=['to_q','to_k','to_v','to_out.0']))
        vrank = cfg['model']['vae_lora_rank']
        if cfg['model']['train_vae_decoder'] and vrank:
            from peft import LoraConfig
            targets = filter_supported_modules(self.vae)
            if not targets: raise ValueError('No decoder LoRA targets matched the pinned AutoencoderDC')
            self.vae.add_adapter(LoraConfig(r=vrank,lora_alpha=vrank,init_lora_weights='gaussian',target_modules=targets))
        elif cfg['model']['train_vae_decoder']:
            self.vae.decoder.requires_grad_(True)
        self.install_spatial_hooks()

    def install_spatial_hooks(self):
        self.spatial_layers = nn.ModuleList()
        blocks = self.dit.transformer_blocks
        capture_indices = set(torch.linspace(0,len(blocks)-1,min(4,len(blocks))).round().int().tolist())
        for i, block in enumerate(blocks):
            for name in ('norm1','norm2'):
                norm = getattr(block,name)
                width = norm.normalized_shape[-1]
                adapter = SpatialAdaLN(width); self.spatial_layers.append(adapter)
                def norm_hook(module, args, out, adapter=adapter):
                    if self._field is None: return out
                    t,beta,lam,grid = self._field
                    return adapter(out,t,beta,lam,grid)
                self._handles.append(norm.register_forward_hook(norm_hook))
            if i in capture_indices:
                self._handles.append(block.attn2.register_forward_pre_hook(self._attention_hook, with_kwargs=True))
        self.last_attention = None

    def _attention_hook(self, attn, args, kwargs):
        if not self._capture: return
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        context = kwargs.get('encoder_hidden_states')
        if hidden is None or context is None: return
        with torch.no_grad():
            q = attn.to_q(hidden); k = attn.to_k(context)
            if attn.norm_q is not None: q = attn.norm_q(q)
            if attn.norm_k is not None: k = attn.norm_k(k)
            b,n,d = q.shape; h = attn.heads
            q = q.reshape(b,n,h,d//h).transpose(1,2).float()
            k = k.reshape(b,-1,h,d//h).transpose(1,2).float()
            score = q @ k.transpose(-2,-1) / (d//h)**.5
            mask = kwargs.get('attention_mask')
            if mask is not None:
                if mask.dtype == torch.bool: score = score.masked_fill(~mask[:,None,None,:], -1e4)
                else:
                    while mask.ndim < 4: mask = mask.unsqueeze(1)
                    score = score + mask.float()
            self.captured.append(score.softmax(-1).mean(1).detach())

    def train(self, mode=True):
        super().train(mode)
        self.aux_encoder.eval(); self.vae.encoder.eval()
        return self

    def analysis(self, x):
        z = self.vae.encode(x).latent * self.vae.config.scaling_factor
        return z, self.aux_encoder((x+1)/2)

    def decode(self, z):
        return self.vae.decode(z/self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)

    def condition(self, latent): return self.prompter(latent)

    def predict(self, noisy, text, mask, t, beta, log_lambda, spatial=True, capture=False):
        if self._field is not None: raise RuntimeError('Nested SANA forward is unsupported')
        if getattr(self.dit,'gradient_checkpointing',False):
            raise RuntimeError('Disable stock gradient checkpointing when using spatial norm hooks')
        patch = self.dit.config.patch_size
        grid = (noisy.shape[-2]//patch, noisy.shape[-1]//patch)
        self._field = (t,beta,log_lambda,grid) if spatial else None
        self._capture = capture; self.captured = []
        try:
            # Standard scalar base embedding plus our token-level field correction.
            timestep = t.mean((1,2,3)) * 1000 * self.dit.config.timestep_scale
            with torch.autocast(device_type=noisy.device.type, dtype=torch.bfloat16,
                                enabled=self.cfg['train']['amp'] and noisy.device.type == 'cuda'):
                result = self.dit(noisy, encoder_hidden_states=text, encoder_attention_mask=mask,
                                  timestep=timestep, return_dict=False)[0]
            result = result.float()
            if capture:
                if not self.captured: raise RuntimeError('No SANA cross-attention was captured')
                a = torch.stack(self.captured).mean(0)
                self.last_attention = a.transpose(1,2).reshape(a.shape[0],a.shape[-1],*grid)
            return result
        finally:
            self._field = None; self._capture = False; self.captured = []
