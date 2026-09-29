"""Configuration is explicit, serializable and checked before model allocation."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import yaml

DEFAULT = {
 'backend': 'sana', 'method': 'idea2', 'seed': 2026, 'device': 'cuda',
 'assets': {'sana': 'SANA', 'ditic': None, 'elic': None, 'text': None,
            'local_files_only': True, 'revision': None,
            'discriminator': 'facebook/dinov2-with-registers-small'},
 'model': {'channels': 320, 'latent_channels': 32, 'text_dim': 2304,
           'pad_multiple': 256, 'lora_rank': 16, 'vae_lora_rank': 8,
           'max_tokens': 77, 'train_vae_decoder': True},
 'features': {'token_weights': True, 'texture': True, 'importance': True, 'conditional_score': True,
              'correction': True, 'prompt_entropy': True, 'prompt_decoder': True,
              'field': True, 'semantic_bias': True, 'field_residual': True,
              'spatial_adaln': True, 'waterfill': True, 'cell': True,
              'distillation': True, 'realism': False},
 'rate': {'lambda_rd': 256., 'beta': 1., 'beta_max': 2., 'prompt_mode': 'included'},
 'loss': {'perceptual': 1., 'alignment': .1, 'cell': .05, 'waterfill': .05,
          'importance_head': .1, 'map': 1., 'distill': 1., 'realism': .01},
 'teacher': {'log_snr_min': -6., 'log_snr_max': 4., 'nodes': 8,
             'noise_samples': 2, 'steps': 4, 'drop_text': .1, 'drop_latent': .1},
 'train': {'stage': 'codec', 'manifest': None, 'val_manifest': None, 'val_max_images': 16, 'output': 'aigi02_runs/idea2',
           'crop': 256, 'batch_size': 1, 'steps': 50000, 'lr': 1e-4, 'aux_lr': 1e-3,
           'num_workers': 4, 'accumulation': 1, 'save_every': 1000, 'log_every': 20,
           'clip_grad': 1., 'amp': False, 'resume': None, 'init': None, 'teacher_checkpoint': None,
           'trainable': 'all', 'map_source': 'distilled'},
}


def deep_merge(base: dict, changes: dict, strict: bool = True) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if strict and key not in result:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value, strict)
        else: result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | dict, overrides: dict | None = None) -> dict:
    if isinstance(path, dict): raw = path
    else:
        with Path(path).open(encoding='utf-8') as f: raw = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT, raw)
    if overrides: cfg = deep_merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(c: dict) -> None:
    if c['backend'] not in ('sana', 'tiny'): raise ValueError('backend must be sana or tiny')
    if c['method'] not in ('idea1', 'idea2', 'texture_only', 'no_adaptation'):
        raise ValueError('Unknown method; the original baseline has a separate command')
    if not all(math.isfinite(float(v)) for v in (c['rate']['lambda_rd'],c['rate']['beta'],c['rate']['beta_max'])): raise ValueError('Rate controls must be finite')
    if c['rate']['lambda_rd'] <= 0 or c['rate']['beta'] < 0 or c['rate']['beta_max'] < c['rate']['beta']: raise ValueError('Invalid rate controls')
    for key in ('drop_text','drop_latent'):
        if not 0 <= c['teacher'][key] <= 1: raise ValueError('Drop probabilities must lie in [0,1]')
    if c['rate']['prompt_mode'] not in ('included', 'free'): raise ValueError('Invalid prompt mode')
    if c['model']['channels'] % 4: raise ValueError('Main channels must be divisible by four')
    if c['model']['pad_multiple'] < 16: raise ValueError('pad_multiple must be at least 16')
    if c['backend'] == 'sana':
        if c['model']['channels'] != 320 or c['model']['latent_channels'] != 32:
            raise ValueError('Pinned DiT-IC production transforms require 320 main and 32 VAE channels')
        if c['model']['pad_multiple'] % 256: raise ValueError('SANA codec images must pad to a multiple of 256')
        if c['model']['text_dim'] != 2304: raise ValueError('Pinned SANA text channels must be 2304')
    if c['teacher']['nodes'] < 2 or c['teacher']['noise_samples'] < 1 or c['teacher']['steps'] < 1:
        raise ValueError('Invalid teacher integration settings')
    if c['teacher']['log_snr_min'] >= c['teacher']['log_snr_max']: raise ValueError('Invalid log-SNR range')
    if c['train']['stage'] not in ('maps', 'codec', 'field_pretrain', 'flow_teacher', 'distill', 'gan'):
        raise ValueError('Unknown training stage')
    if c['train']['map_source'] not in ('distilled', 'oracle'): raise ValueError('Invalid map source')
    if c['train']['trainable'] not in ('all', 'maps', 'allocation', 'entropy', 'field', 'decoder'):
        raise ValueError('Invalid component training scope')
    for k in ('steps', 'batch_size', 'accumulation', 'save_every', 'log_every', 'val_max_images'):
        if c['train'][k] < 1: raise ValueError(f'train.{k} must be positive')
    if c['train']['crop'] % c['model']['pad_multiple']:
        raise ValueError('Training crop must be divisible by pad_multiple')


def architecture_id(c: dict) -> str:
    """Paths, device and beta are not architecture identities; weights are checked separately."""
    identity = {k: c[k] for k in ('backend', 'method', 'model', 'features')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def tiny_config(method: str = 'idea2') -> dict:
    return load_config({'backend': 'tiny', 'device': 'cpu', 'method': method,
      'model': {'channels': 16, 'latent_channels': 8, 'text_dim': 32, 'pad_multiple': 16,
                'lora_rank': 0, 'vae_lora_rank': 0, 'max_tokens': 32},
      'loss': {'perceptual': 0.},
      'train': {'crop': 32, 'num_workers': 0, 'steps': 2, 'save_every': 1, 'log_every': 1}})
