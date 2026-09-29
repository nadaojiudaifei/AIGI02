"""Checkpoint-compatible helpers from pinned DiT-IC (see NOTICE_AIGI02.md).

The production backbone needs only the MLP latent prompt alignment, not the
original Codec constructor or its legacy GAN/metric import side effects.
Names/shapes below match the upstream prompter state dictionary exactly.
"""
import re
import torch
from torch import nn
import torch.nn.functional as F


class LatentConditionAlignment(nn.Module):
    def __init__(self,in_channels=320,embed_dim=2304,num_tokens=77,mode='mlp',use_clip_contrast=False):
        super().__init__()
        if mode!='mlp' or use_clip_contrast:raise ValueError('The pinned merged checkpoint uses MLP alignment without CLIP contrast')
        self.fixed_tokens=48
        self.pattern=nn.Parameter(torch.randn(1,num_tokens-self.fixed_tokens,in_channels))
        self.align=nn.Sequential(nn.Linear(in_channels,embed_dim),nn.SiLU(),nn.Linear(embed_dim,embed_dim),nn.SiLU(),nn.Linear(embed_dim,embed_dim))
    def forward(self,latent):
        x=F.adaptive_avg_pool1d(latent.flatten(2),self.fixed_tokens).transpose(1,2)
        return self.align(torch.cat([x,self.pattern.expand(latent.shape[0],-1,-1)],dim=1))


def filter_supported_modules(model):
    pattern=re.compile(r'^decoder\..*(conv1|conv2|conv_in|conv_shortcut|conv_inverted|conv_point|to_k|to_q|to_v|to_out\.0)$')
    return [name for name,module in model.named_modules() if pattern.match(name) and isinstance(module,(nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear))]
