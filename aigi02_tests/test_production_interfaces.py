"""Optional CPU tests of real libraries; these do NOT download pretrained weights."""
import pytest
import torch
from torch import nn
from aigi02.config import tiny_config
from aigi02.backbone import SanaBackbone
from aigi02.latent import SemanticLatent


def test_actual_compressai_rans_roundtrip():
    pytest.importorskip('compressai')
    from compressai.entropy_models import EntropyBottleneck,GaussianConditional
    torch.manual_seed(42);torch.set_num_threads(1)
    cfg=tiny_config();model=SemanticLatent(cfg).eval()
    model.core.entropy_bottleneck=EntropyBottleneck(8)
    model.core.gaussian_conditional=GaussianConditional(None)
    table=torch.exp(torch.linspace(torch.tensor(.11).log(),torch.tensor(256.).log(),64))
    model.core.gaussian_conditional.update_scale_table(table,force=True)
    model.core.entropy_bottleneck.update(force=True)
    text=torch.randn(1,32,32);mask=torch.ones(1,32,dtype=torch.bool);lam=torch.tensor([5.5])
    z=torch.randn(1,8,16,16);aux=torch.randn_like(z)
    with torch.no_grad():
        streams,shape,enc=model.compress(z,aux,text,mask,lam)
        dec=model.decompress(streams,shape,text,mask)
    assert torch.equal(enc['qhat'],dec['qhat'])
    assert torch.equal(enc['yhat'],dec['yhat'])
    assert len(streams)==5 and all(isinstance(s,bytes) for s in streams)


def test_real_sana_spatial_hooks_forward_backward():
    pytest.importorskip('diffusers')
    from diffusers import SanaTransformer2DModel
    torch.manual_seed(32);torch.set_num_threads(1)
    model=SanaBackbone.__new__(SanaBackbone);nn.Module.__init__(model)
    model.cfg=tiny_config();model._field=None;model._capture=False;model.captured=[];model._handles=[]
    model.dit=SanaTransformer2DModel(in_channels=8,out_channels=8,num_attention_heads=2,
        attention_head_dim=16,num_layers=2,num_cross_attention_heads=2,cross_attention_head_dim=16,
        cross_attention_dim=32,caption_channels=32,mlp_ratio=2,sample_size=4,patch_size=1)
    model.install_spatial_hooks()
    x=torch.randn(1,8,4,4,requires_grad=True);text=torch.randn(1,7,32);mask=torch.ones(1,7,dtype=torch.bool)
    t=torch.rand(1,1,4,4);beta=torch.tensor([1.]);lam=torch.tensor([5.])
    output=model.predict(x,text,mask,t,beta,lam,spatial=True,capture=True)
    assert output.shape==x.shape and model.last_attention.shape==(1,7,4,4)
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spatial_layers.parameters())
    assert model._field is None and model._capture is False
