import pytest
import torch
from aigi02.math_ops import *
from aigi02.config import tiny_config,load_config,architecture_id


def test_information_quadrature_and_signed_e():
    nodes=torch.tensor([-2.,0.,2.]);a=torch.ones(3,2,1,2,2)*3;b=torch.ones_like(a)
    c,e=information_scores(a,b,nodes)
    assert torch.allclose(c,torch.full_like(c,6))
    assert torch.allclose(e,torch.full_like(e,-4))

@pytest.mark.parametrize('nodes',[torch.tensor([1.]),torch.tensor([1.,0.]),torch.tensor([1.,1.])])
def test_bad_quadrature(nodes):
    with pytest.raises(ValueError):information_scores(torch.zeros(len(nodes),1,1,2,2),torch.zeros(len(nodes),1,1,2,2),nodes)


def test_waterfill_total_budget():
    torch.manual_seed(2);w=torch.rand(3,1,4,5)+.1;c=torch.rand_like(w)*4;r=torch.rand_like(w)*3
    target=waterfill_target(w,c,r)
    assert torch.allclose(target.sum((1,2,3)),r.sum((1,2,3)),atol=1e-4)
    assert (target>=0).all()


def test_waterfill_zero_budget():
    w=torch.rand(2,1,2,3)+.1;c=torch.rand_like(w);r=torch.zeros_like(w)
    assert waterfill_target(w,c,r).abs().max()<1e-5


def test_snr_closed_form():
    mu=torch.ones(1,4,2,2)*2;sigma=torch.ones_like(mu)*3
    assert torch.allclose(local_snr(mu,sigma),torch.full((1,1,2,2),156.))


def test_semantic_bias_sign():
    snr=torch.ones(1,1,2,2);w=torch.tensor([[[[.9,.1],[.9,.1]]]])
    no=snr_timestep(snr,w,0.);yes=snr_timestep(snr,w,2.)
    assert yes[0,0,0,0]<no[0,0,0,0]
    assert yes[0,0,0,1]>no[0,0,0,1]


def test_cell_uses_mean_offset_and_detaches_center():
    x=torch.tensor([1.2,2.3],requires_grad=True);q=torch.tensor([1.2,1.2],requires_grad=True);m=torch.ones(2,requires_grad=True)
    loss=cell_loss(x,m,q);loss.backward()
    assert torch.allclose(loss,torch.tensor(.3))
    assert q.grad is None and m.grad is None and x.grad is not None


def test_spatial_zero_init_identity():
    layer=SpatialAdaLN(8);x=torch.randn(2,6,8);t=torch.rand(2,1,2,3)
    assert torch.equal(layer(x,t,torch.ones(2),torch.ones(2),(2,3)),x)


def test_ugaq_bounds_and_gradients():
    module=SemanticUGAQ(8);y=torch.randn(2,8,3,3,requires_grad=True);w=torch.rand(2,1,3,3)*.9+.1
    maps={'S':torch.rand_like(w),'c':torch.ones_like(w),'e':-torch.ones_like(w)}
    m=module(y,torch.zeros_like(y),maps,w,torch.ones(2));m.mean().backward()
    assert m.min()>=1 and m.max()<=32 and y.grad is not None


def test_normalize_constant_is_finite():
    assert torch.equal(normalize_map(torch.ones(1,1,4,4)),torch.zeros(1,1,4,4))

@pytest.mark.parametrize('changes',[{'unknown':1},{'rate':{'lambda_rd':0}}, {'rate':{'lambda_rd':float('nan')}},
                                  {'model':{'channels':15}},{'train':{'crop':31}},{'teacher':{'drop_text':2}}])
def test_config_validation(changes):
    cfg=tiny_config()
    from aigi02.config import deep_merge
    with pytest.raises(ValueError):load_config(deep_merge(cfg,changes))


def test_architecture_id_ignores_paths_beta():
    a=tiny_config();b=tiny_config();b['device']='cuda';b['assets']['sana']='/other/path';b['rate']['beta']=0
    assert architecture_id(a)==architecture_id(b)
    b['features']['cell']=False
    assert architecture_id(a)!=architecture_id(b)
