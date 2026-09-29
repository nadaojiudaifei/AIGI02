import pytest
import torch
from aigi02.bitstream import Packet,bounded_uncompress,varint,read_varint
from aigi02.reference_coder import encode_symbols,decode_symbols
from aigi02.latent import masks_four,squeeze,unsqueeze

@pytest.mark.parametrize('count',[0,1,7,128,1000])
def test_reference_arithmetic_roundtrip(count):
    g=torch.Generator().manual_seed(19)
    values=torch.randint(-500,500,(count,),generator=g).tolist()
    indexes=torch.randint(0,64,(count,),generator=g).tolist()
    encoded=encode_symbols(values,indexes)
    assert decode_symbols(encoded,indexes)==values


def test_arithmetic_escape_extremes():
    values=[-2**31,2**31-1,0,-255,255]
    assert decode_symbols(encode_symbols(values,[40]*5),[40]*5)==values


def test_arithmetic_corrupt_framing():
    with pytest.raises(ValueError):decode_symbols(b'bad',[1])
    blob=encode_symbols([1,2],[1,1])
    with pytest.raises(ValueError):decode_symbols(blob,[1])

@pytest.mark.parametrize('n',[0,127,128,10000,2**28])
def test_varint(n):assert read_varint(varint(n),0)==(n,len(varint(n)))


def test_bounded_prompt_decompression():
    import zlib
    assert bounded_uncompress(zlib.compress(b'abc'),3)==b'abc'
    with pytest.raises(ValueError):bounded_uncompress(zlib.compress(b'a'*10000),100)
    with pytest.raises(ValueError):bounded_uncompress(zlib.compress(b'a')+b'garbage',10)


def test_four_groups_are_partition():
    x=torch.randn(2,16,7,9);masks=masks_four(x)
    assert torch.equal(sum(masks),torch.ones_like(x))
    restored=sum(unsqueeze(squeeze(x,m),m) for m in masks)
    assert torch.equal(restored,x)

@pytest.mark.parametrize('shape',[(16,16),(31,27),(32,48),(1,1)])
def test_packet_independent_roundtrip(model,shape):
    x=torch.rand(1,3,*shape)*2-1
    packet,_=model.compress(x,'a red ball')
    stored=Packet.unpack(packet.pack());y,field=model.decompress(stored)
    y2,_=model.decompress(stored)
    assert y.shape==x.shape and torch.isfinite(y).all() and torch.equal(y,y2)
    assert packet.rates()['bytes_header_and_integrity']>=64
    assert packet.pack()==stored.pack()
    assert not {'m','w','S','c','e','original_image'}&stored.header.keys()


def test_prompt_free_contract(model):
    packet,_=model.compress(torch.zeros(1,3,32,32),'exact prompt',prompt_mode='free')
    decoded=Packet.unpack(packet.pack())
    assert decoded.segments[0]==b''
    with pytest.raises(ValueError):model.decompress(decoded)
    with pytest.raises(ValueError):model.decompress(decoded,'wrong prompt')
    model.decompress(decoded,'exact prompt')


def test_packet_corruption_rejected(model):
    packet,_=model.compress(torch.zeros(1,3,32,32),'x');raw=bytearray(packet.pack());raw[-10]^=1
    with pytest.raises(ValueError):Packet.unpack(bytes(raw))
    with pytest.raises(ValueError):Packet.unpack(packet.pack()[:-1])


def test_model_mismatch_rejected(model):
    packet,_=model.compress(torch.zeros(1,3,32,32),'x')
    with torch.no_grad():next(model.parameters()).add_(.01)
    with pytest.raises(ValueError):model.decompress(Packet.unpack(packet.pack()))


def test_beta_changes_decoder_not_stream(model):
    x=torch.rand(1,3,32,32)*2-1;p,_=model.compress(x,'blue and red')
    a,_=model.decompress(p,beta=0);b,_=model.decompress(p,beta=2)
    assert not torch.equal(a,b)
    q,_=model.compress(x,'blue and red');assert p.pack()==q.pack()


def test_encoding_matches_differentiable_symbols(model):
    x=torch.rand(1,3,32,32)*2-1;text,mask,_=model.text(['symbol test'],'cpu');lam=torch.tensor([1.])
    with torch.no_grad():
        z,aux=model.backbone.analysis(x)
        forward=model.latent(z,aux,text,mask,lam)
        streams,shape,encoded=model.latent.compress(z,aux,text,mask,lam)
        decoded=model.latent.decompress(streams,shape,text,mask)
    assert torch.equal(encoded['qhat'],decoded['qhat'])
    assert torch.equal(encoded['yhat'],decoded['yhat'])
    assert torch.allclose(forward['qhat'],encoded['qhat'],atol=1e-6)
