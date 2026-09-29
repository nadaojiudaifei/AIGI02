#!/usr/bin/env python
"""User-run pretrained GPU acceptance. This file is not a completed test report."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from aigi02.config import load_config
from aigi02.model import AIGIModel
from aigi02.data import training_image,save_image
from aigi02.training import seed_everything,load_model,file_sha256
from aigi02.bitstream import Packet


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--image',required=True);p.add_argument('--prompt',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=load_config(a.config)
    if cfg['backend']!='sana' or not torch.cuda.is_available():raise RuntimeError('This acceptance test requires the production SANA backend and CUDA')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'ACCEPTANCE.json').exists():raise FileExistsError('Choose a fresh acceptance output directory')
    seed_everything(cfg['seed']);torch.set_num_threads(1)
    model=AIGIModel(cfg).to('cuda');model.train()
    x=training_image(a.image,cfg['train']['crop']).unsqueeze(0).cuda()
    out=model(x,[a.prompt]);loss=out['bpp_estimate']+(x-out['xhat']).square().mean()+model.quantization_cell_loss(out)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise AssertionError('Missing or non-finite production gradients')
    model.zero_grad(set_to_none=True);del out;model.eval()
    checkpoint=root/'acceptance.pt';torch.save({'format':'aigi02-checkpoint-v1','config':cfg,'model':model.state_dict()},checkpoint)
    model.checkpoint_id=file_sha256(checkpoint)
    packet,_=model.compress(x,a.prompt);packet.write(root/'image.bin')
    recon,_=model.decompress(Packet.read(root/'image.bin'));save_image(recon,root/'first.png')
    del model;torch.cuda.empty_cache()
    loaded,_=load_model(checkpoint,'cuda');independent,_=loaded.decompress(Packet.read(root/'image.bin'))
    save_image(independent,root/'independent.png')
    exact=torch.equal(recon,independent)
    if not exact:raise AssertionError('Reloaded checkpoint decoder differs under this execution profile')
    report={'pretrained_gpu_forward_backward':True,'real_rans_roundtrip':True,'reloaded_decoder_tensor_exact':exact,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'loss':float(loss.detach()),
            'trainable_gradient_tensors':len(gradients),'rates':packet.rates(),
            'paper_reproduction':False,'rate_distortion_gain_verified':False}
    (root/'ACCEPTANCE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
