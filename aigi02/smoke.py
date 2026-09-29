"""Mechanics-only smoke run, never a scientific codec benchmark."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import torch
import yaml
from .config import tiny_config
from .data import save_image,write_manifest
from .training import run_training,load_model,seed_everything
from .teacher import cache_maps
from .benchmark import run_roundtrip
from .metrics import evaluate


def run_smoke(output):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'train').exists():raise FileExistsError('Choose a fresh smoke output directory')
    seed_everything(2026)
    rows=[]
    for i in range(2):
        axis=torch.linspace(-1,1,32);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        image=torch.stack((xx,yy,torch.sin((i+1)*3*xx)*torch.cos(3*yy)))
        path=root/f'input_{i}.png';save_image(image,path)
        rows.append({'id':f'image{i}','image':str(path),'prompt':f'colored geometric pattern {i}',
                     'prompt_kind':'caption','generator':'procedural-test-only'})
    manifest=root/'input.jsonl';write_manifest(rows,manifest)
    cfg=tiny_config();cfg['train'].update(manifest=str(manifest),output=str(root/'train'))
    checkpoint=run_training(cfg);model,_=load_model(checkpoint,'cpu')
    rates=run_roundtrip(model,manifest,root/'roundtrip')
    summary=evaluate(manifest,root/'roundtrip/rec',root/'metrics',['psnr'],'cpu',False,rates)
    # A new interpreter has only checkpoint+bitstream, not original image tensors/maps.
    fresh=root/'independent.png'
    subprocess.run([sys.executable,'-m','aigi02','decode','--checkpoint',checkpoint,
                    '--input',str(root/'roundtrip/bin/image0.bin'),'--output',str(fresh),'--device','cpu'],check=True)
    same=fresh.read_bytes()==(root/'roundtrip/rec/image0.png').read_bytes()
    if not same:raise AssertionError('Independent decoding differs from roundtrip decoding')
    report={'backend':'tiny-reference-arithmetic-test-only','train_steps':cfg['train']['steps'],
            'images':2,'independent_decode_png_bit_exact':same,'measured_metrics':summary,
            'not_validated':['pretrained SANA GPU training','original DiT-IC reproduction','rate-distortion improvements']}
    (root/'SMOKE_REPORT.json').write_text(json.dumps(report,indent=2))
    return report
