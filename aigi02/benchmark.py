"""Shared measurement harness. Original DiT-IC calls remain unmodified."""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import read_manifest,read_image,save_image
from .training import file_sha256,seed_everything


class BaselinePacket:
    FIXED=struct.Struct('>4sHHI16s')
    def __init__(self,size,seed,model_key,payload):self.size=size;self.seed=seed;self.model_key=model_key;self.payload=payload
    def pack(self):
        body=self.FIXED.pack(b'DIC0',*self.size,self.seed,bytes.fromhex(self.model_key))+self.payload
        return body+hashlib.sha256(body).digest()[:8]
    @classmethod
    def unpack(cls,data):
        if len(data)<cls.FIXED.size+28 or hashlib.sha256(data[:-8]).digest()[:8]!=data[-8:]:raise ValueError('Invalid baseline stream')
        magic,h,w,seed,key=cls.FIXED.unpack(data[:cls.FIXED.size])
        if magic!=b'DIC0' or h==0 or w==0:raise ValueError('Invalid baseline header')
        return cls((h,w),seed,key.hex(),data[cls.FIXED.size:-8])
    def write(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(self.pack())
    def rates(self):
        pixels=self.size[0]*self.size[1]
        return {'bytes_total':len(self.pack()),'bytes_original_ditic_container':len(self.payload),
                'bytes_prompt':0,'bpp_total':len(self.pack())*8/pixels,
                'bpp_original_ditic_container':len(self.payload)*8/pixels,'bpp_prompt_payload':0.}


class OriginalBaseline:
    """Pinned original architecture, forward, entropy coder and scheduler.

    A small exterior frame supplies image dimensions, seed, model identity and
    integrity for independent decoding. Both original and framed rates are saved.
    """
    def __init__(self,config,checkpoint,device='cuda',sana_path=None):
        import yaml
        from models.DiT_IC import Codec
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('The pinned original baseline is evaluated on CUDA; it is not the tiny test model')
        with open(config) as f:cfg=yaml.safe_load(f)
        params=cfg['model']['params'];params['codec_path']=str(checkpoint);params['training']=False
        if sana_path:params['dit_path']=sana_path
        self.device=torch.device(device);self.cfg=cfg
        self.model=Codec(device=str(device),**params).to(device).eval();self.model.codec.update(force=True)
        self.key=hashlib.sha256((file_sha256(checkpoint)+json.dumps(cfg,sort_keys=True)).encode()).hexdigest()[:32]

    @torch.no_grad()
    def compress(self,x,prompt='',seed=2026,prompt_mode=None,oracle=None):
        from eval.compress_utils import write_body
        x=x.to(self.device);h,w=x.shape[-2:];ph=(-h)%256;pw=(-w)%256
        if ph>=h or pw>=w:raise ValueError('Original reflect-padding rule cannot handle this small image')
        padded=F.pad(x,(0,pw,0,ph),mode='reflect')
        seed_everything(seed);out=self.model.compress(padded);buffer=io.BytesIO()
        write_body(buffer,out['shape'],out['strings'])
        return BaselinePacket((h,w),seed,self.key,buffer.getvalue()),{}

    @torch.no_grad()
    def decompress(self,packet,external_prompt=None,beta=None):
        from eval.compress_utils import read_body
        if packet.model_key!=self.key:raise ValueError('Baseline checkpoint/configuration mismatch')
        if beta is not None:raise ValueError('The original baseline does not have a beta control')
        seed_everything(packet.seed)
        stream=io.BytesIO(packet.payload);strings,shape=read_body(stream)
        if stream.tell()!=len(packet.payload):raise ValueError('Trailing original bitstream bytes')
        image=self.model.decompress(strings,shape)
        return image[...,:packet.size[0],:packet.size[1]],{}


def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def run_roundtrip(model,manifest,output,seed=2026,beta=None,prompt_mode=None,oracle_maps=False):
    rows=read_manifest(manifest);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    (root/'rec').mkdir(exist_ok=True);(root/'bin').mkdir(exist_ok=True);(root/'maps').mkdir(exist_ok=True)
    record_file=root/'rates.jsonl'
    if record_file.exists():raise FileExistsError('Refusing to mix results with an existing run; choose a new output directory')
    with record_file.open('w',encoding='utf-8') as f:
        for index,row in enumerate(rows):
            image=read_image(row['image']).unsqueeze(0).to(model.device)
            if torch.device(model.device).type=='cuda':torch.cuda.reset_peak_memory_stats(model.device)
            oracle=None
            if oracle_maps:
                if not row.get('maps'):raise ValueError('Oracle inference requires full-resolution cached maps')
                from .data import sample_hash
                with np.load(row['maps'],allow_pickle=False) as archive:
                    meta=json.loads(str(archive['metadata']))
                    if meta.get('preprocessing')!='full-pad-v1' or meta['sample_sha256']!=sample_hash(image[0].cpu(),row['prompt'],0):
                        raise ValueError('Oracle inference cache is stale or uses training crops')
                    if abs(meta['lambda_rd']-model.cfg['rate']['lambda_rd'])>1e-9:raise ValueError('Oracle operating point mismatch')
                    oracle={k:torch.from_numpy(archive[k]).unsqueeze(0) for k in ('S','c','e')}
            synchronize(model.device);start=time.perf_counter()
            packet,diagnostics=model.compress(image,row['prompt'],seed=seed,prompt_mode=prompt_mode,oracle=oracle)
            synchronize(model.device);encoded=time.perf_counter()
            # Reload the serialized representation. No encoder-side object/tensor is a decoder input.
            raw=packet.pack();restored=type(packet).unpack(raw)
            reconstructed,fields=model.decompress(restored,external_prompt=row['prompt'],beta=beta)
            synchronize(model.device);decoded=time.perf_counter()
            packet.write(root/'bin'/f'{row["id"]}.bin');save_image(reconstructed,root/'rec'/f'{row["id"]}.png')
            diagnostics.update(fields)
            if diagnostics:np.savez_compressed(root/'maps'/f'{row["id"]}.npz',**{k:v.cpu().numpy() for k,v in diagnostics.items()})
            record={'id':row['id'],'prompt_kind':row['prompt_kind'],'source_image':row['image'],
                    'map_source':'oracle' if oracle_maps else 'distilled_or_baseline',
                    'oracle_compute_included_in_encode_ms':False if oracle_maps else None,
                    'height':image.shape[-2],'width':image.shape[-1],
                    'encode_ms':(encoded-start)*1000,'decode_ms':(decoded-encoded)*1000,
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated(model.device) if torch.device(model.device).type=='cuda' else 0,
                    **packet.rates()}
            f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
            print(json.dumps({'completed':index+1,'total':len(rows),'id':row['id'],'bpp_total':record['bpp_total']}),flush=True)
    return str(record_file)
