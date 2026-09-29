"""Manifest data interface with identical image preprocessing and map-cache checks."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

EXTENSIONS={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}


def read_image(path):
    with Image.open(path) as image:
        image=ImageOps.exif_transpose(image).convert('RGB')
        array=np.array(image,dtype=np.float32)/255.
    return torch.from_numpy(array).permute(2,0,1)*2-1


def save_image(tensor,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=((tensor.detach().cpu().squeeze(0).clamp(-1,1)+1)*127.5).round().byte()
    Image.fromarray(value.permute(1,2,0).numpy()).save(path)


def training_image(path,crop):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        w,h=im.size
        if min(w,h)<crop:
            scale=crop/min(w,h); im=im.resize((max(crop,round(w*scale)),max(crop,round(h*scale))),Image.Resampling.BICUBIC)
            w,h=im.size
        left=(w-crop)//2;top=(h-crop)//2
        im=im.crop((left,top,left+crop,top+crop))
        a=np.array(im,dtype=np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1


def sample_hash(image,prompt,crop):
    h=hashlib.sha256();h.update(image.contiguous().numpy().tobytes())
    h.update(prompt.encode());h.update(str(crop).encode());h.update(b'center-resize-v1')
    return h.hexdigest()


def read_manifest(path):
    path=Path(path).resolve();rows=[];seen=set()
    with path.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            row=json.loads(line)
            for key in ('id','image','prompt','prompt_kind'):
                if key not in row:raise ValueError(f'{path}:{line_no}: missing {key}')
            if not isinstance(row['prompt'],str):raise ValueError('prompt must be a string, including for explicit empty prompts')
            if row['prompt_kind'] not in ('true_prompt','caption','empty'):
                raise ValueError('prompt_kind must distinguish true prompts, captions and empty conditions')
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',str(row['id'])):raise ValueError('Unsafe or invalid sample id')
            if row['id'] in seen:raise ValueError(f'Duplicate sample id: {row["id"]}')
            seen.add(row['id'])
            for key in ('image','maps'):
                if row.get(key):
                    p=Path(row[key]);row[key]=str(p if p.is_absolute() else path.parent/p)
            if not Path(row['image']).is_file():raise FileNotFoundError(row['image'])
            rows.append(row)
    if not rows:raise ValueError('Manifest is empty')
    return rows


def write_manifest(rows,path):
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    import os
    with path.open('w',encoding='utf-8') as f:
        for row in rows:
            row=dict(row)
            for key in ('image','maps'):
                if row.get(key):row[key]=os.path.relpath(Path(row[key]).resolve(),path.parent)
            f.write(json.dumps(row,ensure_ascii=False)+'\n')


class ManifestDataset(Dataset):
    def __init__(self,path,crop,require_maps=False,lambda_rd=None):
        self.rows=read_manifest(path);self.crop=crop;self.require_maps=require_maps;self.lambda_rd=lambda_rd
        if require_maps and any(not r.get('maps') for r in self.rows):
            raise ValueError('This stage requires cached maps for every image; run cache-maps first')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];x=training_image(row['image'],self.crop)
        sample={'image':x,'prompt':row['prompt'],'id':row['id']}
        if row.get('maps'):
            with np.load(row['maps'],allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata']))
                if metadata['sample_sha256']!=sample_hash(x,row['prompt'],self.crop):
                    raise ValueError(f'Stale/misaligned map cache for {row["id"]}')
                if self.lambda_rd is not None and abs(metadata['lambda_rd']-self.lambda_rd)>1e-9:
                    raise ValueError('Teacher maps have a different rate operating point; regenerate per lambda')
                sample['maps']={k:torch.from_numpy(archive[k].astype('float32')) for k in ('S','c','e')}
        return sample


def import_folder(root,output,metadata=None,prompt_kind='empty',prompt_column='prompt',image_column='image'):
    root=Path(root).resolve();lookup={}
    if metadata:
        meta=Path(metadata)
        if meta.suffix=='.csv':
            with meta.open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
        else:
            with meta.open(encoding='utf-8') as f: records=[json.loads(x) for x in f if x.strip()]
        for record in records:lookup[str(record[image_column])]=str(record[prompt_column])
    rows=[]
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in EXTENSIONS:continue
        relative=path.relative_to(root).as_posix()
        prompt=lookup.get(relative,lookup.get(path.name))
        if prompt is None and prompt_kind!='empty':raise ValueError(f'Missing prompt metadata: {relative}')
        rows.append({'id':hashlib.sha256(relative.encode()).hexdigest()[:20],'image':str(path),
                     'prompt':prompt or '', 'prompt_kind':prompt_kind,'generator':'unknown'})
    if not rows:raise ValueError('No images found')
    write_manifest(rows,output);return len(rows)


def split_manifest(source,outdir,validation=.05,test=.05,seed=2026,holdout_generator=None):
    if not 0<=validation<1 or not 0<=test<1 or validation+test>=1:raise ValueError('Invalid split fractions')
    rows=read_manifest(source);parent=list(range(len(rows)));lookup={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for index,row in enumerate(rows):
        image_hash=hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
        row['image_sha256']=image_hash
        prompt=' '.join(row['prompt'].split()).casefold()
        keys=['image:'+image_hash]
        if prompt:keys.append('prompt:'+prompt)
        for key in keys:
            if key in lookup:union(index,lookup[key])
            else:lookup[key]=index
    components={}
    for index,row in enumerate(rows):components.setdefault(find(index),[]).append(row)
    groups={'train':[],'validation':[],'test':[]}
    for component in components.values():
        key=min(r['image_sha256'] for r in component)
        value=int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16],16)/2**64
        held=holdout_generator and any(r.get('generator')==holdout_generator for r in component)
        name='test' if held or value<test else ('validation' if value<test+validation else 'train')
        groups[name].extend(component)
    outdir=Path(outdir)
    for name,part in groups.items():
        if part:write_manifest(part,outdir/f'{name}.jsonl')
    return {k:len(v) for k,v in groups.items()}
