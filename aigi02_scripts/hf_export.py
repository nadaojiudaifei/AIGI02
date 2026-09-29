#!/usr/bin/env python3
"""Standalone HF exporter; launch with python -I to avoid datasets name shadowing."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from PIL import Image,ImageOps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--name');p.add_argument('--split',default='train');p.add_argument('--revision')
    p.add_argument('--limit',type=int,default=1000,help='0 explicitly exports the full split')
    p.add_argument('--image-column',default='image');p.add_argument('--prompt-column')
    p.add_argument('--prompt-kind',choices=['true_prompt','caption','empty'],default='empty')
    p.add_argument('--generator',default='unknown');p.add_argument('--local-disk',action='store_true')
    a=p.parse_args()
    if a.limit<0:p.error('--limit must be nonnegative')
    if a.prompt_kind!='empty' and not a.prompt_column:p.error('A prompt/caption column must be specified')
    from datasets import load_dataset,load_from_disk,DatasetDict
    if a.local_disk:
        dataset=load_from_disk(a.source)
        if isinstance(dataset,DatasetDict):dataset=dataset[a.split]
    else:
        dataset=load_dataset(a.source,name=a.name,split=a.split,revision=a.revision,streaming=True)
    root=Path(a.output).resolve();images=root/'images';images.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.jsonl'
    if manifest.exists():raise FileExistsError('Export destination already has a manifest; use a new output directory')
    count=0
    with manifest.open('w',encoding='utf-8') as f:
        for index,item in enumerate(dataset):
            if a.limit and count>=a.limit:break
            if a.image_column not in item:raise KeyError(f'Missing image column {a.image_column}; actual columns: {list(item)}')
            source=item[a.image_column]
            if isinstance(source,Image.Image):image=source
            elif isinstance(source,dict) and source.get('bytes'):image=Image.open(io.BytesIO(source['bytes']))
            elif isinstance(source,dict) and source.get('path'):image=Image.open(source['path'])
            elif isinstance(source,str):
                import fsspec
                with fsspec.open(source,'rb') as stream:image=Image.open(io.BytesIO(stream.read()))
            else:raise TypeError('Unsupported image representation; choose an actual image/HR column')
            prompt='' if a.prompt_kind=='empty' else item[a.prompt_column]
            if not isinstance(prompt,str):raise TypeError('Prompt must be a string; explicitly normalize list-valued captions first')
            key=hashlib.sha256(f'{a.source}|{a.name}|{a.split}|{index}'.encode()).hexdigest()[:24]
            path=images/f'{key}.png';ImageOps.exif_transpose(image).convert('RGB').save(path)
            row={'id':key,'image':f'images/{key}.png','prompt':prompt,'prompt_kind':a.prompt_kind,
                 'generator':a.generator,'source':a.source,'source_revision':a.revision,'source_index':index,'source_split':a.split}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
            if count%100==0:print(json.dumps({'exported':count}),flush=True)
    if count==0:raise ValueError('No records were exported')
    (root/'source.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    print(json.dumps({'exported':count,'manifest':str(manifest)}))

if __name__=='__main__':main()
