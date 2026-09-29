"""Explicit downloads and local import. Ordinary training never installs packages."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from .data import write_manifest,read_manifest,EXTENSIONS

ROOT=Path(__file__).resolve().parents[1]


def hub_download(repo,destination,repo_type='model',revision=None,filename=None,patterns=None,offline=False):
    from huggingface_hub import snapshot_download,hf_hub_download,HfApi
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    resolved=revision
    if not offline:
        info=HfApi().repo_info(repo,repo_type=repo_type,revision=revision)
        resolved=info.sha
    if filename:
        path=hf_hub_download(repo,filename,repo_type=repo_type,revision=resolved,
                             local_dir=str(destination),local_files_only=offline)
    else:
        path=snapshot_download(repo,repo_type=repo_type,revision=resolved,
                               local_dir=str(destination),allow_patterns=patterns,local_files_only=offline)
    lock={'repo_id':repo,'repo_type':repo_type,'revision':resolved,'filename':filename,
          'allow_patterns':patterns,'local_path':str(Path(path).resolve())}
    (destination/'aigi02_asset_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    return lock


def checked_url_download(url,destination,sha256):
    if not url.startswith('https://'):raise ValueError('Only explicit HTTPS model URLs are accepted')
    if len(sha256)!=64 or any(x not in '0123456789abcdefABCDEF' for x in sha256):
        raise ValueError('A complete SHA-256 is required for a direct URL download')
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.download');digest=hashlib.sha256()
    try:
        with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as f:
            while block:=response.read(8*1024*1024):digest.update(block);f.write(block)
        if digest.hexdigest().lower()!=sha256.lower():raise ValueError('Downloaded model SHA-256 mismatch')
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()
    return str(path)


def clone_source(url,destination,revision):
    if not url.startswith('https://github.com/'):raise ValueError('Only explicit github.com HTTPS repositories are supported')
    path=Path(destination)
    if path.exists():raise FileExistsError('Refusing to overwrite an existing source directory')
    subprocess.run(['git','clone','--no-checkout',url,str(path)],check=True)
    subprocess.run(['git','-C',str(path),'checkout','--detach',revision],check=True)
    commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    return {'path':str(path),'commit':commit}


def safe_unzip(archive,destination,max_bytes=32*1024**3,max_files=10000):
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>max_files or sum(i.file_size for i in infos)>max_bytes:raise ValueError('Archive expansion exceeds limits')
        for i in infos:
            relative=PurePosixPath(i.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in i.filename:
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(i.external_attr>>16):raise ValueError('Archive symlinks are not accepted')
            target=(root/str(relative)).resolve()
            if not target.is_relative_to(root):raise ValueError('Archive path leaves destination')
        z.extractall(root)
    return root


def diffusiondb(parts,destination,output_manifest,revision=None,offline=False):
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for part in parts:
        if not 1<=part<=2000:raise ValueError('DiffusionDB 2M part number must be between 1 and 2000')
        name=f'part-{part:06d}';folder=root/name
        archives=root/'archives'
        if not folder.exists():
            lock=hub_download('poloclub/diffusiondb',archives,'dataset',revision,
                               filename=f'images/{name}.zip',offline=offline)
            safe_unzip(lock['local_path'],folder)
        metadata=list(folder.rglob(f'{name}.json'))
        if len(metadata)!=1:raise ValueError(f'Expected exactly one metadata JSON for {name}')
        entries=json.loads(metadata[0].read_text(encoding='utf-8'))
        for image_name,item in entries.items():
            image=metadata[0].parent/image_name
            if not image.is_file():raise FileNotFoundError(image)
            if not isinstance(item.get('p'),str):raise ValueError('DiffusionDB record is missing its original prompt')
            rows.append({'id':f'diffusiondb-{part:06d}-{Path(image_name).stem}',
                         'image':str(image),'prompt':item['p'],'prompt_kind':'true_prompt',
                         'generator':'stable-diffusion','generation_seed':item.get('se'),
                         'source':'poloclub/diffusiondb','source_part':part})
    write_manifest(rows,output_manifest)
    return len(rows)


def export_hf(repo,output,split='train',name=None,limit=1000,image_column='image',prompt_column=None,
              prompt_kind='empty',revision=None,local_disk=False,generator='unknown'):
    # The original repo contains a package named datasets. Use -I so the installed
    # Hugging Face package cannot accidentally resolve to that local directory.
    if not local_disk and revision is None:
        from huggingface_hub import HfApi
        revision=HfApi().repo_info(repo,repo_type='dataset').sha
    command=[sys.executable,'-I',str(ROOT/'aigi02_scripts/hf_export.py'),
      '--source',str(repo),'--output',str(output),'--split',split,'--limit',str(limit),
      '--image-column',image_column,'--prompt-kind',prompt_kind,'--generator',generator]
    if name:command+=['--name',name]
    if prompt_column:command+=['--prompt-column',prompt_column]
    if revision:command+=['--revision',revision]
    if local_disk:command+=['--local-disk']
    subprocess.run(command,check=True)
    return str(Path(output).resolve()/'manifest.jsonl')


def caption_manifest(manifest,output,model_path,device='cuda',offline=True):
    from transformers import BlipProcessor,BlipForConditionalGeneration
    import torch
    from PIL import Image,ImageOps
    processor=BlipProcessor.from_pretrained(model_path,local_files_only=offline)
    model=BlipForConditionalGeneration.from_pretrained(model_path,local_files_only=offline).to(device).eval()
    rows=read_manifest(manifest)
    for row in rows:
        with Image.open(row['image']) as image:
            image=ImageOps.exif_transpose(image).convert('RGB')
            inputs=processor(images=image,return_tensors='pt').to(device)
        with torch.no_grad():tokens=model.generate(**inputs,max_new_tokens=64,num_beams=3,do_sample=False)
        row['original_prompt']=row['prompt'];row['prompt']=processor.decode(tokens[0],skip_special_tokens=True)
        row['prompt_kind']='caption';row['caption_model']=model_path
        row.pop('maps',None) # Changing the prompt invalidates previous map caches.
    write_manifest(rows,output);return len(rows)
