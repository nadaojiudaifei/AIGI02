"""Paired metrics by manifest ID, not lexicographic folder position."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import read_manifest,read_image
from .math_ops import resize


def psnr(x,y,mask=None):
    error=((x-y)/2).square().mean(1,keepdim=True)
    if mask is None:mse=error.mean()
    else:mse=(error*mask).sum()/mask.sum().clamp_min(1)
    return -10*math.log10(max(float(mse),1e-12))


def patch_count(h,w,size=256):
    count=(h//size)*(w//size)
    if h>=1.5*size and w>=1.5*size:count+=((h-size//2)//size)*((w-size//2)//size)
    return count


def evaluate(manifest,recon_dir,output,metrics=('psnr',),device='cpu',distribution=False,rates=None,semantic_config=None):
    rows=read_manifest(manifest);root=Path(recon_dir);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    names=[x.strip() for x in metrics if x.strip()];valid={'psnr','lpips','dists','ms_ssim','clipiqa','musiq','niqe'}
    if not set(names)<=valid:raise ValueError(f'Unsupported metrics: {set(names)-valid}')
    modules={};spatial_lpips=None
    if set(names)-{'psnr'}:
        import pyiqa
        for name in names:
            if name!='psnr':modules[name]=pyiqa.create_metric(name,device=device)
        if 'lpips' in names:
            import lpips
            spatial_lpips=lpips.LPIPS(net='alex',spatial=True).eval().to(device)
    fid=kid=None;npatches=0
    if distribution:
        from PIL import Image,ImageOps
        for row in rows:
            with Image.open(row['image']) as image:
                image=ImageOps.exif_transpose(image);w,h=image.size
            if min(h,w)<256:raise ValueError('FID/256 requires every image dimension >=256')
            npatches+=patch_count(h,w)
        if npatches<2:raise ValueError('Too few patches for a distribution metric')
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from eval._update_patch_fid import update_patch_fid
        fid=FrechetInceptionDistance(feature=2048,normalize=False).to(device)
        if len(rows)>50:kid=KernelInceptionDistance(subsets=100,subset_size=min(1000,npatches),normalize=False).to(device)
    semantic=None
    if semantic_config:
        from .semantic_metrics import SemanticMetrics
        semantic=SemanticMetrics(semantic_config,device)
    records=[];rate_lookup={}
    if rates:
        with open(rates) as f:rate_lookup={r['id']:r for r in map(json.loads,f)}
        if set(rate_lookup)!={r['id'] for r in rows}:raise ValueError('Rate records do not match the evaluation manifest exactly')
    with torch.no_grad():
        for row in rows:
            path=root/f'{row["id"]}.png'
            if not path.exists():raise FileNotFoundError(f'Missing reconstruction for exact sample ID: {path}')
            x=read_image(row['image']).unsqueeze(0).to(device);y=read_image(path).unsqueeze(0).to(device)
            if x.shape!=y.shape:raise ValueError(f'Size mismatch for {row["id"]}')
            record={'id':row['id']}
            if semantic is not None:record.update(semantic(row['image'],path,row['prompt']))
            for name in names:
                if name=='psnr':value=psnr(x,y)
                elif name in ('clipiqa','musiq','niqe'):value=float(modules[name]((y+1)/2).item())
                else:value=float(modules[name]((y+1)/2,(x+1)/2).item())
                if not math.isfinite(value):raise FloatingPointError(f'Non-finite {name} for {row["id"]}')
                record[name]=value
            if row.get('maps'):
                with np.load(row['maps'],allow_pickle=False) as a:
                    metadata=json.loads(str(a['metadata']))
                    if metadata.get('preprocessing')!='full-pad-v1':
                        raise ValueError('Regional evaluation needs full-resolution oracle maps, not training crops')
                    if metadata['original_size']!=list(x.shape[-2:]):raise ValueError('Oracle map/image dimensions differ')
                    from .data import sample_hash
                    if metadata['sample_sha256']!=sample_hash(x[0].cpu(),row['prompt'],0):raise ValueError('Stale regional map cache')
                    s=torch.from_numpy(a['S']).unsqueeze(0).to(device)
                    s=resize(s,tuple(metadata['padded_size']))[...,:x.shape[-2],:x.shape[-1]]
                high=(s>=torch.quantile(s.flatten(),.75)).float();low=1-high
                record['high_area_fraction']=float(high.mean())
                for label,mask in [('high',high),('low',low)]:
                    if mask.sum()==0:
                        record[f'{label}_psnr']=None;continue
                    record[f'{label}_psnr']=psnr(x,y,mask)
                    if spatial_lpips is not None:
                        lp=spatial_lpips(y,x);m=resize(mask,lp.shape[-2:])
                        record[f'{label}_lpips']=float((lp*m).sum()/m.sum().clamp_min(1e-8))
            if fid is not None:update_patch_fid((x+1)/2,(y+1)/2,fid_metric=fid,kid_metric=kid)
            if rates:record.update({k:v for k,v in rate_lookup[row['id']].items() if k!='id'})
            records.append(record)
    keys=sorted({k for r in records for k,v in r.items() if isinstance(v,(int,float))})
    summary={k:float(np.mean([r[k] for r in records if isinstance(r.get(k),(int,float))])) for k in keys}
    summary['images']=len(rows);summary['metric_protocol']='PNG-roundtrip; exact manifest ID pairing'
    import importlib.metadata
    summary['metric_software']={}
    for name in ('torch','pyiqa','lpips','torchmetrics','torch-fidelity','transformers'):
        try:summary['metric_software'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:summary['metric_software'][name]=None
    if rates:
        total_bits=sum(r['bytes_total']*8 for r in records)
        pixels=sum(r['height']*r['width'] for r in records)
        summary['bpp_total_pixel_weighted']=total_bits/pixels
        summary['bpp_total_image_mean']=float(np.mean([r['bpp_total'] for r in records]))
    if fid is not None:
        summary['fid256']=float(fid.compute());summary['fid256_patches']=npatches
        if kid is not None:
            torch.manual_seed(2026);mean,std=kid.compute()
            summary.update(kid256_mean=float(mean),kid256_std=float(std),kid_subset_size=min(1000,npatches))
        else:summary['kid256_status']='not reported: upstream requires more than 50 images'
    with (out/'per_image.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    columns=sorted({k for r in records for k in r})
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
    return summary


def aggregate(pattern,output):
    import glob
    rows=[]
    for path in sorted(glob.glob(pattern,recursive=True)):
        item=json.loads(Path(path).read_text());rows.append({'run':str(Path(path).parent),**item})
    if not rows:raise ValueError('No completed metric summaries matched')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for row in rows for k in row})
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return len(rows)
