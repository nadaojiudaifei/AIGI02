"""Command line entry points. Run `python -m aigi02 --help`."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import yaml


def config_with_overrides(path,items):
    from .config import load_config,deep_merge
    cfg=load_config(path);changes={}
    for item in items or []:
        if '=' not in item:raise ValueError('--set requires dotted.key=value')
        key,value=item.split('=',1);target=changes;parts=key.split('.')
        for part in parts[:-1]:target=target.setdefault(part,{})
        target[parts[-1]]=yaml.safe_load(value)
    return load_config(deep_merge(cfg,changes))


def get_prompt(args,optional=False):
    if getattr(args,'prompt_file',None):return Path(args.prompt_file).read_text(encoding='utf-8')
    if getattr(args,'prompt',None) is not None:return args.prompt
    if optional:return None
    raise ValueError('Supply --prompt or --prompt-file; --prompt "" explicitly selects an empty prompt')


def asset_overrides(path):
    if not path:return None
    raw=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return raw.get('assets',raw)


def doctor(cfg):
    import torch
    packages=['torch','torchvision','numpy','Pillow','PyYAML','compressai','diffusers','transformers','peft','lpips','pyiqa','datasets']
    versions={}
    for name in packages:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    errors=[]
    if cfg['backend']=='sana':
        for name in ('compressai','diffusers','transformers','peft'):
            if versions[name] is None:errors.append(f'Missing package: {name}')
        if not torch.cuda.is_available():errors.append('CUDA is unavailable')
        if versions['compressai'] and not versions['compressai'].startswith('1.2.8'):errors.append('Use pinned CompressAI 1.2.8')
        if versions['diffusers'] and not versions['diffusers'].startswith('0.35.2'):errors.append('Use pinned Diffusers 0.35.2')
        if cfg['assets']['local_files_only']:
            root=Path(cfg['assets']['sana'])
            for name in ('vae/config.json','transformer/config.json'):
                if not (root/name).is_file():errors.append(f'Missing local SANA config: {root/name}')
            text=Path(cfg['assets']['text'] or cfg['assets']['sana'])
            for name in ('tokenizer','text_encoder'):
                if not (text/name).is_dir():errors.append(f'Missing local text component: {text/name}')
    return {'backend':cfg['backend'],'python':sys.version,'packages':versions,'cuda_available':torch.cuda.is_available(),
            'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'errors':errors,'status':'ready_for_attempt' if not errors else 'dependencies_or_assets_missing',
            'note':'Dependency inspection is not a pretrained GPU forward/backward validation.'}


def parser():
    p=argparse.ArgumentParser(prog='python -m aigi02',description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    def add(name,help):return sub.add_parser(name,help=help)
    def cfg(a):a.add_argument('--config',required=True);a.add_argument('--set',action='append',default=[])
    def checkpoint(a):
        a.add_argument('--checkpoint',required=True);a.add_argument('--device',default=None);a.add_argument('--asset-config')
    def prompt(a):
        g=a.add_mutually_exclusive_group();g.add_argument('--prompt');g.add_argument('--prompt-file')
    a=add('doctor','Inspect dependencies, CUDA and local model paths');cfg(a)
    a=add('train','Train/resume a stage on CPU-test or GPU-production backend');cfg(a)
    a=add('cache-maps','Compute finite-SNR maps from actual teacher denoising/attention');cfg(a)
    a.add_argument('--checkpoint');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a.add_argument('--output-manifest',required=True);a.add_argument('--full-resolution',action='store_true')
    a=add('encode','Encode one image to a standalone AIG2 file');checkpoint(a);prompt(a)
    a.add_argument('--image',required=True);a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--prompt-mode',choices=['included','free']);a.add_argument('--maps-output')
    a=add('decode','Decode using only checkpoint, stream and optional free prompt');checkpoint(a);prompt(a)
    a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--beta',type=float)
    a=add('roundtrip','Encode/decode a manifest with actual byte and time accounting');checkpoint(a)
    a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--beta',type=float)
    a.add_argument('--seed',type=int,default=2026);a.add_argument('--prompt-mode',choices=['included','free']);a.add_argument('--oracle-maps',action='store_true')
    a=add('baseline','Original DiT-IC under the same manifest and byte measurement harness')
    a.add_argument('--config',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--sana')
    a.add_argument('--device',default='cuda');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=2026)
    a=add('baseline-decode','Independently decode a framed original DiT-IC bitstream')
    a.add_argument('--config',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--sana');a.add_argument('--device',default='cuda')
    a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    a=add('evaluate','Common full-reference/no-reference/FID256 evaluation')
    a.add_argument('--manifest',required=True);a.add_argument('--recon',required=True);a.add_argument('--output',required=True)
    a.add_argument('--rates');a.add_argument('--device',default='cuda');a.add_argument('--metrics',default='psnr,lpips,dists,ms_ssim,clipiqa,musiq,niqe');a.add_argument('--distribution',action='store_true');a.add_argument('--semantic-config')
    a=add('aggregate','Aggregate measured summaries without inventing missing values');a.add_argument('--glob',required=True);a.add_argument('--output',required=True)
    a=add('ablation-configs','Write explicit component ablation and training-scope YAML files');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a=add('matrix','Expand explicit checkpoint/dataset mappings into an experiment plan');a.add_argument('--spec',required=True);a.add_argument('--output',required=True)
    a=add('run-matrix','Print or synchronously execute a checked experiment plan');a.add_argument('--plan',required=True);a.add_argument('--execute',action='store_true');a.add_argument('--resume',action='store_true')
    a=add('upstream-train','Run an original training script using a separate generated config');a.add_argument('--config',required=True);a.add_argument('--nproc',type=int,default=1);a.add_argument('--execute',action='store_true')
    a=add('pipeline','Plan or execute all training stages in fresh processes');a.add_argument('--config',required=True);a.add_argument('--output',required=True);a.add_argument('--execute',action='store_true');a.add_argument('--resume',action='store_true');a.add_argument('--nproc',type=int,default=1)
    a=add('prepare-upstream','Create separate original-training configs and manifest-matched symlink image views')
    a.add_argument('--train-manifest',required=True);a.add_argument('--validation-manifest',required=True);a.add_argument('--sana',required=True);a.add_argument('--elic',required=True);a.add_argument('--merged');a.add_argument('--output',required=True);a.add_argument('--batch',type=int,default=8);a.add_argument('--steps',type=int,default=100001)
    a=add('hub-download','Download model/data files with resolved revision provenance')
    a.add_argument('--repo',required=True);a.add_argument('--output',required=True);a.add_argument('--repo-type',choices=['model','dataset'],default='model')
    a.add_argument('--revision');a.add_argument('--filename');a.add_argument('--pattern',action='append');a.add_argument('--offline',action='store_true')
    a=add('url-download','Download a direct HTTPS model asset with a mandatory SHA-256')
    a.add_argument('--url',required=True);a.add_argument('--sha256',required=True);a.add_argument('--output',required=True)
    a=add('git-download','Clone a model/code repository at an explicit ref without running code')
    a.add_argument('--url',required=True);a.add_argument('--revision',required=True);a.add_argument('--output',required=True)
    a=add('prepare-hf','Export a streamed HF dataset or load_from_disk snapshot to a unified manifest')
    a.add_argument('--repo',required=True);a.add_argument('--output',required=True);a.add_argument('--split',default='train');a.add_argument('--name');a.add_argument('--revision')
    a.add_argument('--limit',type=int,default=1000);a.add_argument('--image-column',default='image');a.add_argument('--prompt-column')
    a.add_argument('--prompt-kind',choices=['true_prompt','caption','empty'],default='empty');a.add_argument('--generator',default='unknown');a.add_argument('--local-disk',action='store_true')
    a=add('prepare-diffusiondb','Download/import explicit original-prompt DiffusionDB ZIP shards')
    a.add_argument('--parts',required=True,help='Comma-separated integers, e.g. 1,2,3')
    a.add_argument('--output',required=True);a.add_argument('--manifest',required=True);a.add_argument('--revision');a.add_argument('--offline',action='store_true')
    a=add('prepare-folder','Import local images and CSV/JSONL prompt metadata')
    a.add_argument('--root',required=True);a.add_argument('--output',required=True);a.add_argument('--metadata')
    a.add_argument('--prompt-kind',choices=['true_prompt','caption','empty'],default='empty');a.add_argument('--prompt-column',default='prompt');a.add_argument('--image-column',default='image')
    a=add('caption','Make explicitly labelled BLIP captions for natural or AIGI images')
    a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--model',required=True);a.add_argument('--device',default='cuda');a.add_argument('--online',action='store_true')
    a=add('split','Split with exact image and normalized-prompt grouping')
    a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--validation',type=float,default=.05)
    a.add_argument('--test',type=float,default=.05);a.add_argument('--seed',type=int,default=2026);a.add_argument('--holdout-generator')
    a=add('merge-manifests','Merge natural/AIGI manifests with unique sample IDs')
    a.add_argument('--manifest',action='append',required=True);a.add_argument('--output',required=True)
    a=add('verify-upstream','Verify original source bytes against pinned SHA-256 manifest')
    a.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a=add('smoke','Run local tiny-backend training, independent decoding and evaluations')
    a.add_argument('--output',required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv);result=None
    if a.command=='doctor':result=doctor(config_with_overrides(a.config,a.set))
    elif a.command=='train':
        from .training import run_training
        result={'checkpoint':run_training(config_with_overrides(a.config,a.set))}
    elif a.command=='cache-maps':
        from .training import load_model,seed_everything
        from .model import AIGIModel
        from .teacher import cache_maps
        cfg=config_with_overrides(a.config,a.set);seed_everything(cfg['seed'])
        model=load_model(a.checkpoint,cfg['device'],cfg['assets'])[0] if a.checkpoint else AIGIModel(cfg).to(cfg['device'])
        model.cfg['teacher']=cfg['teacher'];model.cfg['train']['crop']=cfg['train']['crop'];model.cfg['rate']=cfg['rate']
        cache_maps(model,a.manifest,a.output,a.output_manifest,a.full_resolution)
        result={'manifest':a.output_manifest}
    elif a.command in ('encode','decode','roundtrip'):
        from .training import load_model
        from .data import read_image,save_image
        from .bitstream import Packet
        model,_=load_model(a.checkpoint,a.device,asset_overrides(a.asset_config))
        if a.command=='encode':
            packet,diagnostics=model.compress(read_image(a.image).unsqueeze(0).to(model.device),get_prompt(a),a.seed,a.prompt_mode)
            packet.write(a.output);result=packet.rates()
            if a.maps_output:
                import numpy as np
                np.savez_compressed(a.maps_output,**{k:v.numpy() for k,v in diagnostics.items()})
        elif a.command=='decode':
            packet=Packet.read(a.input);image,_=model.decompress(packet,get_prompt(a,True),a.beta)
            save_image(image,a.output);result={'output':a.output,**packet.rates()}
        else:
            from .benchmark import run_roundtrip
            result={'rates':run_roundtrip(model,a.manifest,a.output,a.seed,a.beta,a.prompt_mode,a.oracle_maps)}
    elif a.command in ('baseline','baseline-decode'):
        from .benchmark import OriginalBaseline,BaselinePacket,run_roundtrip
        model=OriginalBaseline(a.config,a.checkpoint,a.device,a.sana)
        if a.command=='baseline':result={'rates':run_roundtrip(model,a.manifest,a.output,a.seed)}
        else:
            from .data import save_image
            packet=BaselinePacket.unpack(Path(a.input).read_bytes());image,_=model.decompress(packet)
            save_image(image,a.output);result={'output':a.output,**packet.rates()}
    elif a.command=='evaluate':
        from .metrics import evaluate
        result=evaluate(a.manifest,a.recon,a.output,a.metrics.split(','),a.device,a.distribution,a.rates,a.semantic_config)
    elif a.command=='aggregate':
        from .metrics import aggregate
        result={'runs':aggregate(a.glob,a.output),'output':a.output}
    elif a.command=='ablation-configs':
        from .experiments import make_configs
        result=make_configs(a.config,a.output)
    elif a.command=='matrix':
        from .experiments import build_matrix
        result={'jobs':build_matrix(a.spec,a.output),'plan':a.output}
    elif a.command=='run-matrix':
        from .experiments import run_matrix
        run_matrix(a.plan,a.execute,a.resume)
    elif a.command=='upstream-train':
        from .experiments import original_training_command
        import shlex
        command=original_training_command(a.config,a.nproc);print(shlex.join(command))
        if a.execute:subprocess.run(command,check=True)
    elif a.command=='pipeline':
        from .pipeline import staged_plan
        result=staged_plan(a.config,a.output,a.execute,a.resume,a.nproc)
    elif a.command=='prepare-upstream':
        from .pipeline import prepare_original_configs
        result=prepare_original_configs(a.train_manifest,a.validation_manifest,a.sana,a.elic,a.output,a.merged,a.batch,a.steps)
    elif a.command=='hub-download':
        from .assets import hub_download
        result=hub_download(a.repo,a.output,a.repo_type,a.revision,a.filename,a.pattern,a.offline)
    elif a.command=='url-download':
        from .assets import checked_url_download
        result={'path':checked_url_download(a.url,a.output,a.sha256)}
    elif a.command=='git-download':
        from .assets import clone_source
        result=clone_source(a.url,a.output,a.revision)
    elif a.command=='prepare-hf':
        from .assets import export_hf
        result={'manifest':export_hf(a.repo,a.output,a.split,a.name,a.limit,a.image_column,a.prompt_column,a.prompt_kind,a.revision,a.local_disk,a.generator)}
    elif a.command=='prepare-diffusiondb':
        from .assets import diffusiondb
        result={'images':diffusiondb([int(x) for x in a.parts.split(',')],a.output,a.manifest,a.revision,a.offline)}
    elif a.command=='prepare-folder':
        from .data import import_folder
        result={'images':import_folder(a.root,a.output,a.metadata,a.prompt_kind,a.prompt_column,a.image_column)}
    elif a.command=='caption':
        from .assets import caption_manifest
        result={'images':caption_manifest(a.manifest,a.output,a.model,a.device,not a.online)}
    elif a.command=='split':
        from .data import split_manifest
        result=split_manifest(a.manifest,a.output,a.validation,a.test,a.seed,a.holdout_generator)
    elif a.command=='merge-manifests':
        from .data import read_manifest,write_manifest
        rows=[]
        for source in a.manifest:
            prefix=hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[:8]
            for row in read_manifest(source):row['id']=prefix+'-'+row['id'];rows.append(row)
        write_manifest(rows,a.output);result={'images':len(rows)}
    elif a.command=='verify-upstream':
        root=Path(a.root);manifest=json.loads((root/'aigi02_docs/upstream_sha256.json').read_text());bad=[]
        for name,expected in manifest.items():
            path=root/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:bad.append(name)
        if bad:raise RuntimeError(f'Original upstream files changed or missing: {bad}')
        result={'verified_files':len(manifest),'modified':0}
    elif a.command=='smoke':
        from .smoke import run_smoke
        result=run_smoke(a.output)
    if result is not None:print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return 0
