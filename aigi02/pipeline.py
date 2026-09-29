"""Explicit staged plans and additive original-training configuration generation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import yaml
from .config import load_config
from .data import read_manifest


def staged_plan(config,output,execute=False,resume=False,nproc=1):
    """One process per stage releases frozen teacher/text/optimizer memory cleanly."""
    cfg=load_config(config);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    if not cfg['train']['manifest']:raise ValueError('Set the prepared, map-cached training manifest')
    if nproc<1:raise ValueError('nproc must be positive')
    stages=['maps','codec']
    if cfg['method']=='idea2':
        stages+=['field_pretrain','flow_teacher']
        if cfg['features']['distillation']:stages+=['distill']
    if cfg['features']['realism']:stages+=['gan']
    jobs=[];last=cfg['train']['init'];teacher=None
    for stage in stages:
        sc=copy.deepcopy(cfg)
        if cfg['method']=='idea2' and stage in ('maps','codec'):sc['method']='idea1'
        sc['train'].update(stage=stage,init=last,resume=None,output=str(root/stage))
        if stage=='maps':sc['train']['steps']=min(10000,cfg['train']['steps'])
        if stage=='field_pretrain':sc['train']['steps']=min(10000,cfg['train']['steps'])
        if stage=='flow_teacher':sc['train']['steps']=min(20000,cfg['train']['steps'])
        if stage=='gan':sc['train']['steps']=min(20000,cfg['train']['steps'])
        if stage=='distill':sc['train']['teacher_checkpoint']=teacher
        checkpoint=root/stage/'last.pt'
        if resume and checkpoint.exists():
            import torch
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if state['step']>=sc['train']['steps']:
                last=str(checkpoint)
                if stage=='flow_teacher':teacher=last
                jobs.append({'stage':stage,'status':'already_completed','checkpoint':last});continue
            sc['train']['resume']=str(checkpoint);sc['train']['init']=None
        path=root/f'{stage}.yaml';path.write_text(yaml.safe_dump(sc,sort_keys=False))
        prefix=[sys.executable,'-m','aigi02'] if nproc==1 else [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={nproc}','-m','aigi02']
        argv=prefix+['train','--config',str(path)]
        job={'stage':stage,'argv':argv,'checkpoint':str(checkpoint),'status':'planned'}
        if execute:
            log=root/f'{stage}.console.log'
            with log.open('a') as f:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{stage} failed; inspect {log}')
            job['status']='completed'
        jobs.append(job);last=str(checkpoint)
        if stage=='flow_teacher':teacher=last
    plan={'jobs':jobs,'final_checkpoint':last,'executed':execute,'note':'Completed stages are actual training, not a performance claim.'}
    (root/'pipeline.json').write_text(json.dumps(plan,indent=2));return plan


def prepare_original_configs(train_manifest,validation_manifest,sana,elic,output,merged=None,batch=8,steps=100001):
    """Materialize symlink views so original ImageFolder sees exactly the manifest set."""
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    views={}
    for split,manifest in [('train',train_manifest),('validation',validation_manifest)]:
        folder=root/f'{split}_images';folder.mkdir(exist_ok=True)
        for row in read_manifest(manifest):
            source=Path(row['image']).resolve();target=folder/(row['id']+source.suffix)
            if target.exists():
                if target.resolve()!=source:raise FileExistsError(f'Different existing dataset link: {target}')
            else:target.symlink_to(source)
        views[split]=str(folder)
    repository=Path(__file__).resolve().parents[1];written=[]
    for path in sorted((repository/'configs').glob('train_*.yaml')):
        if 'merge' in path.name and not merged:continue
        cfg=yaml.safe_load(path.read_text());cfg['data']['data_path']=[views['train']];cfg['data']['valid_path']=[views['validation']]
        params=cfg['model']['params'];params['dit_path']=sana;params['elic_path']=elic
        if 'merge' in path.name:params['codec_path']=merged
        cfg['train']['output_dir']=str(root/'runs');cfg['train']['resume']=False
        cfg['train']['global_batch_size']=batch;cfg['train']['max_steps']=steps
        dest=root/path.name;dest.write_text(yaml.safe_dump(cfg,sort_keys=False));written.append(str(dest))
    return {'configs':written,'image_views':views,'upstream_files_modified':False,
            'note':'Run non-GAN first; then explicitly set the chosen raw or merged checkpoint in the GAN config.'}
