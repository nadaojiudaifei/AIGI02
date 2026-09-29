"""Executable experiment plans with explicit state, never synthetic benchmark numbers."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import yaml
from .config import load_config,deep_merge

ABLATIONS={
 'full':{},
 'texture_only':{'method':'texture_only','features':{'importance':False,'conditional_score':False,'correction':False,'field':False,'spatial_adaln':False}},
 'no_token_weights':{'features':{'token_weights':False}},
 'no_importance':{'features':{'importance':False}},
 'no_conditional_score':{'features':{'conditional_score':False}},
 'no_texture':{'features':{'texture':False}},
 'no_correction':{'features':{'correction':False}},
 'no_prompt_entropy':{'features':{'prompt_entropy':False}},
 'no_prompt_decoder':{'features':{'prompt_decoder':False}},
 'no_waterfill':{'features':{'waterfill':False}},
 'no_cell':{'features':{'cell':False}},
 'fixed_variance_schedule':{'features':{'field':False,'spatial_adaln':False}},
 'snr_only':{'features':{'semantic_bias':False,'field_residual':False}},
 'no_field_residual':{'features':{'field_residual':False}},
 'no_spatial_adaln':{'features':{'spatial_adaln':False}},
 'no_distillation':{'features':{'distillation':False}},
 'no_gan':{'features':{'realism':False}},
 'oracle_maps':{'train':{'map_source':'oracle'}},
}


def make_configs(base,output):
    base=load_config(base);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    result={}
    for name,change in ABLATIONS.items():
        cfg=deep_merge(base,change)
        cfg['train']['output']=f'aigi02_runs/ablations/{name}'
        cfg['train']['resume']=None
        path=root/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        result[name]=str(path)
    for scope in ('maps','allocation','entropy','field','decoder'):
        cfg=copy.deepcopy(base);cfg['train']['trainable']=scope;cfg['train']['resume']=None
        cfg['train']['output']=f'aigi02_runs/components/{scope}'
        path=root/f'component_{scope}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        result[f'component_{scope}']=str(path)
    (root/'index.json').write_text(json.dumps(result,indent=2));return result


def build_matrix(spec_path,output):
    """Expand explicit datasets/checkpoint mappings into sequential argv jobs.

    No assumed checkpoint names, silent missing runs or asynchronous launches.
    Baseline checkpoints may specify different original LoRA-rank configs.
    """
    with open(spec_path) as f:spec=yaml.safe_load(f)
    jobs=[];python=sys.executable
    root=Path(spec.get('output','aigi02_results'))
    for dataset,manifest in spec['datasets'].items():
        for name,settings in spec['models'].items():
            for beta in settings.get('betas',[None]):
                tag=name+(f'_beta{beta}' if beta is not None else '')
                out=root/dataset/tag
                if settings['kind']=='baseline':
                    if beta is not None:raise ValueError('Baseline does not accept beta sweeps')
                    argv=[python,'-m','aigi02','baseline','--config',settings['config'],'--checkpoint',settings['checkpoint'],
                          '--manifest',manifest,'--output',str(out),'--device',spec.get('device','cuda')]
                    if settings.get('sana'):argv+=['--sana',settings['sana']]
                elif settings['kind']=='aigi02':
                    argv=[python,'-m','aigi02','roundtrip','--checkpoint',settings['checkpoint'],
                          '--manifest',manifest,'--output',str(out),'--device',spec.get('device','cuda')]
                    if beta is not None:argv+=['--beta',str(beta)]
                    if settings.get('asset_config'):argv+=['--asset-config',settings['asset_config']]
                    if settings.get('prompt_mode'):argv+=['--prompt-mode',settings['prompt_mode']]
                    if settings.get('oracle'):argv+=['--oracle-maps']
                else:raise ValueError('Model kind must be baseline or aigi02')
                jobs.append({'name':f'{dataset}/{tag}/roundtrip','argv':argv,'expected':str(out/'rates.jsonl')})
                argv=[python,'-m','aigi02','evaluate','--manifest',manifest,'--recon',str(out/'rec'),
                      '--rates',str(out/'rates.jsonl'),'--output',str(out/'metrics'),
                      '--device',spec.get('device','cuda'),'--metrics',','.join(spec.get('metrics',['psnr','lpips','dists','ms_ssim','clipiqa','musiq','niqe']))]
                if spec.get('distribution',True):argv+=['--distribution']
                if spec.get('semantic_config'):argv+=['--semantic-config',spec['semantic_config']]
                jobs.append({'name':f'{dataset}/{tag}/metrics','argv':argv,'expected':str(out/'metrics/summary.json')})
    plan={'format':'aigi02-experiment-plan-v1','spec_sha256':hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
          'jobs':jobs,'note':'Planning does not imply execution or paper reproduction.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(plan,indent=2))
    return len(jobs)


def run_matrix(path,execute=False,resume=False):
    plan=json.loads(Path(path).read_text());status_path=Path(path).with_suffix('.status.jsonl')
    completed=set()
    if resume and status_path.exists():
        for line in status_path.read_text().splitlines():
            record=json.loads(line)
            if record['returncode']==0:completed.add(record['argv_sha256'])
    for job in plan['jobs']:
        digest=hashlib.sha256(json.dumps(job['argv']).encode()).hexdigest()
        if resume and digest in completed and Path(job['expected']).exists():continue
        print(shlex.join(job['argv']),flush=True)
        if not execute:continue
        started=time.monotonic();log=Path(path).parent/'job_logs'/f'{digest[:16]}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:result=subprocess.run(job['argv'],stdout=f,stderr=subprocess.STDOUT)
        record={'name':job['name'],'argv_sha256':digest,'returncode':result.returncode,
                'elapsed_s':time.monotonic()-started,'log':str(log)}
        with status_path.open('a') as f:f.write(json.dumps(record)+'\n')
        if result.returncode!=0:raise RuntimeError(f'Experiment failed; inspect {log}. Subsequent jobs were not run.')


def original_training_command(config,nproc=1):
    name=Path(config).name
    if not name.startswith('train_'):raise ValueError('Select an original training config')
    script='train_nogan_ddp.py' if 'nogan' in name else 'train_ddp.py'
    return [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={nproc}',script,'--config',config]
