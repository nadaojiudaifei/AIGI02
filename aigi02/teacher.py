"""Finite-SNR oracle map estimation and a spatial two-condition flow teacher."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from .math_ops import information_scores,normalize_map,resize
from .model import seeded_noise
from .data import read_manifest,write_manifest,training_image,sample_hash,read_image


@torch.no_grad()
def oracle_maps(model,x,prompts,seed=2026):
    model.eval();cfg=model.cfg;dev=x.device
    z0,_=model.backbone.analysis(x)
    text,mask,content=model.text(prompts,dev)
    null,nullmask,_=model.text(['']*len(prompts),dev)
    teacher=cfg['teacher']
    # Larger lambda_rd weights fidelity more: more high-SNR analysis nodes.
    upper=min(8.,max(-2.,teacher['log_snr_max']+math.log(cfg['rate']['lambda_rd']/256.)))
    lower=teacher['log_snr_min']
    if lower>=upper:raise ValueError('Configured finite SNR integration interval is empty')
    nodes=torch.linspace(lower,upper,teacher['nodes'],device=dev)
    conditional=[];unconditional=[];attentions=[]
    lam=torch.full((len(prompts),),math.log(cfg['rate']['lambda_rd']),device=dev)
    beta=torch.zeros(len(prompts),device=dev)
    for k,log_snr in enumerate(nodes):
        time=1/(1+torch.exp(.5*log_snr))
        t=torch.ones_like(z0[:,:1])*time
        ec=[];eu=[]
        for draw in range(teacher['noise_samples']):
            noise=seeded_noise(z0.shape,seed+1009*k+draw,dev)
            noisy=(1-t)*z0+t*noise
            vc=model.backbone.predict(noisy,text,mask,t,beta,lam,spatial=False,capture=True)
            a=model.backbone.last_attention
            if a is None:raise RuntimeError('The chosen teacher did not return actual cross-attention')
            attentions.append(a)
            vu=model.backbone.predict(noisy,null,nullmask,t,beta,lam,spatial=False,capture=False)
            # RF v=epsilon-z0. Convert to epsilon_hat; do NOT treat v as epsilon.
            eps_c=noisy+(1-t)*vc;eps_u=noisy+(1-t)*vu
            ec.append((noise-eps_c).square().sum(1,keepdim=True))
            eu.append((noise-eps_u).square().sum(1,keepdim=True))
        conditional.append(torch.stack(ec).mean(0));unconditional.append(torch.stack(eu).mean(0))
    c,e=information_scores(torch.stack(conditional),torch.stack(unconditional),nodes)
    attention=torch.stack(attentions).mean(0)
    # Fixed oracle convention: equal weights over valid non-special text tokens.
    # The distilled predictor learns a mapping instead of requiring an NLP tagger.
    per_token=normalize_map(attention)
    saliency=normalize_map((per_token*content[:,:,None,None]).sum(1,keepdim=True))
    return {'S':saliency,'c':c,'e':e}, {'log_snr_nodes':nodes.cpu().tolist(),
            'noise_samples':teacher['noise_samples'],'oracle_token_weights':'uniform-non-special',
            'score_interpretation':'finite_conditional_denoising_difficulty_not_exact_MI'}


def cache_maps(model,manifest,output,output_manifest,full_resolution=False):
    rows=read_manifest(manifest);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    crop=model.cfg['train']['crop'];model_id=model.model_id()
    for index,row in enumerate(rows):
        image=read_image(row['image']) if full_resolution else training_image(row['image'],crop)
        model_input,_=model.pad(image.unsqueeze(0).to(model.device))
        maps,meta=oracle_maps(model,model_input,[row['prompt']],seed=model.cfg['seed'])
        meta.update(sample_sha256=sample_hash(image,row['prompt'],0 if full_resolution else crop),crop=None if full_resolution else crop,
                    preprocessing='full-pad-v1' if full_resolution else 'center-resize-v1',
                    original_size=list(image.shape[-2:]),padded_size=list(model_input.shape[-2:]),
                    teacher_checkpoint=model_id,lambda_rd=model.cfg['rate']['lambda_rd'],
                    backend=model.cfg['backend'],prompt_kind=row['prompt_kind'])
        path=root/f'{row["id"]}.npz'
        np.savez_compressed(path,**{k:v[0].cpu().numpy() for k,v in maps.items()},metadata=json.dumps(meta))
        row['maps']=str(path.resolve())
        print(json.dumps({'cached':index+1,'total':len(rows),'id':row['id']}),flush=True)
    write_manifest(rows,output_manifest)


def flow_training_loss(model,out,smooth_random=False):
    """Independent p/y dropout affects the generation model, never the entropy model."""
    z0=out['z0'].detach();b=z0.shape[0];dev=z0.device
    if smooth_random:
        field=torch.rand(b,1,4,4,device=dev)*.95+.025
        t=resize(field,z0.shape[-2:])
    else:t=out['t'].detach()
    noise=torch.randn_like(z0);noisy=(1-t)*z0+t*noise
    drop_p=torch.rand(b,device=dev)<model.cfg['teacher']['drop_text']
    drop_y=torch.rand(b,device=dev)<model.cfg['teacher']['drop_latent']
    condition,mask=model.conditions(out,out['text'],out['text_mask'],drop_p,drop_y)
    prediction=model.backbone.predict(noisy,condition,mask,t,out['beta'],out['log_lambda'],spatial=True)
    return (prediction-(noise-z0)).square().mean()


@torch.no_grad()
def guided_target(teacher,out):
    """Euler integration with per-token step dt=-t0/K; fixed decoded conditions."""
    clean={k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in out.items()}
    b=clean['z0'].shape[0];dev=clean['z0'].device
    true=torch.ones(b,device=dev,dtype=torch.bool);false=~true
    c00,m00=teacher.conditions(clean,clean['text'],clean['text_mask'],true,true)
    cp0,mp0=teacher.conditions(clean,clean['text'],clean['text_mask'],false,true)
    cpy,mpy=teacher.conditions(clean,clean['text'],clean['text_mask'],false,false)
    state=clean['noisy'].clone();t0=clean['t'];steps=teacher.cfg['teacher']['steps']
    w=resize(clean['w_hat'],t0.shape[-2:]);snr=resize(clean['snr'],t0.shape[-2:])
    zy=1+.5*w*snr/(1+snr)
    for k in range(steps):
        t=(t0*(1-k/steps)).clamp_min(1e-4)
        args=(t,clean['beta'],clean['log_lambda'])
        v00=teacher.backbone.predict(state,c00,m00,*args,spatial=True)
        vp0=teacher.backbone.predict(state,cp0,mp0,*args,spatial=True)
        vpy=teacher.backbone.predict(state,cpy,mpy,*args,spatial=True)
        zp=1+.5*t*(1-w)
        v=v00+zp*(vp0-v00)+zy*(vpy-vp0)
        state=state-t0/steps*v
    return state+clean['res']
