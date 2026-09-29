"""Staged training, component freezing, checkpoint resume and distributed execution."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
import random
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from .config import architecture_id,load_config
from .model import AIGIModel
from .math_ops import resize,weighted_mse,waterfill_target
from .data import ManifestDataset
from .teacher import flow_training_loss,guided_target
from .backbone import safe_torch_load,frozen


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def rng_state():
    n=np.random.get_state()
    return {'torch':torch.get_rng_state(),'python':random.getstate(),
            'numpy':{'kind':n[0],'keys':torch.tensor(n[1].astype('int64')),
                     'position':n[2],'has_gauss':n[3],'cached':n[4]},
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state['torch']);random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['kind'],n['keys'].numpy().astype('uint32'),n['position'],n['has_gauss'],n['cached']))
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def load_model(checkpoint,device=None,asset_overrides=None,with_text=True):
    state=safe_torch_load(checkpoint)
    if state.get('format')!='aigi02-checkpoint-v1':raise ValueError('Expected an AIGI02 checkpoint, not a raw upstream model')
    cfg=load_config(state['config'])
    if device:cfg['device']=str(device)
    if asset_overrides:cfg['assets'].update(asset_overrides)
    # Backbone shape construction can use the original initialization assets;
    # local paths may be overridden when moving between machines.
    model=AIGIModel(cfg,with_text=with_text,initialize_backbone=False)
    model.load_weights(state['model']);model.to(cfg['device']).eval()
    model.checkpoint_id=file_sha256(checkpoint)
    return model,state


def set_scope(model,scope):
    """Never unfreeze originally frozen base weights. Scopes only restrict eligibility."""
    eligible={name:p.requires_grad for name,p in model.named_parameters()}
    prefixes={'maps':('latent.predictor.',),
      'allocation':('latent.predictor.','latent.ugaq.'),
      'entropy':('latent.core.h_a.','latent.core.h_s.','latent.core.adapter_',
                 'latent.core.g_c.','latent.core.entropy_bottleneck.',
                 'latent.hyper_attention.','latent.context_attention.'),
      'field':('latent.field.', 'backbone.spatial_layers.','backbone.adaln.'),
      'decoder':('backbone.','latent.field.','latent.core.g_s.','latent.core.scale.',
                 'latent.core.aux.','latent.core.prompt.')}
    if scope!='all':
        for name,p in model.named_parameters():p.requires_grad_(eligible[name] and name.startswith(prefixes[scope]))
    return [n for n,p in model.named_parameters() if p.requires_grad]


class PatchDiscriminator(nn.Module):
    """Frozen DINOv2-with-registers features plus trainable patch/text projection."""
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        if cfg['backend']=='tiny':
            self.features=nn.Sequential(nn.Conv2d(3,32,4,4),nn.SiLU())
            dim=32;self.dino=None
        else:
            from transformers import AutoModel
            self.dino=frozen(AutoModel.from_pretrained(cfg['assets']['discriminator'],
                    local_files_only=cfg['assets']['local_files_only']))
            dim=self.dino.config.hidden_size
        self.head=nn.Sequential(nn.Linear(dim,dim),nn.LeakyReLU(.2),nn.Linear(dim,1))
        self.text=nn.Linear(cfg['model']['text_dim'],dim)
        self.dim=dim
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))

    def train(self,mode=True):
        super().train(mode)
        if self.dino is not None:self.dino.eval()
        return self

    def forward(self,image,text,mask):
        if self.dino is None:
            f=self.features(image);hh,ww=f.shape[-2:];tokens=f.flatten(2).transpose(1,2)
        else:
            inp=(F.interpolate((image+1)/2,size=(224,224),mode='bilinear',align_corners=False)-self.mean)/self.std
            tokens=self.dino(pixel_values=inp).last_hidden_state
            nreg=getattr(self.dino.config,'num_register_tokens',0)
            tokens=tokens[:,1+nreg:];hh=ww=int(math.isqrt(tokens.shape[1]))
            if hh*ww!=tokens.shape[1]:raise ValueError('Unexpected DINO patch token count')
        pooled=(text*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        projection=(tokens*self.text(pooled)[:,None,:]).sum(-1,keepdim=True)/self.dim**.5
        return (self.head(tokens)+projection).transpose(1,2).reshape(image.shape[0],1,hh,ww)


@contextlib.contextmanager
def no_parameter_grad(module):
    state=[p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False);yield
    finally:
        for p,flag in zip(module.parameters(),state):p.requires_grad_(flag)


def map_loss(predicted,targets):
    loss=predicted['S'].sum()*0
    for key in ('S','c','e'):
        a=predicted[key];b=resize(targets[key].to(a),a.shape[-2:])
        if key=='c':a=a.log1p();b=b.clamp_min(0).log1p()
        if key=='e':a=a.sign()*a.abs().log1p();b=b.sign()*b.abs().log1p()
        loss=loss+F.l1_loss(a,b)
    return loss


class Objective(nn.Module):
    def __init__(self,model,cfg,teacher=None,discriminator=None):
        super().__init__();self.model=model;self.cfg=cfg
        # Frozen teacher and separately optimized discriminator must not enter DDP generator state.
        object.__setattr__(self,'teacher_model',teacher)
        object.__setattr__(self,'discriminator',discriminator)
        self.lpips=None
        if cfg['loss']['perceptual']>0 and cfg['train']['stage'] not in ('maps','field_pretrain','flow_teacher'):
            import lpips
            self.lpips=frozen(lpips.LPIPS(net='alex',spatial=True)).to(model.device)
        self.last_fake=None

    def forward(self,x,prompts,targets,step):
        cfg=self.cfg;stage=cfg['train']['stage'];model=self.model
        if stage=='maps':
            with torch.no_grad():
                z,aux=model.backbone.analysis(x);y=model.latent.core.g_a(z,aux)
            text,mask,_=model.text(prompts,x.device)
            lam=torch.full((x.shape[0],),math.log(cfg['rate']['lambda_rd']),device=x.device)
            pred=model.latent.predictor(y,text,mask,lam)
            loss=map_loss(pred,targets)
            return loss, {'loss':loss.detach(),'map':loss.detach()}, loss.detach()*0
        oracle=targets if cfg['train']['map_source']=='oracle' else None
        out=model(x,prompts,oracle=oracle)
        logs={};aux=model.latent.core.aux_loss()
        if stage in ('field_pretrain','flow_teacher'):
            loss=flow_training_loss(model,out,smooth_random=stage=='field_pretrain')
            logs.update(loss=loss.detach(),flow=loss.detach())
            return loss,logs,aux
        flags=cfg['features'];weights=cfg['loss']
        t=resize(out['t'],out['w'].shape[-2:])
        a=out['w']*(1-t) if cfg['method']=='idea2' and flags['field'] else out['w']
        distortion=weighted_mse(x,out['xhat'],a)
        rate=out['bpp_estimate']
        # Lossless text payload is a constant rate term, not a differentiable proxy.
        import zlib
        text_bpp=(sum(8*len(zlib.compress(p.encode(),9)) for p in prompts)/(len(prompts)*x.shape[-2]*x.shape[-1])
                  if cfg['rate']['prompt_mode']=='included' else 0.)
        loss=rate+text_bpp+cfg['rate']['lambda_rd']*distortion
        logs.update(bpp_estimate=rate.detach(),text_bpp=text_bpp,mse=distortion.detach())
        if self.lpips is not None:
            spatial=self.lpips(out['xhat'].float(),x.float())
            w=resize(out['w'].detach(),spatial.shape[-2:])
            perceptual=(spatial*w).sum()/w.sum().clamp_min(1e-8)
            loss=loss+cfg['rate']['lambda_rd']*weights['perceptual']*perceptual
            logs['lpips']=perceptual.detach()
        align=(1-F.cosine_similarity(out['latent_hat'],out['z0'].detach(),dim=1)).mean()
        loss=loss+weights['alignment']*align;logs['align']=align.detach()
        importance_loss=F.l1_loss(out['w_hat'],out['w'].detach())
        loss=loss+weights['importance_head']*importance_loss;logs['importance_head']=importance_loss.detach()
        if targets and cfg['train']['map_source']!='oracle':
            maps=map_loss(out['maps'],targets);loss=loss+weights['map']*maps;logs['map']=maps.detach()
        if flags['cell']:
            cell=model.quantization_cell_loss(out);loss=loss+weights['cell']*cell;logs['cell']=cell.detach()
        if flags['waterfill']:
            target=waterfill_target(out['w'],out['maps']['c'],out['rate_map'])
            wf=F.l1_loss(out['rate_map'],target)
            anneal=max(0.,1-step/max(1,cfg['train']['steps']*.5))
            loss=loss+weights['waterfill']*anneal*wf;logs['waterfill']=wf.detach()
        if stage=='distill':
            if self.teacher_model is None:raise ValueError('Distillation requires a trained flow_teacher checkpoint')
            if not flags['distillation']:raise ValueError('distill stage conflicts with disabled distillation feature')
            target=guided_target(self.teacher_model,out)
            kd=F.mse_loss(out['latent_hat'],target);loss=loss+weights['distill']*kd;logs['distill']=kd.detach()
        if stage=='gan':
            if self.discriminator is None:raise ValueError('GAN stage requires features.realism=true')
            with no_parameter_grad(self.discriminator):
                score=self.discriminator(out['xhat'],out['text'],out['text_mask'])
                wt=resize(out['t'].detach(),score.shape[-2:])
                adv=-(score*wt).mean()
            loss=loss+weights['realism']*adv;logs['adversarial']=adv.detach()
            self.last_fake=(out['xhat'].detach(),out['text'].detach(),out['text_mask'].detach())
        logs['loss']=loss.detach()
        return loss,logs,aux


def _move_targets(batch,device):
    return {k:v.to(device) for k,v in batch.get('maps',{}).items()}


@torch.no_grad()
def validate_model(model,cfg,output,step):
    """Bounded deterministic validation; no optimizer update and RNG is restored."""
    path=cfg['train']['val_manifest']
    if not path:return
    saved=rng_state();was_training=model.training
    try:
        model.eval();ds=ManifestDataset(path,cfg['train']['crop'],False,cfg['rate']['lambda_rd'])
        records=[]
        for i in range(min(len(ds),cfg['train']['val_max_images'])):
            batch=ds[i];x=batch['image'].unsqueeze(0).to(model.device)
            out=model(x,[batch['prompt']],seed=cfg['seed'])
            mse=((x-out['xhat'])/2).square().mean()
            records.append((float(-10*torch.log10(mse.clamp_min(1e-12))),float(out['bpp_estimate'])))
        row={'step':step,'images':len(records),'psnr':sum(x[0] for x in records)/len(records),
             'bpp_estimate':sum(x[1] for x in records)/len(records),'rate_is_actual_bitstream':False}
        with (Path(output)/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    finally:
        model.train(was_training);restore_rng(saved)


def run_training(cfg):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; tiny tests use device=cpu')
    device=torch.device(f'cuda:{local}' if cfg['device'].startswith('cuda') else cfg['device'])
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:torch.distributed.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    seed_everything(cfg['seed']+rank)
    if not cfg['train']['manifest']:raise ValueError('Set train.manifest to a prepared JSONL manifest')
    if (Path(cfg['train']['output'])/'last.pt').exists() and not cfg['train']['resume']:
        raise FileExistsError('Output already contains a checkpoint; set train.resume or use a new output directory')
    source=cfg['train']['resume'] or cfg['train']['init']
    model=AIGIModel(cfg,initialize_backbone=not bool(source)).to(device)
    state=None
    if source:
        state=safe_torch_load(source)
        if state.get('format')!='aigi02-checkpoint-v1':raise ValueError('train.init/resume must be an AIGI02 checkpoint')
        if cfg['train']['resume'] and architecture_id(state['config'])!=architecture_id(cfg):
            raise ValueError('Cannot resume a changed architecture; use train.init for stage transitions')
        model.load_weights(state['model'])
    scope=cfg['train']['trainable']
    if cfg['train']['stage']=='maps':scope='maps'
    if cfg['train']['stage'] in ('field_pretrain','flow_teacher','distill') and scope=='all':scope='decoder'
    trainable=set_scope(model,scope)
    if not trainable:raise ValueError('No trainable parameters in selected scope')
    params=[p for n,p in model.named_parameters() if p.requires_grad and not n.endswith('.quantiles')]
    aux_params=[p for n,p in model.named_parameters() if p.requires_grad and n.endswith('.quantiles')]
    optimizer=torch.optim.AdamW(params,lr=cfg['train']['lr'],weight_decay=1e-4)
    aux_optimizer=torch.optim.Adam(aux_params,lr=cfg['train']['aux_lr']) if aux_params else None
    teacher=None
    if cfg['train']['stage']=='distill':
        path=cfg['train']['teacher_checkpoint']
        if not path:raise ValueError('Set train.teacher_checkpoint to a completed flow_teacher run')
        teacher,teacher_state=load_model(path,str(device),cfg['assets'],with_text=False)
        teacher.text_encoder=model.text_encoder
        if teacher_state['config']['train']['stage']!='flow_teacher':
            raise ValueError('The distillation teacher must have been trained with stage=flow_teacher')
        frozen(teacher)
    discriminator=PatchDiscriminator(cfg).to(device) if cfg['train']['stage']=='gan' and cfg['features']['realism'] else None
    if world>1 and discriminator is not None:
        for tensor in list(discriminator.parameters())+list(discriminator.buffers()):torch.distributed.broadcast(tensor.data,src=0)
    disc_optimizer=(torch.optim.Adam([p for p in discriminator.parameters() if p.requires_grad],lr=cfg['train']['lr'],betas=(.5,.9))
                    if discriminator is not None else None)
    objective=Objective(model,cfg,teacher,discriminator)
    start=0;epoch=0;batch_offset=0
    if cfg['train']['resume']:
        optimizer.load_state_dict(state['optimizer'])
        if aux_optimizer is not None and state['aux_optimizer']:aux_optimizer.load_state_dict(state['aux_optimizer'])
        if disc_optimizer is not None:
            discriminator.load_state_dict(state['discriminator']);disc_optimizer.load_state_dict(state['disc_optimizer'])
        if state.get('data_manifest_sha256')!=file_sha256(cfg['train']['manifest']):raise ValueError('Resume manifest has changed')
        start=state['step'];epoch=state['data_epoch'];batch_offset=state['data_batches']
        if state['world_size']!=world:raise ValueError('Exact resume requires unchanged distributed world size')
        restore_rng(state['rng_by_rank'][rank])
    if world>1:
        objective=torch.nn.parallel.DistributedDataParallel(objective,device_ids=[local] if device.type=='cuda' else None,
                                                          find_unused_parameters=True,broadcast_buffers=False)
    require_maps=(cfg['train']['stage']=='maps' or cfg['train']['map_source']=='oracle' or
                  (cfg['backend']=='sana' and source is None and cfg['train']['stage']=='codec'))
    dataset=ManifestDataset(cfg['train']['manifest'],cfg['train']['crop'],require_maps,cfg['rate']['lambda_rd'])
    if len(dataset)<world*cfg['train']['batch_size']:raise ValueError('Dataset is smaller than one distributed batch')
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,seed=cfg['seed'],shuffle=True) if world>1 else None
    def loader_for(ep):
        if sampler is not None:sampler.set_epoch(ep)
        return DataLoader(dataset,batch_size=cfg['train']['batch_size'],sampler=sampler,
              shuffle=sampler is None,generator=torch.Generator().manual_seed(cfg['seed']+ep),
              num_workers=cfg['train']['num_workers'],pin_memory=device.type=='cuda',drop_last=True)
    loader=loader_for(epoch);iterator=iter(loader)
    for _ in range(batch_offset):
        try:next(iterator)
        except StopIteration:raise ValueError('Checkpoint data offset is invalid')
    output=Path(cfg['train']['output']);output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        import yaml
        (output/'config.resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        (output/'trainable_parameters.json').write_text(json.dumps(trainable,indent=2))
    model.train();started=time.monotonic()
    def next_batch():
        nonlocal epoch,batch_offset,loader,iterator
        try:batch=next(iterator)
        except StopIteration:
            epoch+=1;batch_offset=0;loader=loader_for(epoch);iterator=iter(loader);batch=next(iterator)
        batch_offset+=1;return batch
    try:
        for step in range(start,cfg['train']['steps']):
            optimizer.zero_grad(set_to_none=True)
            if aux_optimizer is not None:aux_optimizer.zero_grad(set_to_none=True)
            combined={};accum=cfg['train']['accumulation']
            for micro in range(accum):
                batch=next_batch();x=batch['image'].to(device);targets=_move_targets(batch,device)
                sync=(objective.no_sync() if world>1 and micro<accum-1 else contextlib.nullcontext())
                with sync:
                    loss,logs,aux=objective(x,batch['prompt'],targets,step)
                    if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at step {step}')
                    if aux_optimizer is not None and aux.requires_grad:
                        total=loss+aux
                    else:total=loss
                    (total/accum).backward()
                for k,v in logs.items():combined[k]=combined.get(k,0)+float(v)/accum
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['train']['clip_grad'],error_if_nonfinite=True)
            optimizer.step()
            if aux_optimizer is not None:aux_optimizer.step()
            raw=objective.module if world>1 else objective
            if discriminator is not None:
                fake,text,mask=raw.last_fake;disc_optimizer.zero_grad(set_to_none=True)
                real_score=discriminator(x.detach(),text,mask);fake_score=discriminator(fake,text,mask)
                dloss=(F.relu(1-real_score).mean()+F.relu(1+fake_score).mean())*.5
                dloss.backward()
                if world>1:
                    for p in discriminator.parameters():
                        if p.grad is not None:torch.distributed.all_reduce(p.grad);p.grad.div_(world)
                disc_optimizer.step();combined['discriminator']=float(dloss.detach())
            lr=cfg['train']['lr']*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/cfg['train']['steps'])))
            for group in optimizer.param_groups:group['lr']=lr
            if (step+1)%cfg['train']['log_every']==0 and rank==0:
                record={'step':step+1,'elapsed_s':time.monotonic()-started,'grad_norm':float(norm),'lr':lr,**combined}
                print(json.dumps(record,allow_nan=False),flush=True)
                with (output/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if (step+1)%cfg['train']['save_every']==0 or step+1==cfg['train']['steps']:
                local_rng=rng_state();all_rng=[None]*world
                if world>1:torch.distributed.all_gather_object(all_rng,local_rng)
                else:all_rng=[local_rng]
                if rank==0:
                    checkpoint={'format':'aigi02-checkpoint-v1','config':cfg,'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),'aux_optimizer':aux_optimizer.state_dict() if aux_optimizer else None,
                        'discriminator':discriminator.state_dict() if discriminator else None,
                        'disc_optimizer':disc_optimizer.state_dict() if disc_optimizer else None,
                        'step':step+1,'data_epoch':epoch,'data_batches':batch_offset,'world_size':world,'rng_by_rank':all_rng,
                        'data_manifest_sha256':file_sha256(cfg['train']['manifest'])}
                    temporary=output/'last.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(output/'last.pt')
                    validate_model(model,cfg,output,step+1)
                if world>1:torch.distributed.barrier()
    finally:
        if world>1:torch.distributed.destroy_process_group()
    return str(output/'last.pt')
