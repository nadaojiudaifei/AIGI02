"""End-to-end training and independent bitstream encoding/decoding."""
from __future__ import annotations
import contextlib
import hashlib
import math
import zlib
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .config import architecture_id
from .math_ops import resize, cell_loss
from .latent import SemanticLatent
from .backbone import TextConditioner, TinyBackbone, SanaBackbone
from .bitstream import Packet, profile_key


def tensor_hash(x: Tensor) -> str:
    return hashlib.sha256(x.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def seeded_noise(shape, seed: int, device, dtype=torch.float32):
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device,dtype=dtype)


class AIGIModel(nn.Module):
    def __init__(self, cfg: dict, with_text=True, initialize_backbone=True):
        super().__init__(); self.cfg = cfg
        self.latent = SemanticLatent(cfg)
        self.backbone = (TinyBackbone(cfg) if cfg['backend']=='tiny'
                         else SanaBackbone(cfg, self.latent.core, initialize=initialize_backbone))
        # TextConditioner deliberately isn't an nn.Module: frozen, always CPU FP32.
        self.text_encoder = TextConditioner(cfg) if with_text else None
        self.checkpoint_id = None
        self.arch_id = architecture_id(cfg)

    @property
    def device(self): return next(self.backbone.parameters()).device

    def text(self, prompts, device=None):
        if self.text_encoder is None: raise RuntimeError('Text encoder was not constructed')
        return self.text_encoder(prompts, device or self.device)

    def pad(self, x):
        h,w = x.shape[-2:]; p = self.cfg['model']['pad_multiple']
        return F.pad(x,(0,(-w)%p,0,(-h)%p),mode='replicate'), (h,w)

    def generation_field(self, out, beta, lam):
        flags = self.cfg['features']
        field = self.latent.field(out['hyper'],out['mu'],out['sigma'], beta, lam,
                                  semantic=flags['semantic_bias'],residual=flags['field_residual'])
        if not flags['field'] or self.cfg['method']!='idea2':
            # The original variance-map rule, expressed as a [0,1] flow time.
            std = (.5*out['logvar']).exp().abs().clamp_max(4)
            field['t'] = .999 * (1 - .6*(std/4)).mean(1,keepdim=True)
        field['t'] = resize(field['t'],out['mean'].shape[-2:]).clamp(1e-4,1-1e-4)
        return field

    def conditions(self, out, text, mask, drop_text=None, drop_latent=None):
        latent_tokens = self.backbone.condition(out['prompt_latent'])
        b = text.shape[0]
        if drop_text is not None:
            null, nullmask, _ = self.text(['']*b,text.device)
            select = drop_text.reshape(b,1,1)
            text = torch.where(select,null,text); mask = torch.where(select[:,:,0],nullmask,mask)
        if not self.cfg['features']['prompt_decoder']:
            text,mask,_ = self.text(['']*b,text.device)
        latent_mask = torch.ones(latent_tokens.shape[:2],device=text.device,dtype=torch.bool)
        if drop_latent is not None:
            latent_tokens = latent_tokens * (~drop_latent).reshape(b,1,1)
            latent_mask = latent_mask & (~drop_latent).reshape(b,1)
        return torch.cat((text,latent_tokens),1), torch.cat((mask,latent_mask),1)

    def reconstruct(self, out, text, mask, beta, lam, seed, return_latent=False):
        field = self.generation_field(out, beta, lam)
        t = field['t']; noise = seeded_noise(out['mean'].shape,seed,out['mean'].device,out['mean'].dtype)
        noisy = (1-t)*out['mean'] + t*noise
        cond,cmask = self.conditions(out,text,mask)
        pred = self.backbone.predict(noisy,cond,cmask,t,beta,lam,
                  spatial=self.cfg['features']['spatial_adaln'] and self.cfg['method']=='idea2')
        zhat = noisy - t*pred + out['res']
        image = self.backbone.decode(zhat)
        return {'xhat':image,'latent_hat':zhat,'noisy':noisy,**field}

    def forward(self, x, prompts, beta=None, seed=None, oracle=None):
        x,original = self.pad(x)
        b = x.shape[0]; device=x.device
        lam = torch.full((b,),math.log(self.cfg['rate']['lambda_rd']),device=device)
        if beta is None:
            beta = (torch.rand(b,device=device)*self.cfg['rate']['beta_max'] if self.training
                    else torch.full((b,),self.cfg['rate']['beta'],device=device))
        elif not isinstance(beta,Tensor): beta = torch.full((b,),float(beta),device=device)
        text,mask,_ = self.text(prompts,device)
        z0,aux = self.backbone.analysis(x)
        out = self.latent(z0,aux,text,mask,lam,oracle)
        if seed is None: seed=int(torch.randint(0,2**31-1,()).item())
        rec = self.reconstruct(out,text,mask,beta,lam,seed)
        out.update(rec, z0=z0, original=x, text=text, text_mask=mask,
                   beta=beta, log_lambda=lam, seed=seed)
        out['xhat'] = out['xhat'][...,:original[0],:original[1]]
        out['rate_map'] = -out['likelihood_y'].log2().sum(1,keepdim=True)
        pixels=x.shape[-2]*x.shape[-1]
        out['bpp_estimate'] = ((-out['likelihood_y'].log2()).sum()+(-out['likelihood_z'].log2()).sum())/(b*pixels)
        return out

    def quantization_cell_loss(self, out):
        from torch.func import functional_call
        xhat = out['xhat']
        xhat,_ = self.pad(xhat)
        z,aux = self.backbone.analysis(xhat)
        # Encoder weights are detached while gradients still flow to xhat.
        transform = self.latent.core.g_a
        state = {k:v.detach() for k,v in transform.named_parameters()}
        state.update({k:v.detach() for k,v in transform.named_buffers()})
        y = functional_call(transform,state,(z,aux))
        return cell_loss(y,out['m'],out['qhat'])

    @contextlib.contextmanager
    def coding_mode(self):
        was_training=self.training; old=next(self.latent.parameters()).device
        threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.eval(); self.latent.cpu().float()
        try: yield
        finally:
            self.latent.to(old); self.train(was_training); torch.set_num_threads(threads)

    def model_id(self):
        if self.checkpoint_id: return self.checkpoint_id
        digest=hashlib.sha256(self.arch_id.encode())
        for key,p in self.named_parameters():
            digest.update(key.encode()); digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def compress(self, x, prompt, seed=2026, prompt_mode=None, oracle=None):
        if x.ndim!=4 or x.shape[0]!=1 or x.shape[1]!=3: raise ValueError('Expected one RGB image [1,3,H,W]')
        mode = prompt_mode or self.cfg['rate']['prompt_mode']
        if mode not in ('included','free'): raise ValueError('Unknown prompt rate setting')
        x,size=self.pad(x.to(self.device)); model_id=self.model_id()
        with self.coding_mode():
            z,aux=self.backbone.analysis(x)
            text,mask,_=self.text([prompt],'cpu')
            lam=torch.tensor([math.log(self.cfg['rate']['lambda_rd'])])
            streams,shape,out=self.latent.compress(z.cpu().float(),aux.cpu().float(),text,mask,lam,oracle)
            header={'backend':self.cfg['backend'],'architecture':self.arch_id,'checkpoint':model_id,
              'original_size':list(size),'padded_size':list(x.shape[-2:]),**shape,
              'seed':int(seed),'lambda_rd':self.cfg['rate']['lambda_rd'],
              'prompt_mode':mode,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
              'text_features_sha256':tensor_hash(text),
              'entropy_profile':'cpu-fp32-'+str(torch.__version__),
              'map_source':'oracle' if oracle is not None else 'distilled'}
            payload=zlib.compress(prompt.encode(),9) if mode=='included' else b''
            packet=Packet(header,[payload,*streams])
            diagnostic={k:out[k].detach().cpu() for k in ('m','w','mu','sigma')}
            diagnostic.update({k:v.detach().cpu() for k,v in out['maps'].items()})
            diagnostic['estimated_bits_map']=-out['likelihood_y'].log2().sum(1,keepdim=True).cpu()
            return packet,diagnostic

    @torch.no_grad()
    def decompress(self, packet:Packet, external_prompt=None, beta=None):
        h=packet.header
        expected_profile={'architecture':self.arch_id,'checkpoint':self.model_id(),
                          'entropy_profile':'cpu-fp32-'+str(torch.__version__)}
        if profile_key(h)!=profile_key(expected_profile):
            raise ValueError('Bitstream checkpoint, architecture or entropy runtime mismatch')
        if h.get('backend')!=self.cfg['backend']: raise ValueError('Bitstream backend mismatch')
        prompt=packet.prompt(external_prompt)
        p=self.cfg['model']['pad_multiple']; height,width=h['original_size']
        pad=[height+(-height)%p,width+(-width)%p]
        if 'padded_size' in h and h['padded_size']!=pad: raise ValueError('Invalid padded dimensions')
        factor=4 if self.cfg['backend']=='tiny' else 64
        ysize=[pad[0]//factor,pad[1]//factor]; zsize=[(v+3)//4 for v in ysize]
        if 'y_size' in h and (h['y_size']!=ysize or h['z_size']!=zsize): raise ValueError('Invalid latent dimensions')
        shape={'y_size':ysize,'z_size':zsize}
        if len(packet.segments)!=6: raise ValueError('Expected prompt, hyperprior and four main groups')
        with self.coding_mode():
            text,mask,_=self.text([prompt],'cpu')
            if tensor_hash(text)[:16]!=h.get('text_features_sha256','')[:16]:
                raise ValueError('Text features are not bit-exact; check model files and CPU/runtime settings')
            out=self.latent.decompress(packet.segments[1:],shape,text,mask)
            lam=torch.tensor([math.log(float(h['lambda_rd']))])
            b=torch.tensor([self.cfg['rate']['beta'] if beta is None else float(beta)])
            if b.item()<0: raise ValueError('beta must be nonnegative')
            field=self.generation_field(out,b,lam)
            out={k:(v.to(self.device) if isinstance(v,Tensor) else v) for k,v in out.items()}
            text=text.to(self.device);mask=mask.to(self.device);lam=lam.to(self.device);b=b.to(self.device)
            # field is derived entirely on the standardized CPU entropy path.
            t=field['t'].to(self.device)
            noise=seeded_noise(out['mean'].shape,h['seed'],self.device)
            noisy=(1-t)*out['mean']+t*noise
            cond,cmask=self.conditions(out,text,mask)
            pred=self.backbone.predict(noisy,cond,cmask,t,b,lam,
                 spatial=self.cfg['features']['spatial_adaln'] and self.cfg['method']=='idea2')
            image=self.backbone.decode(noisy-t*pred+out['res'])
            return image[...,:height,:width],{k:v.detach().cpu() for k,v in field.items()}

    def load_weights(self,state,strict=True):
        if self.cfg['backend']=='sana':
            # nn.Module's recursive loader doesn't invoke a child's load_state_dict override.
            core={k[len('latent.core.'):]:v for k,v in state.items() if k.startswith('latent.core.')}
            self.latent.core.load_state_dict(core,strict=True)
        return self.load_state_dict(state,strict=strict)
