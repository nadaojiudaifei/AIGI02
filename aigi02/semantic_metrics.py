"""Optional CLIP image/text and DINO image similarity, with explicit local assets."""
import yaml
import torch
from PIL import Image,ImageOps
from transformers import CLIPModel,AutoProcessor,AutoImageProcessor,AutoModel


class SemanticMetrics:
    def __init__(self,config,device):
        with open(config) as f:cfg=yaml.safe_load(f)
        self.device=device;offline=cfg.get('local_files_only',True)
        self.clip=CLIPModel.from_pretrained(cfg['clip'],local_files_only=offline).to(device).eval()
        self.processor=AutoProcessor.from_pretrained(cfg['clip'],local_files_only=offline)
        self.dino=AutoModel.from_pretrained(cfg['dino'],local_files_only=offline).to(device).eval()
        self.image_processor=AutoImageProcessor.from_pretrained(cfg['dino'],local_files_only=offline)
    @torch.no_grad()
    def __call__(self,reference,reconstruction,prompt):
        images=[]
        for path in (reference,reconstruction):
            with Image.open(path) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
        inputs=self.processor(images=images,return_tensors='pt').to(self.device)
        vi=self.clip.get_image_features(**inputs);vi=torch.nn.functional.normalize(vi,dim=-1)
        tokens=self.processor(text=[prompt],padding=True,truncation=True,return_tensors='pt').to(self.device)
        vt=torch.nn.functional.normalize(self.clip.get_text_features(**tokens),dim=-1)
        dino_inputs=self.image_processor(images=images,return_tensors='pt').to(self.device)
        d=self.dino(**dino_inputs).last_hidden_state[:,0]
        d=torch.nn.functional.normalize(d,dim=-1)
        return {'clip_image_cosine':float((vi[0]*vi[1]).sum()),
                'clip_text_cosine':float((vi[1]*vt[0]).sum()),
                'dino_cls_cosine':float((d[0]*d[1]).sum())}
