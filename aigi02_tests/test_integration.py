import copy
import json
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from aigi02.assets import safe_unzip
from aigi02.config import tiny_config
from aigi02.data import read_manifest,write_manifest,save_image,ManifestDataset,split_manifest
from aigi02.model import AIGIModel
from aigi02.teacher import cache_maps,oracle_maps
from aigi02.training import run_training,load_model,set_scope
from aigi02.experiments import make_configs,build_matrix,run_matrix


def samples(root,count=3):
    rows=[]
    for i in range(count):
        path=root/f'image{i}.png';save_image(torch.rand(3,32,32)*2-1,path)
        rows.append({'id':f'i{i}','image':str(path),'prompt':f'pattern {i}', 'prompt_kind':'caption','generator':f'g{i}'})
    path=root/'data.jsonl';write_manifest(rows,path);return path


def test_data_rejects_bad_ids_and_missing_prompts(tmp_path):
    path=samples(tmp_path);rows=read_manifest(path);rows[0]['id']='../../escape'
    write_manifest(rows,path)
    with pytest.raises(ValueError):read_manifest(path)
    rows[0]['id']='ok';rows[0]['prompt']=None;write_manifest(rows,path)
    with pytest.raises(ValueError):read_manifest(path)


def test_grouped_split_transitive(tmp_path):
    path=samples(tmp_path);rows=read_manifest(path)
    rows[1]['image']=rows[0]['image'];rows[2]['prompt']=rows[1]['prompt']
    write_manifest(rows,path)
    counts=split_manifest(path,tmp_path/'split',holdout_generator='g2')
    assert counts=={'train':0,'validation':0,'test':3}


def test_zip_traversal_rejected(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('../escape','bad')
    with pytest.raises(ValueError):safe_unzip(path,tmp_path/'out')
    assert not (tmp_path/'escape').exists()


def test_cache_maps_invalidation(tmp_path,model):
    manifest=samples(tmp_path,1);out=tmp_path/'with_maps.jsonl'
    cache_maps(model,manifest,tmp_path/'maps',out)
    ds=ManifestDataset(out,32,True,256);item=ds[0]
    assert set(item['maps'])=={'S','c','e'}
    assert item['maps']['c'].min()>=0
    rows=read_manifest(out);rows[0]['prompt']='changed';write_manifest(rows,out)
    with pytest.raises(ValueError):ManifestDataset(out,32,True,256)[0]


def test_all_training_stages_and_resume(tmp_path):
    torch.set_num_threads(1)
    manifest=samples(tmp_path,2);cfg=tiny_config()
    cfg['teacher'].update(nodes=2,noise_samples=1,steps=2)
    cfg['train'].update(manifest=str(manifest),steps=1,output=str(tmp_path/'maps_stage'),stage='maps')
    maps_manifest=tmp_path/'maps.jsonl'
    cache_maps(AIGIModel(cfg).eval(),manifest,tmp_path/'maps',maps_manifest)
    cfg['train']['manifest']=str(maps_manifest)
    last=run_training(cfg)
    flow=None
    for stage in ('codec','field_pretrain','flow_teacher','distill','gan'):
        cfg=copy.deepcopy(cfg);cfg['train'].update(stage=stage,init=last,output=str(tmp_path/stage),resume=None)
        cfg['features']['realism']=stage=='gan'
        if stage=='codec':cfg['train']['val_manifest']=str(manifest)
        if stage=='distill':cfg['train']['teacher_checkpoint']=flow
        last=run_training(cfg)
        model,state=load_model(last,'cpu');assert state['step']==1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if stage=='flow_teacher':flow=last
    cfg['train'].update(resume=last,init=None,steps=2)
    last=run_training(cfg);assert load_model(last,'cpu')[1]['step']==2
    assert (tmp_path/'codec'/'validation.jsonl').is_file()
    with pytest.raises(FileExistsError):
        cfg['train']['resume']=None;run_training(cfg)


@pytest.mark.parametrize('scope',['maps','allocation','entropy','field','decoder'])
def test_component_scope(scope,model):
    original={n:p.requires_grad for n,p in model.named_parameters()}
    names=set_scope(model,scope)
    assert names
    assert all(original[n] for n in names)
    assert all(not p.requires_grad for n,p in model.named_parameters() if not original[n])


def test_ablation_configs_and_matrix(tmp_path):
    import yaml
    cfg=tiny_config();path=tmp_path/'base.yaml';path.write_text(yaml.safe_dump(cfg))
    generated=make_configs(path,tmp_path/'configs');assert len(generated)==23
    spec={'datasets':{'toy':'manifest.jsonl'},'models':{'new':{'kind':'aigi02','checkpoint':'a.pt','betas':[0,1,2]}},'distribution':False}
    path=tmp_path/'spec.yaml';path.write_text(yaml.safe_dump(spec));plan=tmp_path/'plan.json'
    assert build_matrix(path,plan)==6
    run_matrix(plan,execute=False)
    assert not plan.with_suffix('.status.jsonl').exists()


def test_stage_plan_respects_ablation_and_original_files(tmp_path):
    import yaml
    from aigi02.pipeline import staged_plan,prepare_original_configs
    manifest=samples(tmp_path);cfg=tiny_config();cfg['train']['manifest']=str(manifest)
    cfg['features']['distillation']=False;cfg['features']['realism']=False
    path=tmp_path/'cfg.yaml';path.write_text(yaml.safe_dump(cfg))
    plan=staged_plan(path,tmp_path/'pipeline',execute=False)
    assert [j['stage'] for j in plan['jobs']]==['maps','codec','field_pretrain','flow_teacher']
    assert all(not Path(j['checkpoint']).exists() for j in plan['jobs'])
    result=prepare_original_configs(manifest,manifest,'SANA','elic.pth',tmp_path/'original')
    assert len(result['configs'])==4 and not result['upstream_files_modified']
    assert all(Path(p).exists() for p in result['configs'])


def test_alignment_helper_matches_pinned_original():
    import ast
    import torch.nn.functional as F
    from torch import nn
    from aigi02.upstream_compat import LatentConditionAlignment
    path=Path(__file__).resolve().parents[1]/'models/DiT_IC.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LatentConditionAlignment')
    namespace={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    original=namespace['LatentConditionAlignment'](16,32,77,mode='mlp',use_clip_contrast=False)
    compatible=LatentConditionAlignment(16,32,77)
    compatible.load_state_dict(original.state_dict(),strict=True)
    x=torch.randn(2,16,4,6)
    assert torch.equal(original(x),compatible(x))
