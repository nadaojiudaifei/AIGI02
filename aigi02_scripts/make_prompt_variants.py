#!/usr/bin/env python
"""Generate explicit correct/empty/shuffled prompt protocols without reusing streams."""
import argparse
import copy
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aigi02.data import read_manifest,write_manifest
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
a=p.parse_args();rows=read_manifest(a.manifest);root=Path(a.output)
prompts=list(dict.fromkeys(row['prompt'] for row in rows))
if len(prompts)<2:raise ValueError('Wrong-prompt protocol requires at least two distinct prompts')
wrong_prompt={value:prompts[(i+1)%len(prompts)] for i,value in enumerate(prompts)}
write_manifest(rows,root/'correct.jsonl')
empty=copy.deepcopy(rows);wrong=copy.deepcopy(rows)
for i,row in enumerate(empty):
    row['original_prompt']=row['prompt'];row['prompt']='';row['prompt_kind']='empty';row.pop('maps',None)
for i,row in enumerate(wrong):
    row['original_prompt']=row['prompt'];row['prompt']=wrong_prompt[row['original_prompt']];row['prompt_protocol']='cyclic_wrong_prompt';row.pop('maps',None)
write_manifest(empty,root/'empty.jsonl');write_manifest(wrong,root/'shuffled.jsonl')
print(root)
