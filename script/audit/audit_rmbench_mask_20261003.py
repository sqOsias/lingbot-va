"""CPU evaluation of current actual mask predicate (no compiled GPU attention)."""
import ast
import hashlib
import json
from functools import partial
from pathlib import Path
import torch
from torch.nn.attention.flex_attention import and_masks, or_masks

root=Path(__file__).resolve().parents[1]
path=root/'wan_va/modules/model.py'
tree=ast.parse(path.read_text());node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='FlexAttnFunc' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_get_mask_mod');node.decorator_list=[]
scope=dict(torch=torch,and_masks=and_masks,or_masks=or_masks,partial=partial)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
# token tuple (sample, interleaved video/action block index, noise flag; 0 noisy,1 clean)
tokens=[(b,f,n) for b in range(2) for f in range(6) for n in (0,1)]+[(-1,-1,-1)]
s,f,n=map(torch.tensor,zip(*tokens));mask=scope['_get_mask_mod'](s,f,n,4)
checks={'cross_sample_allowed':0,'padded_allowed':0,'noisy_reads_same_or_future_clean':0,'noisy_reads_other_noisy_block':0,'clean_reads_future_clean':0,'clean_reads_noisy':0}
for q,tq in enumerate(tokens):
 for k,tk in enumerate(tokens):
  if not mask(torch.tensor(0),torch.tensor(0),q,k):continue
  if tq[0]!=tk[0]:checks['cross_sample_allowed']+=1
  if tq[0]<0 or tk[0]<0:checks['padded_allowed']+=1
  if tq[2]==0 and tk[2]==1 and tk[1]>=tq[1]:checks['noisy_reads_same_or_future_clean']+=1
  if tq[2]==0 and tk[2]==0 and tk[1]!=tq[1]:checks['noisy_reads_other_noisy_block']+=1
  if tq[2]==1 and tk[2]==1 and tk[1]>tq[1]:checks['clean_reads_future_clean']+=1
  if tq[2]==1 and tk[2]==0:checks['clean_reads_noisy']+=1
assert not any(checks.values())
assert mask(torch.tensor(0),torch.tensor(0),tokens.index((0,1,0)),tokens.index((0,0,1)))
report={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'token_count':len(tokens),'pairs':len(tokens)**2,'violations':checks,'noisy_action_can_read_clean_video_of_same_chunk':True,'scope':'scalar predicate only; not full model forward or numerical attention kernel test'}
(root/'docs/rmbench_mask_20261003.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
