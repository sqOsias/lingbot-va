#!/usr/bin/env python3
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path
import numpy as np

repo=Path('/root/wyc/code/lingbot-va2')
dataset=Path(os.environ['RMBENCH_DATASET_PATH'])
sys.path.insert(0,str(repo))
from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset

cfg=copy.deepcopy(VA_CONFIGS['rmbench_overfit_train'])
cfg.dataset_path=str(dataset); cfg.empty_emb_path=str(dataset/'empty_emb.pt'); cfg.cfg_prob=0.; cfg.load_worker=0
with (dataset/'rmbench_action_stats.json').open() as f: stats=json.load(f)
cfg.norm_stat={k:stats[k] for k in ('q01','q99')}
ds=LatentLeRobotDataset(str(dataset),cfg)
assert len(ds)==1, len(ds)
s=ds[0]; used=np.asarray(cfg.used_action_channel_ids)
act=s['actions'].numpy(); mask=s['actions_mask'].numpy()
rows=np.moveaxis(mask,0,-1).reshape(-1,30)[:,used].all(1)
latent_meta=[]
for key in cfg.obs_cam_keys:
    p=dataset/'latents/chunk-000'/key/'episode_000000_0_401.pth'
    import torch
    x=torch.load(p,map_location='cpu',weights_only=False)
    latent_meta.append({'camera':key,'path':str(p),'latent_num_frames':x['latent_num_frames'],'frame_ids':x['frame_ids'],'shape':list(s['latents'].shape)})
report={
  'status':'PASS', 'dataset_windows':len(ds), 'action_shape':list(act.shape),
  'latent_shape':list(s['latents'].shape), 'valid_action_rows':int(rows.sum()),
  'masked_action_rows':int((~rows).sum()), 'first_16_masked':bool((~rows[:16]).all()),
  'remaining_400_valid':bool(rows[16:416].all()), 'latents':latent_meta,
  'checks':{'one_window':len(ds)==1,'26_latents':s['latents'].shape[1]==26,
    '416_action_slots':act.shape[1]*act.shape[2]==416,'400_valid_actions':int(rows.sum())==400,
    'frame_ids_0_to_400':all(x['frame_ids']==list(range(0,401,4)) for x in latent_meta)}
}
report['status']='PASS' if all(report['checks'].values()) else 'FAIL'
out=dataset/'full_context_audit.json'; out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(0 if report['status']=='PASS' else 1)
