"""Paired CPU-offloaded inference diagnostics; no training or policy edits."""
import os, sys, copy, json, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from diffusers.video_processor import VideoProcessor
from infer_rearrange_episode0_offline import sample_chunk as original_sample_chunk, commit_history_cache, physical_actions, seed_everything
from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset
from wan_va.distributed.util import init_distributed
from wan_va.wan_va_server import VA_Server


from einops import rearrange
@torch.no_grad()
def sample_chunk(model, target_latents, target_actions, chunk_id):
    target_latents=target_latents.to(model.device,model.dtype)
    target_actions=target_actions.to(model.device,model.dtype)
    torch.randn_like(target_latents) # match the paired video-noise draw
    actions=torch.randn_like(target_actions)
    frame_st_id=chunk_id*2
    model.transformer.clear_pred_cache(model.cache_name)
    vi=model._prepare_latent_input(target_latents,None,frame_st_id=frame_st_id)['latent_res_lst']
    model.transformer(model._repeat_input_for_cfg(vi),update_cache=1,cache_name=model.cache_name,action_mode=False)
    model.action_scheduler.set_timesteps(50)
    ts=torch.nn.functional.pad(model.action_scheduler.timesteps,(0,1),value=0)
    for i,t in enumerate(ts):
        cond=torch.zeros_like(actions[:,:,:1]) if chunk_id==0 else None
        ai=model._prepare_latent_input(None,actions,t,t,None,cond,frame_st_id=frame_st_id)['action_res_lst']
        pred=model.transformer(model._repeat_input_for_cfg(ai),update_cache=1 if i==len(ts)-1 else 0,cache_name=model.cache_name,action_mode=True)
        if i<len(ts)-1:
            pred=rearrange(pred,'b (f n) c -> b c f n 1',f=2)
            pred=pred[1:]+model.job_config.action_guidance_scale*(pred[:1]-pred[1:]) if model.job_config.action_guidance_scale>1 else pred[:1]
            actions=model.action_scheduler.step(pred,t,actions,return_dict=False)
        if cond is not None: actions[:,:,:1]=cond
    actions[:,~model.action_mask]=0
    return target_latents.cpu(),actions.cpu()

repo=Path('/root/wyc/code/lingbot-va2')
dataset_path=Path('/root/wyc/data/rmbench/lingbot_va2/rearrange_blocks_50demo_w193_apf16_official')
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
init_distributed(world,local,rank)
cfg=copy.deepcopy(VA_CONFIGS['rmbench_overfit_train'])
cfg.rank=rank; cfg.local_rank=local; cfg.world_size=world
cfg.dataset_path=str(dataset_path); cfg.empty_emb_path=str(dataset_path/'empty_emb.pt')
stats=json.loads((dataset_path/'rmbench_action_stats.json').read_text())
cfg.norm_stat={k:stats[k] for k in ('q01','q99')}
cfg.episode_indices=[0]; cfg.cfg_prob=0.; cfg.load_worker=0; cfg.enable_wandb=False
cfg.enable_offload=True; cfg.fsdp_cpu_offload=True
cfg.wan22_pretrained_model_name_or_path=str(repo/'artifacts/rearrange_blocks_episode0_offline_20260916_232251/model')
cfg.save_root=str(out); cfg.num_inference_steps=25; cfg.action_num_inference_steps=50
cfg.frame_chunk_size=2; cfg.attn_window=72
ds=LatentLeRobotDataset(str(dataset_path),cfg); samples=[ds[i] for i in range(len(ds))]
vl=torch.cat([samples[0]['latents']]+[s['latents'][:,1:] for s in samples[1:]],dim=1).unsqueeze(0)
aa=torch.cat([samples[0]['actions']]+[s['actions'][:,1:] for s in samples[1:]],dim=1).unsqueeze(0)
saved=repo/'artifacts/rearrange_blocks_episode0_offline_predicted_20260917_093905'
hist=[torch.load(saved/f'prediction_chunk_{i:02d}.pt',map_location='cpu',weights_only=False) for i in range(7)]
seed_everything(42+rank); model=VA_Server(cfg)
model.use_cfg=cfg.guidance_scale>1 or cfg.action_guidance_scale>1
model.empty_emb=torch.load(cfg.empty_emb_path,map_location='cpu',weights_only=False)
model.video_processor=VideoProcessor(vae_scale_factor=1)
q1=np.asarray(stats['q01'],dtype=np.float32); q9=np.asarray(stats['q99'],dtype=np.float32); used=list(cfg.used_action_channel_ids)
results=[]
def reset():
    model._reset(prompt=None)
    model.prompt_embeds=samples[0]['text_emb'].unsqueeze(0).to(model.device,model.dtype)
    model.negative_prompt_embeds=model.empty_emb.unsqueeze(0).to(model.device,model.dtype)
def record(name,pred,target,start_frame,drop=0):
    p=physical_actions(pred,q1,q9,used).transpose(1,2,0).reshape(-1,16)[drop:]
    t=physical_actions(target,q1,q9,used).transpose(1,2,0).reshape(-1,16)[drop:]
    frames=np.arange(start_frame,start_frame+len(p))
    report={'name':name,'frames':frames.tolist(),'pred':p.tolist(),'target':t.tolist(),'segments':[]}
    for lo,hi in [(193,200),(201,208),(209,216),(217,224)]:
        ix=(frames>=lo)&(frames<=hi)
        if ix.any():
            e=p[ix,:3]-t[ix,:3]
            report['segments'].append({'frames':[lo,hi],'l2_mae_mm':float(np.linalg.norm(e,axis=1).mean()*1000),'z_mae_mm':float(np.abs(e[:,2]).mean()*1000),'pred_min_z_relative':float(p[ix,2].min()),'target_min_z_relative':float(t[ix,2].min())})
    if rank==0:
        results.append(report)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print('RESULT '+json.dumps({k:v for k,v in report.items() if k not in ('pred','target','frames')}),flush=True)
for chunk in (6,):
    for video_source,action_source in [('expert','expert'),('predicted','predicted')]:
        name=f'chunk{chunk}_V{video_source}_A{action_source}'
        if rank==0: print('CASE '+name,flush=True)
        reset()
        for j in range(chunk):
            v=vl[:,:,2*j:2*j+2] if video_source=='expert' else hist[j]['latents']
            a=aa[:,:,2*j:2*j+2] if action_source=='expert' else hist[j]['actions']
            commit_history_cache(model,v,a,2*j)
        # Identical per-rank noise for all paired cases of the same target chunk.
        seed_everything(17000+chunk+rank)
        v,a=sample_chunk(model,vl[:,:,2*chunk:2*chunk+2],aa[:,:,2*chunk:2*chunk+2],chunk)
        if rank==0: torch.save({'latents':v,'actions':a},out/(name+'.pt'))
        record(name,a,aa[:,:,2*chunk:2*chunk+2],(2*chunk-1)*16+1)
        dist.barrier()

if rank==0: (out/'DONE').write_text('complete\n')
dist.barrier(); dist.destroy_process_group()
