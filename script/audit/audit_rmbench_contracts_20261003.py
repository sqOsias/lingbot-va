"""CPU-only statistics provenance, JPEG contract and tiny checkpoint tensor checks."""
import io
import json
import hashlib
from pathlib import Path
import av
import cv2
import h5py
import numpy as np
from PIL import Image
from safetensors import safe_open
from rmbench_data.recompute_rmbench_episode_origin_stats import physical16, relative16, USED_MODEL_CHANNELS

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/root/wyc/data/rmbench/lingbot_va2')

def main():
    report={'stats':[], 'color':[], 'small_tensor_updates':[], 'scope':'CPU; no model or environment rollout'}
    # Current collection pipeline: native RGB -> cv2.imencode (expects BGR) -> PIL RGB.
    rgb=np.zeros((16,16,3),np.uint8);rgb[:]=[250,20,5]
    ok,encoded=cv2.imencode('.jpg',rgb)
    decoded=np.array(Image.open(io.BytesIO(encoded.tobytes())).convert('RGB'))
    report['synthetic_jpeg']={'input_rgb':rgb[0,0].tolist(),'pil_decoded':decoded[0,0].tolist()}
    for task in ('rearrange_blocks','observe_and_pickup','blocks_ranking_try'):
        root=ROOT/f'{task}_50demo_fullctx_apf16_cf8'
        eps=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines() if x]
        values=[];raw_counts=[]
        for ep in eps:
            raw=Path(f'/root/wyc/data/rmbench/raw/{task}/demo_clean/data/episode{ep["episode_index"]}.hdf5')
            with h5py.File(raw,'r') as f: absolute=physical16(f)
            raw_counts.append({'episode':ep['episode_index'],'raw_frames':len(absolute),'parquet_rows':ep['length']})
            values.append(relative16(absolute[1:ep['length']+1],absolute[0]))
        values=np.concatenate(values);lo,hi=np.quantile(values,[.01,.99],axis=0)
        center=(lo+hi)/2;half=np.maximum.reduce([(hi-lo)/2,np.maximum(center-values.min(0),values.max(0)-center)/1.49,np.full(16,1e-6)])
        stats=json.loads((root/'rmbench_action_stats.json').read_text())
        method = 'raw training 0.01/0.99 quantiles expanded to 1.49'
        if task == 'rearrange_blocks':
            center = (values.min(0) + values.max(0))/2
            half = (values.max(0)-values.min(0))/2
            method = 'raw training min/max, matching declared quantiles 0 and 1'
        elif task == 'blocks_ranking_try':
            blo=np.array(stats['base_q01'])[USED_MODEL_CHANNELS];bhi=np.array(stats['base_q99'])[USED_MODEL_CHANNELS]
            center=(blo+bhi)/2;half=(bhi-blo)/2*np.array(stats['range_expansion'])[USED_MODEL_CHANNELS]
            method = 'stored base quantiles plus stored expansion only; original 404-window selection provenance not rederived'
        e=max(abs(np.array(stats['q01'])[USED_MODEL_CHANNELS]-(center-half)).max(),abs(np.array(stats['q99'])[USED_MODEL_CHANNELS]-(center+half)).max())
        report['stats'].append({'task':task,'episodes':len(eps),'samples':len(values),'comparison_method':method,'max_error_for_declared_method':float(e),'raw_counts':raw_counts,'declared_source':stats.get('dataset_root')})
        raw=Path(f'/root/wyc/data/rmbench/raw/{task}/demo_clean/data/episode0.hdf5')
        with h5py.File(raw,'r') as f:
            for name,source in [('cam_high','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
                encoded=f[f'observation/{source}/rgb'][0];b=encoded.tobytes() if hasattr(encoded,'tobytes') else encoded
                decoded=np.array(Image.open(io.BytesIO(b)).convert('RGB'))
                video=root/f'videos/chunk-000/observation.images.{name}/episode_000000.mp4'
                with av.open(str(video)) as con: frame=next(con.decode(video=0)).to_ndarray(format='rgb24')
                if decoded.shape!=frame.shape: decoded=cv2.resize(decoded,(frame.shape[1],frame.shape[0]))
                report['color'].append({'task':task,'camera':name,'source_shape':list(decoded.shape),'rgb24_vs_raw_pil_mae':float(abs(frame.astype(float)-decoded).mean()),'rgb24_vs_raw_pil_swapped_mae':float(abs(frame.astype(float)-decoded[...,::-1]).mean())})
    idx=json.loads((REPO/'checkpoints/base/transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    prov=json.loads((REPO/'docs/rmbench_provenance_20261003.json').read_text())
    # Read only small action output biases; this proves a parameter update, not action quality.
    key='action_proj_out.bias'
    if key not in idx: key=next(k for k in idx if 'action' in k and 'proj' in k and k.endswith('bias'))
    with safe_open(str(REPO/'checkpoints/base/transformer'/idx[key]),framework='pt',device='cpu') as f: base=f.get_tensor(key).float().numpy()
    for h in prov['headers']:
        with safe_open(h['path'],framework='pt',device='cpu') as f: value=f.get_tensor(key).float().numpy()
        report['small_tensor_updates'].append({'checkpoint':h['path'],'key':key,'shape':list(value.shape),'max_abs_delta_vs_base':float(abs(value-base).max()),'finite':bool(np.isfinite(value).all())})
    path=REPO/'docs/rmbench_contracts_20261003.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('stats','small_tensor_updates')},ensure_ascii=False))
    print('stats_errors',[(x['task'],x['max_error_for_declared_method']) for x in report['stats']])

if __name__=='__main__':main()
