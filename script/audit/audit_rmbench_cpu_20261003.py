"""CPU-only checks of the actual RMBench conversion functions and stored data.

Extracts pure functions with AST, without constructing datasets, models or envs.
Writes only its JSON report. Tensor files use restricted weights_only loading.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/root/wyc/data/rmbench/lingbot_va2')
USED = list(range(7)) + [28] + list(range(7, 14)) + [29]
INVERSE = [16] * 30
for physical, model in enumerate(USED):
    INVERSE[model] = physical


def extract(path, names):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        if isinstance(node, ast.ClassDef):
            nodes.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    if {n.name for n in nodes} != set(names):
        raise ValueError((path, names))
    scope = dict(np=np, torch=torch, R=R, rearrange=rearrange)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def main():
    source = REPO / 'wan_va/dataset/lerobot_latent_dataset.py'
    funcs = extract(source, ['get_relative_pose', '_rmbench_action_post_process'])
    client = Path('/root/wyc/code/RMBench/policy/LingBot_VA/lingbot_client.py')
    inverse = extract(client, ['delta_to_absolute_eef'])['delta_to_absolute_eef']
    rng = np.random.default_rng(42)
    pose = np.concatenate([rng.normal(size=(31, 3)), R.random(31, random_state=rng).as_quat()], axis=1)
    rel = funcs['get_relative_pose'](pose).numpy()
    restored = np.array([inverse(p, pose[0]) for p in rel])
    trans_error = float(abs(restored[:, :3] - pose[:, :3]).max())
    rot_error = float((R.from_quat(restored[:, 3:]).inv() * R.from_quat(pose[:, 3:])).magnitude().max())
    assert trans_error < 1e-12 and rot_error < 1e-12
    # Numbered labels: row t really denotes physical target t+1.
    origin = np.array([0, 0, 0, 0, 0, 0, 1, 0] * 2, dtype=np.float32)
    labels = np.tile(origin, (33, 1)); labels[:, 0] = np.arange(1, 34)
    obj = SimpleNamespace(rmbench_origins={(0, 0): origin}, config=SimpleNamespace(
        action_per_frame=16, inverse_used_action_channel_ids=INVERSE),
        q01=np.zeros((1, 30)), q99=np.full((1, 30), 100.0))
    out, mask = funcs['_rmbench_action_post_process'](obj, 0, 33, list(range(0, 33, 4)), labels, 0)
    recovered = (out.numpy()[0, :, :, 0] + 1) / 2 * (100 + 1e-6)
    assert not mask[:, 0].any() and mask[USED, 1:].all()
    assert np.allclose(recovered[1:].reshape(-1), np.arange(1, 33))
    assert not mask[[i for i in range(30) if i not in USED]].any()
    report = {'scope': 'CPU; all 180 primary fullctx episodes parquet numerics; nine latent files numerical spot check; no model forward',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, client)},
              'synthetic': {'relative_absolute_translation_max_error': trans_error,
                            'relative_absolute_rotation_max_radians': rot_error,
                            'numbered_alignment': 'dummy frame0; action frame1=physical1..16; frame2=physical17..32',
                            'shape': list(out.shape), 'valid_channels': USED}, 'datasets': []}
    server = extract(REPO/'wan_va/wan_va_server.py', ['preprocess_action', 'postprocess_action'])
    server['F'] = torch.nn.functional
    stats_path = ROOT/'rearrange_blocks_50demo_fullctx_apf16_cf8/rmbench_action_stats.json'
    norm_stats = json.loads(stats_path.read_text())
    server_obj = SimpleNamespace(job_config=SimpleNamespace(
        inverse_used_action_channel_ids=INVERSE, used_action_channel_ids=USED),
        action_norm_method='quantiles',
        actions_q01=torch.tensor(norm_stats['q01'],dtype=torch.float32).reshape(30,1,1),
        actions_q99=torch.tensor(norm_stats['q99'],dtype=torch.float32).reshape(30,1,1))
    sample = rng.normal(size=(16,2,16)).astype(np.float32)
    normed = server['preprocess_action'](server_obj, sample)
    recovered = server['postprocess_action'](server_obj, normed)
    norm_error=float(abs(recovered-sample).max())
    assert norm_error < 1e-5
    report['synthetic']['actual_server_normalize_denormalize_max_error'] = norm_error
    report['synthetic']['normalization_test_scope'] = 'random float32 values, no clipping, actual server functions; model dtype quantization not tested'
    for task in ('rearrange_blocks', 'observe_and_pickup', 'blocks_ranking_try'):
        for split, count in (('50demo', 50), ('5demo_validation', 5), ('5demo_test', 5)):
            path = ROOT / f'{task}_{split}_fullctx_apf16_cf8'
            episodes = [json.loads(x) for x in (path / 'meta/episodes.jsonl').read_text().splitlines() if x]
            anchors = json.loads((path / 'observation_anchors.json').read_text())
            stats = json.loads((path / 'rmbench_action_stats.json').read_text())
            q01, q99 = np.array(stats['q01']), np.array(stats['q99'])
            summary = {'path': str(path), 'episode_count': len(episodes), 'rows': 0, 'issues': [],
                       'action_hashes': [], 'normalization_clipped_elements': 0, 'normalization_elements': 0,
                       'masked_valid_steps': 0, 'first_label_max_error': 0., 'latent_spot_checks': [],
                       'stats_sha256': hashlib.sha256((path / 'rmbench_action_stats.json').read_bytes()).hexdigest()}
            for ep in episodes:
                eid = ep['episode_index']; ac = ep['action_config'][0]
                table = pq.read_table(path / f'data/chunk-000/episode_{eid:06d}.parquet')
                raw = np.array(table['action'].to_pylist(), dtype=np.float32)
                summary['rows'] += len(raw)
                summary['action_hashes'].append(hashlib.sha256(raw.tobytes()).hexdigest())
                if raw.shape != (ep['length'], 16) or not np.isfinite(raw).all():
                    summary['issues'].append({'episode': eid, 'bad_action_shape_or_nonfinite': list(raw.shape)})
                for offset in (0, 8):
                    norm = np.linalg.norm(raw[:, offset+3:offset+7], axis=1)
                    if not np.allclose(norm, 1, atol=1e-4):
                        summary['issues'].append({'episode': eid, 'bad_quaternion_norm': [float(norm.min()), float(norm.max())]})
                for col in ('episode_index', 'frame_index'):
                    values = np.array(table[col].to_pylist())
                    expected = np.full(len(raw), eid) if col == 'episode_index' else np.arange(len(raw))
                    if not np.array_equal(values, expected):
                        summary['issues'].append({'episode': eid, 'bad_index': col})
                anchor = anchors[f'{eid}:0']
                summary['first_label_max_error'] = max(summary['first_label_max_error'], float(abs(raw[0]-np.array(anchor['first_label'])).max()))
                ori = np.array(anchor['origin'], dtype=np.float32)
                obj.rmbench_origins = {(eid, 0): ori}; obj.q01=q01[None];obj.q99=q99[None]
                ids = list(range(0, ac['end_frame'], 4)); ids=ids[:1+4*((len(ids)-1)//4)]
                actions, masks = funcs['_rmbench_action_post_process'](obj,0,ac['end_frame'],ids,raw,eid)
                expected_steps = min(ac['end_frame']-1, ((len(ids)-1)//4)*16)
                assert masks[0].sum().item() == expected_steps and not masks[:,0].any()
                assert actions.shape == (30,(len(ids)-1)//4+1,16,1)
                summary['masked_valid_steps'] += expected_steps
                relative = np.concatenate([np.concatenate([funcs['get_relative_pose'](np.concatenate([ori[None,o:o+7],raw[:,o:o+7]]))[1:].numpy(),raw[:,o+7:o+8]],axis=1) for o in (0,8)],axis=1)
                normalized = (relative-q01[USED])/(q99[USED]-q01[USED]+1e-6)*2-1
                summary['normalization_clipped_elements'] += int((abs(normalized)>1.5).sum())
                summary['normalization_elements'] += normalized.size
                if eid == 0 and split == '50demo':
                    # Directly verify conversion row t against raw physical t+1.
                    rawfile = Path(f'/root/wyc/data/rmbench/raw/{task}/demo_clean/data/episode0.hdf5')
                    if rawfile.exists():
                        with h5py.File(rawfile,'r') as h5:
                            physical=[]
                            for arm in ('left','right'):
                                p=np.array(h5[f'endpose/{arm}_endpose']); g=np.array(h5[f'endpose/{arm}_gripper'])[:,None]
                                physical.extend([np.concatenate([p[:,:3],p[:,4:7],p[:,3:4]],axis=1),g])
                            physical=np.concatenate(physical,axis=1)
                        summary['raw_episode0_alignment_max_error']=float(abs(raw-physical[1:len(raw)+1]).max())
                        assert summary['raw_episode0_alignment_max_error'] < 1e-6
                    for camera in ('cam_high','cam_left_wrist','cam_right_wrist'):
                        latent=path / f'latents/chunk-000/observation.images.{camera}/episode_{eid:06d}_0_{ac["end_frame"]}.pth'
                        payload=torch.load(latent,map_location='cpu',weights_only=True,mmap=True)
                        summary['latent_spot_checks'].append({'path':str(latent),
                            'finite_latent':bool(torch.isfinite(payload['latent']).all()),
                            'finite_text':bool(torch.isfinite(payload['text_emb']).all()),
                            'latent_std':float(payload['latent'].float().std())})
            assert len(episodes)==count
            report['datasets'].append(summary)
    report['split_action_duplicates'] = {}
    for task in ('rearrange_blocks', 'observe_and_pickup', 'blocks_ranking_try'):
        sets = [set(d['action_hashes']) for d in report['datasets'] if Path(d['path']).name.startswith(task+'_')]
        report['split_action_duplicates'][task] = [len(sets[i] & sets[j]) for i,j in ((0,1),(0,2),(1,2))]
        assert report['split_action_duplicates'][task] == [0,0,0]
    # Execute exactly the actual cache allocation/write/restore pure methods.
    cache_funcs=extract(REPO/'wan_va/modules/model.py', ['init_kv_cache','allocate_slots','_next_cache_id','update_cache','restore_cache'])
    class Cache: pass
    for name in ('init_kv_cache','allocate_slots','_next_cache_id','update_cache','restore_cache'):
        setattr(Cache,name,cache_funcs[name])
    cache=Cache();cache.attn_caches={}
    cache.init_kv_cache('pos',4,1,1,'cpu',torch.float32,1)
    cache.update_cache('pos',torch.arange(4.).reshape(1,4,1,1),torch.zeros(1,4,1,1),False)
    before=cache.attn_caches['pos']['k'].clone()
    slots=cache.update_cache('pos',torch.full((1,2,1,1),99.),torch.zeros(1,2,1,1),False)
    cache.restore_cache('pos',slots)
    report['cache_readonly_reproduction']={'capacity':4,'live_before':4,
        'live_after_readonly_write_restore':int(cache.attn_caches['pos']['mask'].sum()),
        'old_payload_overwritten':not torch.equal(before,cache.attn_caches['pos']['k']),
        'source':'wan_va/modules/model.py:368-414, forward:447-463'}
    assert report['cache_readonly_reproduction']['live_after_readonly_write_restore']==2
    target=REPO/'docs/rmbench_cpu_checks_20261003.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'output':str(target),'datasets':len(report['datasets']),
        'episodes':sum(d['episode_count'] for d in report['datasets']),
        'issues':sum(len(d['issues']) for d in report['datasets']),
        'cache':report['cache_readonly_reproduction']},ensure_ascii=False))


if __name__ == '__main__':
    main()
