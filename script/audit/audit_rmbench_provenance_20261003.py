"""Read-only CPU supplement: W&B records, checkpoint headers, episode inventory.
No wandb.init, model construction, CUDA, or writes outside this audit's docs files.
"""
import csv
import hashlib
import json
import re
import struct
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto.wandb_internal_pb2 import Record
from audit_rmbench_history import header

REPO = Path(__file__).resolve().parents[1]
KEYS = ('dataset_path validation_dataset_path bounded_window_index bounded_validation_index '
        'batch_size gradient_accumulation_steps world_size learning_rate seed resume_from action_dim '
        'action_per_frame frame_chunk_size attn_window height width wan22_pretrained_model_name_or_path '
        'episode_indices cfg_prob warmup_steps max_steps gradient_checkpointing freeze_model '
        'trainable_params use_gradient_checkpointing obs_cam_keys q01 q99').split()


def main():
    report = {'runs': [], 'headers': [], 'code_sha256': {}}
    inv = json.loads((REPO/'docs/rmbench_inventory_20261003.json').read_text())
    for root in ('train_out', 'checkpoints'):
        for log in sorted((REPO/root).rglob('training.log')):
            lines = log.read_text(errors='replace').splitlines()
            entry = {'run': str(log.parent), 'log': str(log), 'effective': {}, 'wandb_record': None}
            paths = re.findall(r'/root/wyc/code/lingbot-va2/wandb/[^\s\x1b]+', '\n'.join(lines))
            for p in dict.fromkeys(paths):
                folder = Path(p.rstrip('.,)'))
                if not folder.is_dir(): continue
                for binary in folder.glob('*.wandb'):
                    ds = DataStore(); ds.open_for_scan(str(binary))
                    try:
                        for _ in range(100):
                            raw = ds.scan_data()
                            if raw is None: break
                            record = Record(); record.ParseFromString(raw)
                            if record.HasField('run'):
                                entry['effective'] = {kv.key: json.loads(kv.value_json) for kv in record.run.config.update if kv.key in KEYS}
                                entry['wandb_record'] = str(binary)
                                entry['git_commit'] = record.run.git.commit or 'unknown'
                                break
                    except Exception as exc: entry['read_error'] = str(exc)
                    if entry['effective']: break
                if entry['effective']: break
            entry['evidence'] = [{'logical_line': i+1, 'text': s[:450]} for i,s in enumerate(lines) if any(t in s for t in ('Independent validation dataset:', 'World size:', 'Resuming from checkpoint:', 'Some weights', 'newly initialized', 'OutOfMemoryError', 'invalid device ordinal', 'KeyboardInterrupt', 'ChildFailedError', 'Error:', 'Error(', 'waiting for debugger'))]
            metrics = log.parent/'training_metrics.jsonl'
            ms = [json.loads(x) for x in metrics.read_text().splitlines() if x] if metrics.exists() else []
            entry['metric_rows'] = len(ms)
            entry['last_metric'] = ms[-1] if ms else None
            summary = log.parent/'training_summary.json'
            entry['summary'] = json.loads(summary.read_text()) if summary.exists() else None
            dates = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '\n'.join(lines))
            entry['log_times'] = [dates[0],dates[-1]] if dates else []
            report['runs'].append(entry)
    base_headers = [header(p) for p in sorted((REPO/'checkpoints/base/transformer').glob('*.safetensors'))]
    base = {'shapes': {k:v for h in base_headers for k,v in h['shapes'].items()},
            'shards': [{k:v for k,v in h.items() if k!='shapes'} for h in base_headers]}
    for root in ('train_out', 'checkpoints'):
        for path in sorted((REPO/root).rglob('*.safetensors')):
            if '/base/' in str(path): continue
            h = header(path); shapes = h.pop('shapes')
            h['missing_vs_base'] = sorted(set(base['shapes'])-set(shapes))
            h['extra_vs_base'] = sorted(set(shapes)-set(base['shapes']))
            h['shape_mismatch_vs_base'] = [k for k in shapes.keys() & base['shapes'].keys() if shapes[k] != base['shapes'][k]]
            h['training_state_files'] = [str(p) for p in path.parent.parent.iterdir() if p.is_file()]
            report['headers'].append(h)
    report['base'] = {k:v for k,v in base.items() if k!='shapes'}
    for p in list((REPO/'wan_va').rglob('*.py')) + list((REPO/'script/rmbench_data').glob('*.py')) + list((REPO/'script/rmbench_train').glob('*.sh')) + list((REPO/'script/rmbench_inference').glob('*.sh')):
        report['code_sha256'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    ep_rows=[]
    for d in inv['datasets']:
        for ep in d['episode_details']:
            for seg in ep['segments']:
                c=seg['cameras'][0] if seg['cameras'] else {}
                ep_rows.append({'dataset':d['path'],'episode':ep['episode'],'parquet_rows':ep['lerobot_rows'],'start':seg['start'],'end':seg['end'],'rgb_selected':c.get('rgb_selected','unknown'),'latent_frames':c.get('latent_frames','unknown'),'valid_actions':seg.get('valid_action_steps_expected','unknown'),'unsupervised_tail_rows':seg.get('tail_rows_not_supervised','unknown'),'cameras':len(seg['cameras']),'existing_latents':sum(x['latent_exists'] for x in seg['cameras'])})
    with (REPO/'docs/rmbench_episode_inventory_20261003.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=ep_rows[0].keys());writer.writeheader();writer.writerows(ep_rows)
    output=REPO/'docs/rmbench_provenance_20261003.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'runs_with_logs':len(report['runs']),'effective_recovered':sum(bool(r['effective']) for r in report['runs']),'checkpoint_headers':len(report['headers']),'header_integrity_pass':all(h['payload_size_matches'] and h['contiguous_offsets'] for h in report['headers']),'output':str(output)}))

if __name__=='__main__': main()
