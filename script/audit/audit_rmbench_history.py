"""Read-only history/fullctx audit; writes only an explicitly chosen report JSON.

Run with the existing lingbot Python environment. No model import, CUDA, training,
environment construction, or dataset modification is performed.
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        raw = stream.read(size)
    data = json.loads(raw)
    tensors = {k: v for k, v in data.items() if k != "__metadata__"}
    spans = sorted(v["data_offsets"] for v in tensors.values())
    contiguous = all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    end = spans[-1][1] if spans else 0
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_count": len(tensors),
        "tensor_elements": sum(math.prod(v["shape"]) for v in tensors.values()),
        "payload_size_matches": path.stat().st_size == 8 + size + end,
        "contiguous_offsets": contiguous and (not spans or spans[0][0] == 0),
        "shapes": {k: v["shape"] for k, v in tensors.items()},
        "dtypes": sorted({v["dtype"] for v in tensors.values()}),
    }


def runs(repo):
    import yaml
    result = []
    config_keys = ["dataset_path", "validation_dataset_path", "bounded_window_index",
                   "bounded_validation_index", "batch_size", "gradient_accumulation_steps",
                   "world_size", "learning_rate", "seed", "resume_from", "action_dim",
                   "action_per_frame", "frame_chunk_size", "attn_window", "height", "width",
                   "wan22_pretrained_model_name_or_path", "episode_indices", "cfg_prob"]
    for path in sorted((repo / "train_out").rglob("run_config.json")):
        run = path.parent
        log_path = run / "training.log"
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        lines = log.splitlines()  # progress CRs count as separate logical lines
        metrics = rows(run / "training_metrics.jsonl")
        validation = rows(run / "validation_metrics.jsonl")
        effective = None
        effective_path = None
        for match in re.findall(r"/root/wyc/code/lingbot-va2/wandb/[^\s]+", log):
            candidate = Path(match) / "files/config.yaml"
            if candidate.exists():
                contents = yaml.safe_load(candidate.read_text())
                effective = {k: contents[k].get("value") for k in config_keys if k in contents}
                effective_path = str(candidate)
                break
        evidence = []
        phrases = ["World size:", "Independent validation dataset:", "Resuming from checkpoint:",
                   "Some weights", "were newly initialized", "Starting training for",
                   "Checkpoint saved successfully", "Early stopping", "OutOfMemoryError",
                   "invalid device ordinal", "ChildFailedError", "KeyboardInterrupt",
                   "waiting for debugger"]
        for i, line in enumerate(lines):
            if any(phrase in line for phrase in phrases):
                evidence.append({"logical_line": i + 1, "text": line[:700]})
                if "Some weights" in line and i + 1 < len(lines):
                    evidence.append({"logical_line": i + 2, "text": lines[i + 1][:700]})
        checkpoints = []
        for weight in sorted(run.rglob("*.safetensors")):
            h = header(weight)
            h.pop("shapes")
            h["config"] = read_json(weight.parent / "config.json")
            h["training_state_exists"] = (weight.parent.parent / "training_state.pt").exists()
            checkpoints.append(h)
        timestamps = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", log)
        result.append({
            "run": str(run), "run_config": read_json(path),
            "effective_config": effective, "effective_config_path": effective_path,
            "summary": read_json(run / "training_summary.json"),
            "metric_count": len(metrics), "last_logged_step": max(
                [m.get("optimizer_step", 0) for m in metrics], default=0),
            "metric_first": metrics[0] if metrics else None,
            "metric_last": metrics[-1] if metrics else None,
            "metrics_finite": all(math.isfinite(v) for m in metrics for v in m.values()
                                  if isinstance(v, (int, float))),
            "steps_contiguous": [m.get("optimizer_step") for m in metrics] == list(range(1, len(metrics) + 1)),
            "validation_last": validation[-1] if validation else None,
            "validation_count": len(validation), "checkpoints": checkpoints,
            "log_first_time": timestamps[0] if timestamps else None,
            "log_last_time": timestamps[-1] if timestamps else None,
            "log_evidence": evidence,
        })
    return result


def datasets(root, check_latent_metadata):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    result = []
    for path in sorted(root.iterdir()):
        info_path = path / "meta/info.json"
        if not info_path.exists():
            continue
        info = read_json(info_path)
        episodes = rows(path / "meta/episodes.jsonl")
        cameras = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
        extraction = read_json(path / "latent_extraction_manifest.json")
        contract = read_json(path / "continuous_contract.json") or read_json(path / "full_context_contract.json")
        fullctx = "fullctx" in path.name
        issues = []
        details = []
        for ep in episodes:
            eid = ep["episode_index"]
            chunk = eid // info.get("chunks_size", 1000)
            parquet = path / info["data_path"].format(episode_chunk=chunk, episode_index=eid)
            if not parquet.exists():
                issues.append({"episode": eid, "missing_parquet": str(parquet)})
            record = {"episode": eid, "lerobot_rows": ep["length"], "segments": []}
            for segment in ep.get("action_config", []):
                start, end = segment["start_frame"], segment["end_frame"]
                item = {"start": start, "end": end, "action_text": segment.get("action_text"), "cameras": []}
                for camera in cameras:
                    latent = path / "latents" / f"chunk-{chunk:03d}" / camera / f"episode_{eid:06d}_{start}_{end}.pth"
                    video = path / info["video_path"].format(episode_chunk=chunk, episode_index=eid, video_key=camera)
                    cr = {"camera": camera, "latent": str(latent), "latent_exists": latent.exists(),
                          "video_exists": video.exists()}
                    if not latent.exists():
                        issues.append({"episode": eid, "missing_latent": str(latent)})
                    elif fullctx and check_latent_metadata:
                        try:
                            with FakeTensorMode():
                                d = torch.load(latent, map_location="cpu", weights_only=True, mmap=True)
                            ids = list(d["frame_ids"])
                            f, h, w = d["latent_num_frames"], d["latent_height"], d["latent_width"]
                            expected_ids = list(range(start, end, 4))
                            expected_ids = expected_ids[:1 + 4 * ((len(expected_ids) - 1) // 4)]
                            checks = {
                                "frame_ids_match": ids == expected_ids,
                                "shape_match": tuple(d["latent"].shape) == (f*h*w, 48),
                                "latent_count_match": f == (len(ids)-1)//4+1,
                                "text_shape_match": tuple(d["text_emb"].shape) == (512, 4096),
                                "text_match": d.get("text") == segment.get("action_text"),
                                "source_range_match": d.get("start_frame") == start and d.get("end_frame") == end,
                            }
                            cr.update({"checks": checks, "latent_frames": f, "latent_hw": [h, w],
                                       "rgb_selected": len(ids), "last_rgb_id": ids[-1],
                                       "text": d.get("text"), "bytes": latent.stat().st_size})
                            if not all(checks.values()):
                                issues.append({"episode": eid, "latent_checks": cr})
                        except Exception as exc:
                            issues.append({"episode": eid, "latent_read_error": str(exc), "path": str(latent)})
                    item["cameras"].append(cr)
                if fullctx and item["cameras"] and "latent_frames" in item["cameras"][0]:
                    item["valid_action_steps_expected"] = min(end-start-1, (item["cameras"][0]["latent_frames"]-1)*16)
                    item["tail_rows_not_supervised"] = ep["length"]-start-item["valid_action_steps_expected"]
                record["segments"].append(item)
            details.append(record)
        if len({ep["episode_index"] for ep in episodes}) != len(episodes):
            issues.append("duplicate episode ids")
        result.append({"path": str(path), "resolved_path": str(path.resolve()), "fullctx": fullctx,
                       "episodes": len(episodes), "segments": sum(len(e.get("action_config", [])) for e in episodes),
                       "info_frames": info.get("total_frames"), "actual_metadata_frames": sum(e["length"] for e in episodes),
                       "fps": info.get("fps"), "robot_type": info.get("robot_type"),
                       "action_shape": info["features"].get("action", {}).get("shape"),
                       "extraction": {k:v for k,v in (extraction or {}).items() if k != "files"},
                       "contract_scalars": {k:v for k,v in (contract or {}).items() if not isinstance(v,(list,dict))},
                       "holdout": read_json(path / "holdout_manifest.json"), "issues": issues, "episode_details": details})
    return result


def evaluations(repo):
    result=[]
    for path in sorted((repo / "artifacts").rglob("_result.txt")):
        run = path.parents[1]
        model = run / "model/transformer"
        text=path.read_text()
        match=re.search(r"Success Rate:\s*([0-9.]+)",text)
        result.append({"path":str(path),"result_text":text,
                       "success_rate":float(match.group(1)) if match else None,
                       "model_resolved":str(model.resolve()) if model.exists() else None,
                       "checkpoint_exists":model.exists()})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root",type=Path,default=Path("/root/wyc/data/rmbench/lingbot_va2"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--latent-metadata",action="store_true")
    args=parser.parse_args()
    report={"scope":"CPU metadata/header audit, no tensor numeric validation", "runs":runs(args.repo),
            "datasets":datasets(args.data_root,args.latent_metadata),"evaluations":evaluations(args.repo)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"runs":len(report["runs"]),"datasets":len(report["datasets"]),
                      "fullctx_datasets":sum(d["fullctx"] for d in report["datasets"]),
                      "fullctx_issues":sum(len(d["issues"]) for d in report["datasets"] if d["fullctx"]),
                      "output":str(args.output)},ensure_ascii=False))


if __name__ == "__main__":
    main()
