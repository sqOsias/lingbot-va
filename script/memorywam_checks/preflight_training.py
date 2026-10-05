"""CPU-only preflight for the first production MemoryWAM GPU smoke.

The script never loads checkpoint tensor contents.  It compares safetensors
headers against a meta-device model, validates that enabling MemoryWAM adds
only the declared gist parameter, and reruns the lossless production Dataset
regression for train and validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from wan_va.modules.model import WanTransformer3DModel

from .verify_production_dataset import verify


RELEVANT_CODE = (
    "wan_va/configs/va_rmbench_train_cfg.py",
    "wan_va/dataset/lerobot_latent_dataset.py",
    "wan_va/distributed/fsdp.py",
    "wan_va/modules/memorywam.py",
    "wan_va/modules/model.py",
    "wan_va/train.py",
    "wan_va/wan_va_server.py",
    "script/memorywam_checks/preflight_training.py",
    "script/memorywam_checks/verify_production_dataset.py",
    "script/rmbench_train/train_memorywam_rearrange.sh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(repo: Path) -> str:
    digest = hashlib.sha256()
    for relative in RELEVANT_CODE:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_header(transformer_root: Path) -> dict:
    config_path = transformer_root / "config.json"
    weight_path = transformer_root / "diffusion_pytorch_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(
            f"Expected consolidated transformer checkpoint in {transformer_root}")
    config = json.loads(config_path.read_text())
    constructor = dict(config)
    for key in ("_class_name", "_diffusers_version", "_name_or_path",
                "attn_mode"):
        constructor.pop(key, None)
    with torch.device("meta"):
        model = WanTransformer3DModel(**constructor)
    expected = {key: list(value.shape)
                for key, value in model.state_dict().items()}
    with safe_open(weight_path, framework="pt", device="cpu") as stream:
        actual = {key: list(stream.get_slice(key).get_shape())
                  for key in stream.keys()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatch = sorted(
        {key: {"model": expected[key], "checkpoint": actual[key]}
         for key in set(expected) & set(actual)
         if expected[key] != actual[key]}.items())
    if missing or unexpected or mismatch:
        raise ValueError({"missing": missing, "unexpected": unexpected,
                          "shape_mismatch": mismatch})

    before = set(model.state_dict())
    model.enable_memorywam(gist_tokens=8, anchor_frames=2, recent_frames=4)
    after = model.state_dict()
    additions = sorted(set(after) - before)
    if additions != ["gist_tokens"]:
        raise ValueError(("unexpected MemoryWAM parameters", additions))
    if list(after["gist_tokens"].shape) != [8, 3072]:
        raise ValueError(("gist shape", list(after["gist_tokens"].shape)))
    return {
        "root": str(transformer_root.resolve()),
        "config_sha256": _sha256(config_path),
        "weight_bytes": weight_path.stat().st_size,
        "legacy_keys": len(actual),
        "legacy_header_exact_match": True,
        "memorywam_new_parameters": additions,
        "gist_shape": list(after["gist_tokens"].shape),
    }


def _alignment_summary(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if not records:
        raise ValueError(f"Empty alignment: {path}")
    identities = [(int(item["episode_index"]),
                   int(item["segment_start_frame"]),
                   int(item["segment_end_frame"])) for item in records]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate alignment identity: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "records": len(records),
        "decisions": sum(len(item["decisions"]) for item in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initializer", type=Path, required=True,
                        help="Checkpoint root containing transformer/")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--train-alignment", type=Path, required=True)
    parser.add_argument("--validation-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = _checkpoint_header(args.initializer / "transformer")
    train_alignment = _alignment_summary(args.train_alignment)
    validation_alignment = _alignment_summary(args.validation_alignment)
    train = verify(args.train_dataset, args.train_alignment)
    validation = verify(args.validation_dataset, args.validation_alignment)
    report = {
        "schema_version": 1,
        "kind": "memorywam_training_cpu_preflight",
        "status": "PASS",
        "code_fingerprint": code_fingerprint(
            Path(__file__).resolve().parents[2]),
        "code_files": list(RELEVANT_CODE),
        "checkpoint": checkpoint,
        "policy": {"gist_tokens": 8, "anchor_frames": 2,
                   "recent_frames": 4, "detach_interval": 0},
        "train_alignment": train_alignment,
        "validation_alignment": validation_alignment,
        "train_dataset": {key: train[key] for key in (
            "dataset_root", "episodes_checked", "latent_frames_checked",
            "valid_action_slots_checked", "legacy_tensors_bitwise_equal")},
        "validation_dataset": {key: validation[key] for key in (
            "dataset_root", "episodes_checked", "latent_frames_checked",
            "valid_action_slots_checked", "legacy_tensors_bitwise_equal")},
        "gpu_checks_deferred": [
            "5B memory-off numerical regression",
            "5B FSDP forward/backward and optimizer update",
            "FSDP checkpoint save/reload",
            "peak VRAM and step latency",
            "online streaming VAE and websocket RMBench round trip",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
