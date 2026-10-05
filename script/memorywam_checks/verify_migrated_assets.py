"""CPU-only structural check for a migrated MemoryWAM workspace.

The check reads JSON/text metadata and file sizes only.  It does not initialize
CUDA and does not load model or latent tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


INITIALIZER_WEIGHT_BYTES = 10_177_831_668
DATASETS = {
    "train": ("rearrange_blocks_50demo_fullctx_apf16_cf8", 50, 150),
    "validation": (
        "rearrange_blocks_5demo_validation_fullctx_apf16_cf8", 5, 15),
    "test": ("rearrange_blocks_5demo_test_fullctx_apf16_cf8", 5, 15),
}


def _record_file(path: Path, *, exact_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if exact_bytes is not None and size != exact_bytes:
        raise ValueError(f"{path}: expected {exact_bytes} bytes, found {size}")
    return {"path": str(path.resolve()), "bytes": size}


def _json_lines(path: Path) -> int:
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def _dataset(root: Path, key: str) -> dict[str, Any]:
    name, expected_episodes, expected_latents = DATASETS[key]
    dataset = root / name
    required = (
        "continuous_contract.json",
        "empty_emb.pt",
        "meta/episodes.jsonl",
        "meta/info.json",
        "rmbench_action_stats.json",
    )
    for relative in required:
        _record_file(dataset / relative)
    episodes = _json_lines(dataset / "meta/episodes.jsonl")
    latents = sum(1 for _ in (dataset / "latents").rglob("*.pth"))
    if episodes != expected_episodes:
        raise ValueError(
            f"{dataset}: expected {expected_episodes} episodes, found {episodes}")
    if latents != expected_latents:
        raise ValueError(
            f"{dataset}: expected {expected_latents} latents, found {latents}")
    return {
        "path": str(dataset.resolve()),
        "episodes": episodes,
        "latent_files": latents,
        "files": sum(1 for item in dataset.rglob("*") if item.is_file()),
    }


def _initializer(root: Path) -> dict[str, Any]:
    transformer = root / "transformer"
    config_path = transformer / "config.json"
    weight_path = transformer / "diffusion_pytorch_model.safetensors"
    config = json.loads(config_path.read_text())
    _record_file(weight_path, exact_bytes=INITIALIZER_WEIGHT_BYTES)
    expected = {
        "num_layers": 30,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "in_channels": 48,
        "out_channels": 48,
        "action_dim": 30,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"initializer config mismatch: {mismatches}")
    return {
        "path": str(root.resolve()),
        "weight_bytes": weight_path.stat().st_size,
        "config": {key: config[key] for key in expected},
    }


def _base_model(root: Path) -> dict[str, Any]:
    required = (
        "configuration.json",
        "tokenizer/tokenizer.json",
        "tokenizer/spiece.model",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00003.safetensors",
        "text_encoder/model-00002-of-00003.safetensors",
        "text_encoder/model-00003-of-00003.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    files = [_record_file(root / relative) for relative in required]
    return {
        "path": str(root.resolve()),
        "required_files": len(files),
        "required_bytes": sum(item["bytes"] for item in files),
    }


def _closed_loop(data_root: Path, rmbench_root: Path) -> dict[str, Any]:
    holdout = data_root / "holdout_work/rearrange_blocks"
    for split in ("validation", "test"):
        manifest = holdout / split / "expert_seeds.json"
        value = json.loads(manifest.read_text())
        if value.get("task") != "rearrange_blocks" or not value.get("complete"):
            raise ValueError(f"invalid expert seed manifest: {manifest}")
        if len(value.get("seeds", ())) != 5:
            raise ValueError(f"expected five seeds: {manifest}")
        hdf5 = sorted((holdout / split / "raw/data").glob("episode*.hdf5"))
        if len(hdf5) != 5:
            raise ValueError(f"expected five HDF5 demonstrations: {split}")
    required_rmbench = (
        "script/eval_policy.py",
        "policy/LingBot_VA/deploy_policy.py",
        "policy/LingBot_VA/lingbot_client.py",
        "envs/rearrange_blocks.py",
    )
    for relative in required_rmbench:
        _record_file(rmbench_root / relative)
    commit = subprocess.check_output(
        ["git", "-C", str(rmbench_root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "holdout": str(holdout.resolve()),
        "rmbench": str(rmbench_root.resolve()),
        "rmbench_head": commit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get(
            "RMBENCH_DATA_ROOT", "/root/wyc/data/rmbench/lingbot_va2")))
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--rmbench-root", type=Path)
    parser.add_argument(
        "--scope", choices=("training", "inference", "closed-loop"),
        default="training")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    initializer = (args.initializer or repo / "train_out" /
                   "rearrange_blocks_50demo_1000step_20260921_180241" /
                   "checkpoints/checkpoint_step_750")
    base_model = args.base_model or repo / "checkpoints/base"
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "memorywam_migrated_asset_check",
        "scope": args.scope,
        "status": "PASS",
        "initializer": _initializer(initializer),
        "datasets": {
            "train": _dataset(args.data_root, "train"),
            "validation": _dataset(args.data_root, "validation"),
        },
    }
    if args.scope in ("inference", "closed-loop"):
        report["base_model"] = _base_model(base_model)
    if args.scope == "closed-loop":
        report["datasets"]["test"] = _dataset(args.data_root, "test")
        if args.rmbench_root is None:
            parser.error("--rmbench-root is required for --scope closed-loop")
        report["closed_loop"] = _closed_loop(
            args.data_root, args.rmbench_root.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2,
                          sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
