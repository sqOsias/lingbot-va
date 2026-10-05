"""不加载模型地冻结 legacy 代码、配置、资产和既有输出。

静态快照与“当前代码重新生成的数值 reference”分开记录。只有后者的
``inference_provenance=current_snapshot`` 才能用于逐值回归。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(["git", *arguments], cwd=repo,
                             text=True, capture_output=True, check=False)
    if process.returncode:
        raise RuntimeError(process.stderr.strip())
    return process.stdout


def _tree(root: Path, *, hash_large: bool,
          always_hash: bool = False) -> list[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*")
                                                 if p.is_file())
    result = []
    for path in paths:
        stat = path.stat()
        item: dict[str, Any] = {
            "path": str(path.relative_to(root.parent)),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if always_hash or hash_large or stat.st_size <= 128 * 1024 * 1024:
            item["sha256"] = _sha256(path)
        else:
            item["sha256"] = None
        result.append(item)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _effective_configs(repo: Path, names: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(repo))
    from wan_va.configs import VA_CONFIGS
    return {name: _jsonable(VA_CONFIGS[name]) for name in names}


def capture(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    status = _git(repo, "status", "--short")
    diff = _git(repo, "diff", "--binary")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "working_tree.patch").write_text(diff)
    code_roots = [repo / "wan_va", repo / "script/memorywam_checks"]
    code = {str(path.relative_to(repo)): _tree(path, hash_large=False,
                                               always_hash=True)
            for path in code_roots}
    inference = {
        str(path.resolve()): _tree(path.resolve(), hash_large=True,
                                   always_hash=True)
        for path in args.inference_output
    }
    manifest = {
        "schema_version": 1,
        "kind": "legacy_regression_baseline",
        "git": {
            "commit": _git(repo, "rev-parse", "HEAD").strip(),
            "status": status,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(("RMBENCH_", "LINGBOT_", "CUDA_", "PYTORCH_"))
        },
        "effective_configs": _effective_configs(repo, args.config),
        "code": code,
        "checkpoint": {
            "root": str(args.checkpoint.resolve()),
            "files": _tree(args.checkpoint.resolve(),
                           hash_large=args.hash_large_files),
            "all_files_content_hashed": args.hash_large_files,
        },
        "dataset": {
            "root": str(args.dataset.resolve()),
            "files": _tree(args.dataset.resolve(),
                           hash_large=args.hash_large_files),
            "all_files_content_hashed": args.hash_large_files,
        },
        "inference_outputs": inference,
        "inference_provenance": args.inference_provenance,
        "numeric_regression_ready": bool(
            inference and args.inference_provenance == "current_snapshot"
        ),
        "note": (
            "A historical output is provenance only. Run the frozen current code on a "
            "fixed request before using this manifest as a numerical regression oracle."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--inference-output", action="append", type=Path,
                        default=[])
    parser.add_argument("--inference-provenance",
                        choices=("none", "historical", "current_snapshot"),
                        default="none")
    parser.add_argument("--hash-large-files", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.config:
        args.config = ["rmbench_train", "rmbench"]
    if args.inference_output and args.inference_provenance == "none":
        parser.error("outputs require an explicit inference provenance")
    manifest = capture(args)
    print(json.dumps({
        "output": str(args.output / "manifest.json"),
        "numeric_regression_ready": manifest["numeric_regression_ready"],
        "checkpoint_content_hashed": args.hash_large_files,
        "dataset_content_hashed": args.hash_large_files,
    }))


if __name__ == "__main__":
    main()
