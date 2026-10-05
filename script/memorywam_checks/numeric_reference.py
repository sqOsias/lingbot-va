"""Capture and compare deterministic tensor outputs for the legacy gate.

This helper deliberately does not start LingBot-VA.  The existing inference path
must export the tensors produced by one fixed request to a ``torch.save`` file.
The helper then records an exact digest plus finite-value statistics, or compares
two exports with explicit tolerances.

Only tensor containers are accepted.  Loading uses ``weights_only=True`` so this
diagnostic never executes arbitrary pickled objects from an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(value[key], child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _walk(item, child)
        return
    raise TypeError(f"unsupported leaf at {prefix or '<root>'}: {type(value)!r}")


def _tensor_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def summarize(path: Path) -> dict[str, Any]:
    tensors = dict(_walk(_load(path)))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensors": {},
    }
    for name, tensor in tensors.items():
        finite = torch.isfinite(tensor) if tensor.is_floating_point() else None
        record: dict[str, Any] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _tensor_digest(tensor),
        }
        if finite is not None:
            record["all_finite"] = bool(finite.all())
            values = tensor[finite].to(torch.float64)
            if values.numel():
                record.update({
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                })
        summary["tensors"][name] = record
    return summary


def compare(reference: Path, candidate: Path, *, atol: float,
            rtol: float) -> dict[str, Any]:
    reference_tensors = dict(_walk(_load(reference)))
    candidate_tensors = dict(_walk(_load(candidate)))
    reference_names = set(reference_tensors)
    candidate_names = set(candidate_tensors)
    result: dict[str, Any] = {
        "schema_version": 1,
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "atol": atol,
        "rtol": rtol,
        "missing": sorted(reference_names - candidate_names),
        "unexpected": sorted(candidate_names - reference_names),
        "tensors": {},
    }
    passed = not result["missing"] and not result["unexpected"]
    for name in sorted(reference_names & candidate_names):
        expected = reference_tensors[name]
        actual = candidate_tensors[name]
        item: dict[str, Any] = {
            "reference_shape": list(expected.shape),
            "candidate_shape": list(actual.shape),
            "reference_dtype": str(expected.dtype),
            "candidate_dtype": str(actual.dtype),
        }
        compatible = expected.shape == actual.shape and expected.dtype == actual.dtype
        if compatible and expected.is_floating_point():
            difference = (expected.to(torch.float64) -
                          actual.to(torch.float64)).abs()
            item["max_abs_error"] = float(difference.max()) if difference.numel() else 0.0
            item["close"] = bool(torch.allclose(expected, actual, atol=atol,
                                                rtol=rtol, equal_nan=False))
        elif compatible:
            item["close"] = bool(torch.equal(expected, actual))
        else:
            item["close"] = False
        passed = passed and item["close"]
        result["tensors"][name] = item
    result["passed"] = passed
    return result


def _write_json(value: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--input", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--atol", type=float, default=1e-5)
    compare_parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == "capture":
        result = summarize(args.input)
    else:
        result = compare(args.reference, args.candidate,
                         atol=args.atol, rtol=args.rtol)
    _write_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "compare" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
