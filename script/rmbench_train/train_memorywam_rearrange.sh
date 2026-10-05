#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: train_memorywam_rearrange.sh STEPS [OUTPUT]

The first run must use STEPS=2.  Runs longer than two optimizer steps require:
  MEMORYWAM_GPU_GATE=/path/to/gpu_smoke_pass.json
  MEMORYWAM_ALLOW_REFERENCE_TRAINING=1

The current implementation is the auditable full-episode reference path.
Measure its step latency and memory in the two-step smoke before authorizing a
long run.
EOF
}

STEPS=${1:-}
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }

REPO=${LINGBOT_VA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON_BIN:-python}
DATA_ROOT=${RMBENCH_DATA_ROOT:-/root/wyc/data/rmbench/lingbot_va2}
INITIALIZER=${MEMORYWAM_INITIALIZER:-${REPO}/train_out/rearrange_blocks_50demo_1000step_20260921_180241/checkpoints/checkpoint_step_750}
TRAIN=${MEMORYWAM_TRAIN_DATASET:-${DATA_ROOT}/rearrange_blocks_50demo_fullctx_apf16_cf8}
VALIDATION=${MEMORYWAM_VALIDATION_DATASET:-${DATA_ROOT}/rearrange_blocks_5demo_validation_fullctx_apf16_cf8}
TRAIN_ALIGNMENT=${MEMORYWAM_TRAIN_ALIGNMENT:-${REPO}/artifacts/memorywam/stage4_alignment/rearrange_train/episodes.jsonl}
VALIDATION_ALIGNMENT=${MEMORYWAM_VALIDATION_ALIGNMENT:-${REPO}/artifacts/memorywam/stage4_alignment/rearrange_validation/episodes.jsonl}
CPU_PREFLIGHT=${MEMORYWAM_CPU_PREFLIGHT:-${REPO}/artifacts/memorywam/stage6_training_ready/cpu_preflight.json}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT=${2:-${REPO}/train_out/rearrange_memorywam_reference_${STEPS}step_${STAMP}}
NGPU=${NGPU:-3}
FSDP_SHARD_SIZE=${RMBENCH_FSDP_SHARD_SIZE:-0}
FSDP_CPU_OFFLOAD=${RMBENCH_FSDP_CPU_OFFLOAD:-1}
FSDP_RESHARD=${RMBENCH_FSDP_RESHARD_AFTER_FORWARD:-1}
FSDP_ROOT_RESHARD=${RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD:-1}
MEMORYWAM_MODE=${RMBENCH_MEMORYWAM_MODE:-hybrid}
MASTER_PORT=${MASTER_PORT:-29681}
PYTHON_EXE=$(command -v "${PYTHON}")
PYTHON_PREFIX=$(cd "$(dirname "${PYTHON_EXE}")/.." && pwd)
CUDA_LIB=${LINGBOT_CUDA_LIB:-${PYTHON_PREFIX}/targets/x86_64-linux/lib:${PYTHON_PREFIX}/lib}

[[ ! -e "${OUTPUT}" ]] || { echo "Output already exists: ${OUTPUT}" >&2; exit 2; }
case "${MEMORYWAM_MODE}" in
  hybrid|recent-only|gist-only|full-history) ;;
  *) echo "Invalid RMBENCH_MEMORYWAM_MODE=${MEMORYWAM_MODE}" >&2; exit 2 ;;
esac
"${PYTHON}" - "${CPU_PREFLIGHT}" "${REPO}" <<'PY'
import json, sys
from pathlib import Path
from script.memorywam_checks.preflight_training import code_fingerprint
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS", report
actual = code_fingerprint(Path(sys.argv[2]))
assert report["code_fingerprint"] == actual, (
    "CPU preflight is stale; rerun run_training_cpu_preflight.sh",
    report["code_fingerprint"], actual)
PY
"${PYTHON}" - "${NGPU}" "${FSDP_SHARD_SIZE}" <<'PY'
import sys
from wan_va.distributed.fsdp import resolve_fsdp_topology

topology = resolve_fsdp_topology(int(sys.argv[1]), int(sys.argv[2]))
print(
    "FSDP topology preflight: "
    f"world={topology.world_size} shard={topology.shard_size} "
    f"replicate={topology.replicate_size} HSDP={topology.uses_hsdp}")
PY

if (( STEPS > 2 )); then
  [[ "${MEMORYWAM_ALLOW_REFERENCE_TRAINING:-0}" == 1 ]] || {
    echo "Long reference training requires MEMORYWAM_ALLOW_REFERENCE_TRAINING=1" >&2
    exit 2
  }
  : "${MEMORYWAM_GPU_GATE:?Set MEMORYWAM_GPU_GATE to a passed two-step report}"
  "${PYTHON}" - "${MEMORYWAM_GPU_GATE}" "${MEMORYWAM_MODE}" \
    "${NGPU}" "${FSDP_SHARD_SIZE}" "${FSDP_CPU_OFFLOAD}" \
    "${FSDP_RESHARD}" "${FSDP_ROOT_RESHARD}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS", report
assert report["optimizer_steps"] == 2, report
assert report["policy_mode"] == sys.argv[2], report
assert report["world_size"] == int(sys.argv[3]), report
expected_fsdp = {
    "shard_size": int(sys.argv[4]),
    "cpu_offload": sys.argv[5] == "1",
    "reshard_after_forward": sys.argv[6] == "1",
    "root_reshard_after_forward": sys.argv[7] == "1",
}
assert report["fsdp"] == expected_fsdp, (report, expected_fsdp)
PY
  EARLY_STOP=${RMBENCH_EARLY_STOP:-1}
  SAVE_INTERVAL=${RMBENCH_SAVE_INTERVAL:-50}
else
  EARLY_STOP=0
  SAVE_INTERVAL=${STEPS}
fi

command -v nvidia-smi >/dev/null
nvidia-smi -L >/dev/null
AVAILABLE=$(nvidia-smi -L | wc -l)
(( AVAILABLE >= NGPU )) || {
  echo "Requested NGPU=${NGPU}, but nvidia-smi reports ${AVAILABLE}" >&2
  exit 2
}

mkdir -p "${OUTPUT}"
cd "${REPO}"
"${PYTHON}" - "${OUTPUT}/run_config.json" "${CPU_PREFLIGHT}" "${INITIALIZER}" "${STEPS}" "${NGPU}" "${FSDP_SHARD_SIZE}" "${FSDP_CPU_OFFLOAD}" "${FSDP_RESHARD}" "${FSDP_ROOT_RESHARD}" "${MEMORYWAM_MODE}" <<'PY'
import hashlib, json, subprocess, sys
output, preflight, initializer, steps, ngpu, shard, offload, reshard, root_reshard, memory_mode = sys.argv[1:]
diff = subprocess.run(["git", "diff", "--binary"], check=True,
                      stdout=subprocess.PIPE).stdout
record = {
    "kind": "memorywam_reference_training",
    "task": "rearrange_blocks",
    "initializer": initializer,
    "cpu_preflight": preflight,
    "cpu_preflight_sha256": hashlib.sha256(open(preflight, "rb").read()).hexdigest(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "optimizer_steps": int(steps),
    "world_size": int(ngpu),
    "fsdp": {
        "shard_size": int(shard),
        "cpu_offload": offload == "1",
        "reshard_after_forward": reshard == "1",
        "root_reshard_after_forward": root_reshard == "1",
    },
    "policy": {"mode": memory_mode, "gist": 8, "anchor": 2,
               "recent": 4, "detach_interval": 0},
}
json.dump(record, open(output, "w"), indent=2)
PY

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-16}

RMBENCH_DATASET_PATH="${TRAIN}" \
RMBENCH_VALIDATION_DATASET_PATH="${VALIDATION}" \
RMBENCH_EARLY_STOP="${EARLY_STOP}" \
RMBENCH_SAVE_INTERVAL="${SAVE_INTERVAL}" \
RMBENCH_LOAD_WORKER="${RMBENCH_LOAD_WORKER:-4}" \
RMBENCH_FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE}" \
RMBENCH_FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD}" \
RMBENCH_FSDP_RESHARD_AFTER_FORWARD="${FSDP_RESHARD}" \
RMBENCH_FSDP_ROOT_RESHARD_AFTER_FORWARD="${FSDP_ROOT_RESHARD}" \
RMBENCH_MEMORYWAM_ENABLED=1 \
RMBENCH_MEMORY_ALIGNMENT_PATH="${TRAIN_ALIGNMENT}" \
RMBENCH_VALIDATION_MEMORY_ALIGNMENT_PATH="${VALIDATION_ALIGNMENT}" \
RMBENCH_MEMORYWAM_GIST_TOKENS=8 \
RMBENCH_MEMORYWAM_ANCHOR_FRAMES=2 \
RMBENCH_MEMORYWAM_RECENT_FRAMES=4 \
RMBENCH_MEMORYWAM_MODE="${MEMORYWAM_MODE}" \
RMBENCH_MEMORYWAM_DETACH_INTERVAL=0 \
LINGBOT_PRETRAINED_MODEL="${INITIALIZER}" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
PYTHONHASHSEED=42 \
LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT}" \
  -m wan_va.train --config-name rmbench_train \
  --save-root "${OUTPUT}" --num-steps "${STEPS}" \
  2>&1 | tee "${OUTPUT}/training.log"

"${PYTHON}" - "${OUTPUT}" "${STEPS}" "${MEMORYWAM_MODE}" <<'PY'
import json, math, sys
from pathlib import Path
from safetensors import safe_open

root, expected_steps, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
uses_gist = mode in ("hybrid", "gist-only")
run_config = json.loads((root / "run_config.json").read_text())
summary = json.loads((root / "training_summary.json").read_text())
assert summary["stop_step"] == expected_steps, summary
rows = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines()]
assert len(rows) == expected_steps, len(rows)
for row in rows:
    for key in ("latent_loss", "action_loss", "grad_norm",
                "optimizer_step_seconds",
                "global_latent_frames_per_second"):
        assert math.isfinite(row[key]), (key, row)
    assert row["optimizer_step_seconds"] > 0, row
    assert row["global_latent_frames"] > 0, row
    assert row["global_latent_frames_per_second"] > 0, row
    if uses_gist:
        assert math.isfinite(row["gist_grad_norm"]), row
        assert math.isfinite(row["gist_update_max"]), row
        assert row["gist_grad_norm"] > 0, row
    else:
        assert "gist_grad_norm" not in row, row
        assert "gist_update_max" not in row, row
if uses_gist:
    assert rows[-1]["gist_update_max"] > 0, rows
checkpoint = root / "checkpoints" / f"checkpoint_step_{expected_steps}" / "transformer"
config = json.loads((checkpoint / "config.json").read_text())
assert config["memorywam_enabled"] is True, config
assert config["memorywam_mode"] == mode, config
assert config["memorywam_gist_tokens"] == (8 if uses_gist else 0), config
with safe_open(checkpoint / "diffusion_pytorch_model.safetensors",
               framework="pt", device="cpu") as stream:
    if uses_gist:
        assert "gist_tokens" in stream.keys()
    else:
        assert "gist_tokens" not in stream.keys()
report = {
    "status": "PASS",
    "optimizer_steps": expected_steps,
    "policy_mode": mode,
    "world_size": run_config["world_size"],
    "fsdp": run_config["fsdp"],
    "checkpoint": str(checkpoint.parent),
    "last_metrics": rows[-1],
}
(root / "gpu_smoke_pass.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

echo "PASS: ${OUTPUT}/gpu_smoke_pass.json"
