#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: run_full_model_cpu_sequence.sh [OUTPUT_ROOT]

Runs the full-model CPU checks in this exact order:
  plan -> load -> forward -> train-gist (2 optimizer steps by default)

Environment:
  MEMORYWAM_CPU_GIST_STEPS  Positive number of train-gist steps (default: 2)

The script stops at the first failing stage. It never runs train-all.
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

REPO=${LINGBOT_VA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
WRAPPER=${REPO}/script/memorywam_checks/run_full_model_cpu_smoke.sh
STEPS=${MEMORYWAM_CPU_GIST_STEPS:-2}
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${1:-${REPO}/artifacts/memorywam/stage8_full_cpu_sequence_${STAMP}}

if [[ ! ${STEPS} =~ ^[1-9][0-9]*$ ]]; then
  echo "MEMORYWAM_CPU_GIST_STEPS must be a positive integer, got: ${STEPS}" >&2
  exit 2
fi

case "${OUTPUT_ROOT}" in
  /*) ;;
  *) OUTPUT_ROOT=${REPO}/${OUTPUT_ROOT} ;;
esac

if [[ -e ${OUTPUT_ROOT} ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
LOG=${OUTPUT_ROOT}/sequence.log
STATUS=${OUTPUT_ROOT}/sequence_status.txt
CURRENT_STAGE=initializing

write_status() {
  local result=$1
  local exit_code=$2
  {
    printf 'status=%s\n' "${result}"
    printf 'last_stage=%s\n' "${CURRENT_STAGE}"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'gist_steps=%s\n' "${STEPS}"
    printf 'output_root=%s\n' "${OUTPUT_ROOT}"
    printf 'finished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${STATUS}.tmp"
  mv "${STATUS}.tmp" "${STATUS}"
}

on_exit() {
  local exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    write_status FAIL "${exit_code}"
    echo "FULL MODEL CPU SEQUENCE FAILED at ${CURRENT_STAGE} (exit ${exit_code})"
    echo "Inspect: ${LOG}"
    echo "Stage artifacts: ${OUTPUT_ROOT}/${CURRENT_STAGE}"
  fi
}
trap on_exit EXIT

exec > >(tee -a "${LOG}") 2>&1
cd "${REPO}"

run_stage() {
  local stage=$1
  shift
  CURRENT_STAGE=${stage}
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${stage}"
  bash "${WRAPPER}" "${stage}" "${OUTPUT_ROOT}/${stage}" "$@"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PASS  ${stage}"
}

run_stage plan
run_stage load
run_stage forward
run_stage train-gist -- --steps "${STEPS}"

CURRENT_STAGE=complete
write_status PASS 0
trap - EXIT

echo
echo "FULL MODEL CPU SEQUENCE PASS"
echo "Status: ${STATUS}"
echo "Log:    ${LOG}"
echo "Output: ${OUTPUT_ROOT}"
