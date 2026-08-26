#!/usr/bin/env bash
# ==============================================================================
# CPU-only offline re-judge of finished eval trajectories.
#
# Re-scores results_task_*.jsonl that are already on disk, changing ONLY the
# number of screenshots handed to the judge. No rollouts, no browsers, no GPUs.
# Runs a paired control (REJUDGE_CONTROL_IMGS) and treatment (REJUDGE_IMGS) over
# the same tasks so judge nondeterminism can be separated from the image count.
#
# Usage (in-container):
#   LEAF=<dir with results_task_*.jsonl> PROTOCOL=webvoyager bash beaker/run_rejudge.sh
# ==============================================================================
set -euo pipefail
set -x

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:/root/Megatron-LM"
export FLASHINFER_DISABLE_VERSION_CHECK=1

: "${LEAF:?set LEAF=<results dir>}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be injected as a gantry secret-env}"
PROTOCOL="${PROTOCOL:-webvoyager}"
LIMIT="${LIMIT:-200}"
REJUDGE_IMGS="${REJUDGE_IMGS:-30}"
REJUDGE_CONTROL_IMGS="${REJUDGE_CONTROL_IMGS:-3}"
OUT="${OUT:-${OPENWEBRL_ROOT}/outputs/rejudge}"
mkdir -p "${OUT}"

export JUDGE_API_MODE="served"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"

TAG="$(basename "$(dirname "${LEAF}")")_${PROTOCOL}"

for N in "${REJUDGE_CONTROL_IMGS}" "${REJUDGE_IMGS}"; do
  echo "########## re-judging ${PROTOCOL} with n_imgs=${N} (limit=${LIMIT}) ##########"
  python beaker/rejudge_full_images.py "${LEAF}" \
    --protocol "${PROTOCOL}" \
    --n-imgs "${N}" \
    --limit "${LIMIT}" \
    --out "${OUT}/${TAG}_n${N}.json"
done
