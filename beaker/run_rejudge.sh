#!/usr/bin/env bash
# CPU-only: re-judge finished eval trajectories with the full screenshot history.
set -euo pipefail
set -x
OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:${PYTHONPATH:-}"
export JUDGE_API_MODE=served
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"
: "${OPENAI_API_KEY:?need OPENAI_API_KEY}"
python3 beaker/rejudge_full_images.py "${EVAL_DIR}" \
  --protocol "${PROTOCOL}" --limit "${LIMIT:-0}" \
  --hf-checkpoint "${HF_CKPT:-OpenWebRL/OpenWebRL-4B}" \
  --out "${OUT_JSON:-${OPENWEBRL_ROOT}/outputs/rejudge_${PROTOCOL}.json}"
