#!/usr/bin/env bash
# ==============================================================================
# In-container entry script: evaluate one checkpoint on one browser benchmark.
#
# Launched by beaker/launch_eval.py via `gantry run ... -- <this script>`.
# One 8-GPU node per benchmark. The OpenWebRL repo is read from the Weka mount;
# the sglang/Megatron stack comes from the Beaker image.
#
# Everything the paper's released harness controls is left at its default so the
# protocol matches: run_evaluation.sh already defaults to --max-steps 30,
# --context-num-screenshots 1, --turn-level, temperature 0.0, TP2/DP4, and the
# per-protocol canonical judge (online_mind2web=o4-mini, webvoyager=gpt-4o w/ 30
# imgs) whenever JUDGE_MODEL is left empty. We override only what is
# site-specific: the checkpoint, the judge endpoint, and the browser env mode.
#
# The one unavoidable deviation from the paper is SLIME_BROWSER_ENV_MODE:
# the paper runs `sandbox` (Orchard K8s pods, per-episode network isolation);
# we have no cluster yet, so this runs `local_process`. The repo's own figure
# for that gap is a website block rate of 25.7% (local) vs 17.7% (sandbox) on
# Online-Mind2Web, so treat these numbers as a floor.
#
# Usage (in-container): CKPT=<path|hf-repo> BENCH=om2w|webvoyager|deepshop bash beaker/run_eval.sh
# ==============================================================================
set -euo pipefail
set -x

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"

BENCH="${BENCH:?set BENCH=om2w|webvoyager|deepshop}"

# ---- Paths ----
export SLIME_REPO_ROOT="${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:/root/Megatron-LM"
export HF_HOME="${HF_HOME:-/weka/oe-training-default/new_peters/cache/hf}"

# ---- Checkpoint under evaluation. Either a local path or a HuggingFace repo
# id (sglang and run_evaluate.py both accept a repo id; HF_HOME caches to Weka). ----
CKPT="${CKPT:-/weka/oe-training-default/new_peters/models/OpenWebRL/OpenWebRL-4B-SFT}"
if [ ! -d "${CKPT}" ] && [[ "${CKPT}" != */* ]]; then
  echo "ERROR: checkpoint is neither a local dir nor a HF repo id: ${CKPT}"; exit 2
fi
export MODEL_LIST_OVERRIDE="${CKPT}"
export MODEL_LIST_DIR=""
# Slug used to keep each checkpoint's results in their own tree.
CKPT_SLUG="$(basename "${CKPT}")"

# ---- Benchmark selection. JUDGE_MAX_ATTACHED_IMGS is pinned per benchmark
# because scripts/run_evaluation.sh defaults it to 3 and ALWAYS passes
# --judge-max-attached-imgs, which defeats run_evaluate.py's documented
# per-protocol default of 30 for webvoyager/deepshop. Setting it here restores
# the canonical protocol described at run_evaluate.py:435 and :493. ----
case "${BENCH}" in
  om2w)
    export TASK_FILE="openwebrl/data/eval/online-mind2web.jsonl"
    export EVAL_PROTOCOL="online_mind2web"
    export JUDGE_MAX_ATTACHED_IMGS=3
    ;;
  webvoyager)
    export TASK_FILE="openwebrl/data/eval/webvoyager_fara.jsonl"
    export EVAL_PROTOCOL="webvoyager"
    export JUDGE_MAX_ATTACHED_IMGS=3
    ;;
  deepshop)
    export TASK_FILE="openwebrl/data/eval/deepshop.jsonl"
    export EVAL_PROTOCOL="deepshop"
    export JUDGE_MAX_ATTACHED_IMGS=3
    ;;
  *)
    echo "ERROR: BENCH must be om2w|webvoyager|deepshop, got '${BENCH}'"; exit 2 ;;
esac
export OUTPUT_ROOT="${OUTPUT_ROOT:-${OPENWEBRL_ROOT}/outputs/eval/${CKPT_SLUG}/${BENCH}}"
mkdir -p "${OUTPUT_ROOT}"

# ---- Judge: canonical per-protocol model, reached through the public OpenAI
# API. JUDGE_MODEL stays EMPTY on purpose so run_evaluate.py applies the
# protocol default (o4-mini / gpt-4o). `served` mode is the only one of the
# three that talks to non-Azure OpenAI; it appends /v1 itself and falls back to
# OPENAI_API_KEY for auth. ----
export JUDGE_API_MODE="served"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-}"   # empty => protocol default (o4-mini om2w / gpt-4o wv)
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be injected as a gantry secret-env}"

# ---- Browser env mode. Defaults to local_process; set SLIME_BROWSER_ENV_MODE=sandbox
# plus SANDBOX_ORCHESTRATOR_URL / SANDBOX_API_KEY / BROWSER_SANDBOX_IMAGE to use
# Orchard pods (the paper's setting). Sandbox mode additionally needs the Orchard
# client on the path at ./sandbox/client -- already symlinked in this checkout. ----
export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"
if [ "${SLIME_BROWSER_ENV_MODE}" = "sandbox" ]; then
  : "${SANDBOX_ORCHESTRATOR_URL:?sandbox mode requires SANDBOX_ORCHESTRATOR_URL}"
  : "${BROWSER_SANDBOX_IMAGE:?sandbox mode requires BROWSER_SANDBOX_IMAGE (registry the cluster can pull)}"
  export SANDBOX_ORCHESTRATOR_URL SANDBOX_API_KEY BROWSER_SANDBOX_IMAGE
  [ -e "${OPENWEBRL_ROOT}/sandbox/client/sandbox_client.py" ] \
    || { echo "ERROR: Orchard client missing at sandbox/client/sandbox_client.py"; exit 5; }
  echo "sandbox mode: orchestrator=${SANDBOX_ORCHESTRATOR_URL} image=${BROWSER_SANDBOX_IMAGE}"
fi
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/root/.cache/ms-playwright}"
# --n-parallel comes from SLIME_BROWSER_SANDBOX_MAX_SANDBOXES (released default
# 20); keep it there so per-site request pressure matches the paper's runs, and
# give the local process pool headroom above it.
export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES:-16}"   # paper Table 8
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-32}"
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${OUTPUT_ROOT}/env_server_logs"
mkdir -p "${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"

# ---- Image packaging workaround: base image ships mismatched flashinfer
# (0.5.3) vs flashinfer-jit-cache (0.6.3). Bypass the strict version check so
# the sglang server starts. ----
export FLASHINFER_DISABLE_VERSION_CHECK=1

# ---- Connectivity probes: fail fast rather than scoring a run that could not
# browse or could not reach the judge. ----
echo "Probing outbound web access..."
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
  || { echo "ERROR: no outbound web access; local_process rollouts cannot browse. Aborting."; exit 3; }
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "openai -> %{http_code}\n" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models \
  || { echo "ERROR: judge endpoint unreachable. Aborting."; exit 4; }

# ---- Chromium for Playwright (python package is in the image's requirements) ----
python -m playwright install chromium || playwright install chromium

export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
export EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
export EVAL_TOP_K="${EVAL_TOP_K:--1}"
echo "=== eval config ==="
echo "BENCH=${BENCH} PROTOCOL=${EVAL_PROTOCOL} TASK_FILE=${TASK_FILE} TASKS=$(wc -l < "${TASK_FILE}")"
echo "CKPT=${CKPT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} env_mode=${SLIME_BROWSER_ENV_MODE}"
echo "==================="

bash "${OPENWEBRL_ROOT}/scripts/run_evaluation.sh"

# ---- Score. run_evaluate.py nests results at
# OUTPUT_ROOT/<model_tag>/eval_<ckpt>_<judge>_turn_<timestamp>/, and
# compute_eval_success_rate.py globs results_task_*.jsonl non-recursively, so
# point it at each leaf that actually holds results. ----
echo "=== success rate ==="
find "${OUTPUT_ROOT}" -name 'results_task_*.jsonl' -printf '%h\n' | sort -u | while read -r leaf; do
  python "${OPENWEBRL_ROOT}/openwebrl/compute_eval_success_rate.py" "${leaf}" || true
done
