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
# CKPT may be a single path/repo-id, or a whitespace-separated LIST. A list is
# evaluated sequentially by run_evaluation.sh (one sglang server at a time,
# results split per model_tag), which is the only safe way to sweep checkpoints
# when the cloud-browser concurrency cap is account-wide rather than per-job.
CKPT="${CKPT:-/weka/oe-training-default/new_peters/models/OpenWebRL/OpenWebRL-4B-SFT}"
for _c in ${CKPT}; do
  if [ ! -d "${_c}" ] && [[ "${_c}" != */* ]]; then
    echo "ERROR: checkpoint is neither a local dir nor a HF repo id: ${_c}"; exit 2
  fi
done
export MODEL_LIST_OVERRIDE="${CKPT}"
export MODEL_LIST_DIR=""
# Slug used to keep each checkpoint's results in their own tree. For a list the
# per-model subdirectory does that job, so the tree is named by RUN_TAG instead.
if [ "$(echo ${CKPT} | wc -w)" -gt 1 ]; then
  CKPT_SLUG="multi"
else
  CKPT_SLUG="$(basename "${CKPT}")"
fi

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
# A subset run (TASK_INDICES set) must not land in the same tree as a full run,
# or compute_eval_success_rate.py will average the two together.
RUN_TAG="${RUN_TAG:-${BENCH}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${OPENWEBRL_ROOT}/outputs/eval/${CKPT_SLUG}/${RUN_TAG}}"
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
if [ "${SLIME_BROWSER_ENV_MODE}" = "browser-use" ]; then
  # Browser-Use cloud browsers -- the paper's own "Browser-Use Stealth Browsers".
  # The integration (openwebrl/env/browser_use_env.py) is the authors' code; we
  # only supply the key and hold concurrency under the account cap.
  : "${BROWSER_USE_API_KEY:?browser-use mode requires BROWSER_USE_API_KEY}"
  export BROWSER_USE_API_KEY
  python -c "import browser_use_sdk" 2>/dev/null \
    || pip install --no-cache-dir browser-use-sdk \
    || { echo "ERROR: could not install the browser-use SDK"; exit 7; }
  # Account cap is 25 concurrent browsers; --n-parallel is one session per task,
  # so keep it strictly below the cap to leave room for teardown lag.
  BU_MAX_CONCURRENCY="${BU_MAX_CONCURRENCY:-25}"
  if [ "${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES:-16}" -ge "${BU_MAX_CONCURRENCY}" ]; then
    echo "ERROR: n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} would meet or exceed the"
    echo "       Browser-Use concurrency cap of ${BU_MAX_CONCURRENCY}. Lower it."
    exit 8
  fi
  # Paper Table 8: browser viewport 1280x1000, DPR 1, coordinate scale 1000.
  # Browser-Use stealth otherwise randomises this per session (~1280x744..788 at
  # dpr 1..2), which puts the observation geometry out of the model's training
  # distribution. Pin it. Set to "" to keep the remote fingerprint.
  export BROWSER_USE_FORCE_VIEWPORT="${BROWSER_USE_FORCE_VIEWPORT-1280x1000}"
  echo "browser-use mode: n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} cap=${BU_MAX_CONCURRENCY} viewport=${BROWSER_USE_FORCE_VIEWPORT:-<remote default>}"
  # Stop anything a previous aborted run left billing.
  python -m openwebrl.env.browser_use_env --cleanup || true
fi
if [ "${SLIME_BROWSER_ENV_MODE}" = "browserbase" ]; then
  # Browserbase: remote stealth Chromium over CDP, residential proxies, in-band
  # CAPTCHA solving. This is the closest available stand-in for the paper's
  # "Browser-Use Stealth Browsers". Sessions bill per browser-minute AND per GB
  # of proxy egress, so keep n_parallel and the task set deliberate.
  : "${BROWSERBASE_API_KEY:?browserbase mode requires BROWSERBASE_API_KEY}"
  : "${BROWSERBASE_PROJECT_ID:?browserbase mode requires BROWSERBASE_PROJECT_ID}"
  export BROWSERBASE_API_KEY BROWSERBASE_PROJECT_ID
  export BROWSERBASE_PROXIES="${BROWSERBASE_PROXIES:-true}"
  export BROWSERBASE_ADVANCED_STEALTH="${BROWSERBASE_ADVANCED_STEALTH:-true}"
  export BROWSERBASE_SOLVE_CAPTCHAS="${BROWSERBASE_SOLVE_CAPTCHAS:-true}"
  # Let the step budget end the episode, not the project's default session cap.
  export BROWSERBASE_SESSION_TIMEOUT_S="${BROWSERBASE_SESSION_TIMEOUT_S:-1800}"
  python -c "import browserbase" 2>/dev/null \
    || pip install --no-cache-dir "browserbase==1.4.0" \
    || { echo "ERROR: could not install the browserbase SDK"; exit 6; }
  echo "browserbase mode: proxies=${BROWSERBASE_PROXIES} stealth=${BROWSERBASE_ADVANCED_STEALTH}"
  # Release anything a previous aborted run left live before opening new sessions.
  python -m openwebrl.env.browserbase_env --cleanup || true
fi
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

# Paper Table 8 defines TWO decoding configs. EVAL_SCORE_MODE selects one:
#   official  -> temp 0.6 / top_p 0.95 / top_k 20   (the Table 2 headline number)
#   noaborted -> temp 0.0 / top_p 1.0  / top_k 1    ("success rate w/o aborted tasks")
# Explicit EVAL_TEMPERATURE/TOP_P/TOP_K still override.
EVAL_SCORE_MODE="${EVAL_SCORE_MODE:-noaborted}"
case "${EVAL_SCORE_MODE}" in
  official)  _T=0.6; _P=0.95; _K=20 ;;
  noaborted) _T=0.0; _P=1.0;  _K=1  ;;
  *) echo "ERROR: EVAL_SCORE_MODE must be official|noaborted, got '${EVAL_SCORE_MODE}'"; exit 9 ;;
esac
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-${_T}}"
export EVAL_TOP_P="${EVAL_TOP_P:-${_P}}"
export EVAL_TOP_K="${EVAL_TOP_K:-${_K}}"
export JUDGE_TIMEOUT_SECS="${JUDGE_TIMEOUT_SECS:-120}"   # paper Table 8
echo "=== eval config ==="
echo "BENCH=${BENCH} PROTOCOL=${EVAL_PROTOCOL} TASK_FILE=${TASK_FILE} TASKS=$(wc -l < "${TASK_FILE}")"
echo "CKPT=${CKPT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} env_mode=${SLIME_BROWSER_ENV_MODE}"
echo "task_indices=${TASK_INDICES:-<all>}"
echo "score_mode=${EVAL_SCORE_MODE} temp=${EVAL_TEMPERATURE} top_p=${EVAL_TOP_P} top_k=${EVAL_TOP_K} judge_timeout=${JUDGE_TIMEOUT_SECS}"
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
