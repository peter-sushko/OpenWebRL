#!/usr/bin/env bash
# ==============================================================================
# In-container entry script: the FULL OpenWebRL-4B MM-GRPO reproduction run.
#
# Launched by beaker/launch_rl_full.py via `gantry run ... -- <this script>`.
# One 8-GPU node. Unlike beaker/run_rl_smoke.sh this does NOT downscale the
# launcher: --num-rollout 100, --rollout-batch-size 48, --n-samples-per-prompt 5,
# --global-batch-size 256, --ppo-epochs 2, --lr 5e-7, --max-steps 30, starting
# from OpenWebRL/OpenWebRL-4B-SFT. Those are the released launcher defaults, i.e.
# the paper's 4B configuration.
#
# Only two things are patched, both site-specific:
#   1. W&B entity/project -> ai2-llm/molmoweb (the launcher hardcodes the
#      authors' yangrui_rl/slime-dev, which our key cannot write to).
#   2. Judge -> `served` against the public OpenAI API (the launcher defaults to
#      Azure `token` mode). The judge MODEL stays gpt-4.1, as in the paper.
#
# The one genuine deviation from the paper is SLIME_BROWSER_ENV_MODE: the paper
# runs `sandbox` (Orchard K8s pods, per-episode network isolation) with a pool of
# 90; we have no cluster, so this runs `local_process` with the same 90-way
# concurrency. Throughput matches (the node ceiling is ~128) but network
# isolation does not, so expect a higher website block rate.
#
# Resumable: --save-interval 5 writes iter_* dirs under SLIME_SAVE_ROOT. To
# resume after preemption, re-launch with SLIME_LOAD_CHECKPOINT=<save-dir>.
# ==============================================================================
set -euo pipefail
set -x

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"

# ---- Paths ----
export SLIME_REPO_ROOT="${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:/root/Megatron-LM"
export SLIME_MODEL_ROOT="${SLIME_MODEL_ROOT:-/weka/oe-training-default/new_peters/models}"
export SLIME_SAVE_ROOT="${SLIME_SAVE_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL/outputs/rl_full}"
export SLIME_OUTPUT_ROOT="${SLIME_OUTPUT_ROOT:-${SLIME_SAVE_ROOT}/debug}"
mkdir -p "${SLIME_SAVE_ROOT}" "${SLIME_OUTPUT_ROOT}"

# Resume support: pass SLIME_LOAD_CHECKPOINT=<dir> to continue a preempted run.
export SLIME_LOAD_CHECKPOINT="${SLIME_LOAD_CHECKPOINT:-}"
# Pin the save dir across restarts so a resumed run keeps writing to one place.
export SLIME_SAVE_DIR="${SLIME_SAVE_DIR:-${SLIME_SAVE_ROOT}/openwebrl_4b_grpo_repro_s${RL_STAGE}}"
mkdir -p "${SLIME_SAVE_DIR}"

# ---- Paper's two-stage rollout-step schedule ----------------------------------
# The paper trains "90 iterations with a maximum of 15 rollout steps, followed by
# 50 iterations with a maximum of 30 rollout steps". The released launcher instead
# hardcodes a flat --num-rollout 100 at --max-steps 30, which matches neither
# stage, so we drive both from RL_STAGE and patch the two values into the
# throwaway launcher copy below.
#
# Stage 2 must resume from stage 1's last checkpoint: launch it with
#   --resume-from <SLIME_SAVE_ROOT>/openwebrl_4b_grpo_repro_s1
RL_STAGE="${RL_STAGE:-1}"
case "${RL_STAGE}" in
  1) NUM_ROLLOUT="${NUM_ROLLOUT:-90}"; BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-15}" ;;
  2) NUM_ROLLOUT="${NUM_ROLLOUT:-50}"; BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}" ;;
  *) echo "ERROR: RL_STAGE must be 1 or 2 (got '${RL_STAGE}')"; exit 6 ;;
esac
export BROWSER_MAX_STEPS

# ---- Model: the authors' released SFT warm start (launcher default) ----
export MODEL_NAME="${MODEL_NAME:-OpenWebRL/OpenWebRL-4B-SFT}"
export HF_HOME="${HF_HOME:-/weka/oe-training-default/new_peters/cache/hf}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

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
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-90}"
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${SLIME_SAVE_DIR}/env_server_logs"
mkdir -p "${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"

# ---- Judge: gpt-4.1 (paper) over the public OpenAI API ----
export JUDGE_API_MODE="served"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be injected as a gantry secret-env}"

# ---- Image packaging workaround: base image ships mismatched flashinfer
# (0.5.3) vs flashinfer-jit-cache (0.6.3). ----
export FLASHINFER_DISABLE_VERSION_CHECK=1

# ---- Memory. On H100 (80 GiB) this run OOM'd in the Megatron backward pass and
# needed mem-fraction-static lowered to 0.5. On B300 (268 GiB usable, confirmed by
# beaker/probe_blackwell.sh) there is 3.3x the headroom, so we leave the launcher's
# own 0.6 alone -- one fewer deviation. Override SGLANG_MEM_FRACTION_STATIC to go
# back to 0.5 if this is ever run on Hopper again.
#
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. The torch OOM
# message suggests it, but it is incompatible with TorchMemorySaver -- the very
# mechanism --colocate uses to release sglang memory during training. It makes
# every SGLangEngine.init() fail with:
#   "TorchMemorySaver is disabled for the current process because
#    expandable_segments is not supported yet."
# (Confirmed the hard way: it killed run v2 at engine init.)
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"

# ---- W&B ----
export WANDB_ENTITY="${WANDB_ENTITY:-ai2-llm}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-molmoweb}"

# ---- Fail fast rather than burn days on a run that cannot browse or judge ----
echo "Probing outbound web access..."
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
  || { echo "ERROR: no outbound web access; local_process rollouts cannot browse. Aborting."; exit 3; }
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "openai -> %{http_code}\n" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models \
  || { echo "ERROR: judge endpoint unreachable. Aborting."; exit 4; }

# ---- Chromium for Playwright ----
python -m playwright install chromium || playwright install chromium

# ---- Patch only the two site-specific lines, in a throwaway copy. Keep it in
# scripts/ so the launcher's REPO_ROOT=$(dirname)/.. math still resolves. ----
SRC="${OPENWEBRL_ROOT}/scripts/run_browser_Qwen3VL_4B_Instruct.sh"
RUN="${OPENWEBRL_ROOT}/scripts/run_browser_Qwen3VL_4B_repro.sh"
cp "${SRC}" "${RUN}"
sed -i -e "s|--wandb-project slime-dev|--wandb-project ${WANDB_PROJECT_NAME}|" \
       -e "s|--num-rollout 100|--num-rollout ${NUM_ROLLOUT}|" "${RUN}"

echo "=== repro launcher diff vs canonical (W&B project + num-rollout) ==="
diff "${SRC}" "${RUN}" || true
echo "=================================================================="
echo "save_dir=${SLIME_SAVE_DIR}  resume_from=${SLIME_LOAD_CHECKPOINT:-<none>}"
echo "browser concurrency=${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES}  judge=${JUDGE_MODEL}"
echo "stage=${RL_STAGE}  num_rollout=${NUM_ROLLOUT}  max_steps=${BROWSER_MAX_STEPS}"

bash "${RUN}"
