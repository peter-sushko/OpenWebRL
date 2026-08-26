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
# The learning rate is also per-stage. Paper Table 7 gives 1e-6 (constant
# schedule) for the 4B browser config; the released launcher instead has
# `--lr 5e-7  ###...###1e-6`. Its own commented DEFAULT_SAVE_DIR names explain
# the discrepancy -- the active one is
#   ..._maxstep20_..._trainstep30fromckpt59_lr5e-7
# i.e. the script as released IS the stage-2 continuation (resumed from
# checkpoint 59, 30 steps, lr lowered to 5e-7), while the commented
# ..._maxstep15_..._frombase variant is the stage-1 shape. So: 1e-6 for stage 1,
# 5e-7 for the stage-2 continuation.
case "${RL_STAGE}" in
  1) NUM_ROLLOUT="${NUM_ROLLOUT:-90}"; BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-15}"; RL_LR="${RL_LR:-1e-6}" ;;
  2) NUM_ROLLOUT="${NUM_ROLLOUT:-50}"; BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}"; RL_LR="${RL_LR:-5e-7}" ;;
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
# env_server logs go to NODE-LOCAL disk, not Weka. One fresh env_server (and
# Chromium) is spawned per task, each streaming stdout/stderr to its own file, so
# at concurrency 48 that is 48 concurrent writers plus a file create per task.
# Weka is a shared network FS sitting at 99% full here, and env_server startup
# latency climbed 60s -> 476s over ~300 task launches against a 600s task
# timeout. Local disk removes that from the startup path. Set
# SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR explicitly to override (e.g. back to Weka
# when you need the logs to survive the job).
# env/config.yaml sets startup_timeout_secs=30, but measured env_server startup
# is 50-500s once training rollouts contend with Megatron + 8 sglang engines for
# CPU. A 30s budget therefore rejects environments that are merely slow, which is
# what the "env_server at http..." failures are. Raising the budget discards no
# work and changes no training math -- it only stops throwing away usable envs.
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS:-300}"
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR:-/tmp/env_server_logs}"
mkdir -p "${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"

# ---- Judge: gpt-4.1 (paper) over the public OpenAI API ----
export JUDGE_API_MODE="served"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
set +x  # the `:` guard below is traced too, and would echo the key
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be injected as a gantry secret-env}"
set -x

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

# ---- Optional: activation recompute (RL_RECOMPUTE=1) --------------------------
# Lets the run fit on H100 (80 GiB), where three earlier attempts OOM'd in the
# Megatron backward pass -- the last one needed only 70 MiB more. Recompute is
# mathematically identical: same gradients, same update, activations are just
# recomputed in the backward instead of stored. It costs step time (~30-40%),
# nothing else, so an H100 arm here is a scheduling fallback, not a deviation in
# results. The launcher ships these three flags commented out at :471-473.
RL_RECOMPUTE="${RL_RECOMPUTE:-0}"

# ---- Eval cadence (pure observation, no effect on the trained model) ---------
# The launcher evals every 5 iterations on the full WebVoyager val set. Measured
# on jupiter: one eval pass = ~80 min (428 tasks judged in 61 min, ~7/min), so at
# interval 5 over 90 iterations that is ~18 passes = ~24 h of pure eval on top of
# training. The paper does not specify an eval interval, and eval never touches
# the weights, so raising this costs only monitoring resolution.
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

# ---- W&B ----
export WANDB_ENTITY="${WANDB_ENTITY:-ai2-llm}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-molmoweb}"

# ---- Fail fast rather than burn days on a run that cannot browse or judge ----
echo "Probing outbound web access..."
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
  || { echo "ERROR: no outbound web access; local_process rollouts cannot browse. Aborting."; exit 3; }
# xtrace off around the judge probe: with set -x the Bearer token lands in the
# Beaker logs in plaintext, readable by anyone with workspace access.
set +x
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "openai -> %{http_code}\n" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models \
  || { echo "ERROR: judge endpoint unreachable. Aborting."; set -x; exit 4; }
set -x

# ---- Chromium for Playwright ----
python -m playwright install chromium || playwright install chromium

# ---- Rollout task timeout ----------------------------------------------------
# Paper Table 7 sets "Rollout task timeout 600 s", but that assumes a Kubernetes
# sandbox whose browser is ready in ~1 s. In local_process the env_server is cold
# started per task and, once training rollouts contend for CPU, startup measured
# 420 s mean / 599 s max -- so a 600 s task budget leaves an episode ~100 s for 15
# steps and it is killed mid-way. Measured across three runs at concurrency 90 and
# 48, with logs on Weka and on local disk: same result every time.
#
# Raising the task timeout keeps the paper's EFFECTIVE episode budget (600 s of
# actual browsing) rather than spending most of it on a startup tax the paper does
# not have. It changes no optimization hyperparameter. Set ROLLOUT_TASK_TIMEOUT to
# 600 to go back to the literal paper value.
ROLLOUT_TASK_TIMEOUT="${ROLLOUT_TASK_TIMEOUT:-1800}"
CFG_SRC="${OPENWEBRL_ROOT}/openwebrl/browser_training_config.yaml"
CFG_RUN="${OPENWEBRL_ROOT}/openwebrl/browser_training_config_repro.yaml"
cp "${CFG_SRC}" "${CFG_RUN}"
sed -i -e "s|^rollout_task_timeout_secs:.*|rollout_task_timeout_secs: ${ROLLOUT_TASK_TIMEOUT}.0|" "${CFG_RUN}"
echo "=== config diff (rollout_task_timeout only) ==="
diff "${CFG_SRC}" "${CFG_RUN}" || true

# ---- Patch only the two site-specific lines, in a throwaway copy. Keep it in
# scripts/ so the launcher's REPO_ROOT=$(dirname)/.. math still resolves. ----
SRC="${OPENWEBRL_ROOT}/scripts/run_browser_Qwen3VL_4B_Instruct.sh"
RUN="${OPENWEBRL_ROOT}/scripts/run_browser_Qwen3VL_4B_repro.sh"
cp "${SRC}" "${RUN}"
sed -i -e "s|^set -ex$|set -e|" \
       -e '/--wandb-key "${WANDB_API_KEY}"/d' \
       -e "s|--wandb-project slime-dev|--wandb-project ${WANDB_PROJECT_NAME}|" \
       -e "s|--num-rollout 100|--num-rollout ${NUM_ROLLOUT}|" \
       -e "s|--lr 5e-7|--lr ${RL_LR}|" \
       -e "s|--eval-interval 5|--eval-interval ${EVAL_INTERVAL}|" \
       -e "s|--custom-config-path openwebrl/browser_training_config.yaml|--custom-config-path openwebrl/browser_training_config_repro.yaml|" "${RUN}"
if [ "${RL_RECOMPUTE}" = "1" ]; then
  sed -i -e "s|# --recompute-granularity full|--recompute-granularity full|" \
         -e "s|# --recompute-method uniform|--recompute-method uniform|" \
         -e "s|# --recompute-num-layers 1|--recompute-num-layers 1|" "${RUN}"
fi

echo "=== repro launcher diff vs canonical (xtrace off + W&B project + num-rollout + lr) ==="
diff "${SRC}" "${RUN}" || true
echo "=================================================================="
echo "save_dir=${SLIME_SAVE_DIR}  resume_from=${SLIME_LOAD_CHECKPOINT:-<none>}"
echo "browser concurrency=${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES}  judge=${JUDGE_MODEL}"
echo "stage=${RL_STAGE}  num_rollout=${NUM_ROLLOUT}  max_steps=${BROWSER_MAX_STEPS}  lr=${RL_LR}  recompute=${RL_RECOMPUTE}  eval_interval=${EVAL_INTERVAL}  task_timeout=${ROLLOUT_TASK_TIMEOUT}"

bash "${RUN}"
