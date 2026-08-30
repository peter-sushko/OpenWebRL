#!/usr/bin/env bash
# In-container entry point for the full OpenWebRL-4B MM-GRPO run, launched by
# beaker/launch_rl_full.py on one 8-GPU node. Keeps the released launcher's
# hyperparameters; patches only W&B (ai2-llm/molmoweb) and the judge (served
# OpenAI instead of Azure). Resume a preempted run with SLIME_LOAD_CHECKPOINT.
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

# ---- Paper's two-stage rollout schedule: 90 iters @ 15 steps, then 50 @ 30.
# The released launcher is flat (100 @ 30) and its save-dir names show it is
# really the stage-2 continuation, hence the per-stage lr. Stage 2 must resume
# from stage 1: --resume-from <SLIME_SAVE_ROOT>/openwebrl_4b_grpo_repro_s1
RL_STAGE="${RL_STAGE:-1}"
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

# ---- Browser env mode. sandbox (the paper's setting) needs the Orchard client
# symlinked at ./sandbox/client plus the three SANDBOX_* vars. ----
export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"
if [ "${SLIME_BROWSER_ENV_MODE}" = "sandbox" ]; then
  : "${SANDBOX_ORCHESTRATOR_URL:?sandbox mode requires SANDBOX_ORCHESTRATOR_URL}"
  : "${BROWSER_SANDBOX_IMAGE:?sandbox mode requires BROWSER_SANDBOX_IMAGE (registry the cluster can pull)}"
  export SANDBOX_ORCHESTRATOR_URL SANDBOX_API_KEY BROWSER_SANDBOX_IMAGE
  [ -e "${OPENWEBRL_ROOT}/sandbox/client/sandbox_client.py" ] \
    || { echo "ERROR: Orchard client missing at sandbox/client/sandbox_client.py"; exit 5; }
  echo "sandbox mode: orchestrator=${SANDBOX_ORCHESTRATOR_URL} image=${BROWSER_SANDBOX_IMAGE}"
fi
if [ "${SLIME_BROWSER_ENV_MODE}" = "browser-use" ]; then
  # An iteration is 48 prompts x 5 samples = 240 episodes; without this gate all
  # 240 vendor sessions open at once.
  set +x
  : "${BROWSER_USE_API_KEY:?browser-use mode requires BROWSER_USE_API_KEY}"
  set -x
  export SLIME_BROWSER_ROLLOUT_CONCURRENCY="${SLIME_BROWSER_ROLLOUT_CONCURRENCY:-25}"
  echo "browser-use mode: rollout concurrency=${SLIME_BROWSER_ROLLOUT_CONCURRENCY}"
fi
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/root/.cache/ms-playwright}"
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-90}"
# One env_server per task, each streaming to its own log file. On Weka that
# pushed startup latency 60s -> 476s; node-local disk removes it from the
# startup path. Measured startup is 50-500s under CPU contention, so the
# config's 30s budget rejects environments that are merely slow.
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

# ---- Memory. The launcher's 0.6 fits Blackwell; drop to 0.5 on H100, which
# OOM'd in the Megatron backward pass.
#
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, however much the
# torch OOM message suggests it -- TorchMemorySaver (what --colocate uses to
# release sglang memory) refuses to initialise when it is on.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"

# ---- Optional activation recompute, for fitting on H100. Same gradients, same
# update, ~30-40% slower steps. Uncomments three launcher flags at :471-473.
RL_RECOMPUTE="${RL_RECOMPUTE:-0}"

# ---- Eval cadence. One pass is ~80 min, so the launcher's interval of 5 adds
# ~24 h over 90 iterations. Raising it costs only monitoring resolution.
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

# ---- W&B ----
export WANDB_ENTITY="${WANDB_ENTITY:-ai2-llm}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-molmoweb}"

# ---- Fail fast rather than burn days on a run that cannot browse or judge ----
echo "Probing outbound web access..."
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
  || { echo "ERROR: no outbound web access; rollouts cannot browse. Aborting."; exit 3; }
# xtrace off: set -x would put the Bearer token in the Beaker logs.
set +x
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "openai -> %{http_code}\n" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" https://api.openai.com/v1/models \
  || { echo "ERROR: judge endpoint unreachable. Aborting."; set -x; exit 4; }
set -x

# ---- Chromium for Playwright ----
python -m playwright install chromium || playwright install chromium

# ---- Token-capped micro-batching. Bounds the logits tensor (tokens x vocab
# 151936), which --recompute does not -- that is what OOM'd on H100. It is also
# the throughput knob: the launcher's static --micro-batch-size 1 measured 34
# TFLOPS/GPU because every rank idles until its longest single sample finishes.
# Memory scales linearly with this number; lower it if the train phase OOMs.
RL_MAX_TOKENS_PER_GPU="${RL_MAX_TOKENS_PER_GPU:-32768}"

# ---- Triton's bundled ptxas is 12.8 and rejects --gpu-name sm_103, so every
# JIT compile fails on B300. The system 12.9 ptxas accepts it. ----
if [ -x /usr/local/cuda/bin/ptxas ]; then
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
  echo "TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH} ($(/usr/local/cuda/bin/ptxas --version 2>/dev/null | tail -1))"
fi

# ---- sglang CUDA graphs. B300 freezes during capture (every new shape JITs
# once flashinfer's AOT cache is gone). 1 skips capture; SGLANG_CUDA_GRAPH_MAX_BS
# is the middle ground, keeping graphs but capping the captured shapes. ----
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-}"

# ---- sglang attention backend. Auto-select picks the Blackwell-only TRTLLM
# decode path, whose sgl_kernel signature does not match the downgraded sglang
# 0.5.6.post2 in this image. Force a backend that avoids that op; empty = auto.
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}"

# ---- Rollout task timeout. Table 7 says 600 s, which assumes a warm sandbox
# browser. local_process cold starts one per task (420 s mean under contention),
# leaving an episode ~100 s, so only that mode gets the inflated budget.
if [ "${SLIME_BROWSER_ENV_MODE}" = "local_process" ]; then
  ROLLOUT_TASK_TIMEOUT="${ROLLOUT_TASK_TIMEOUT:-1800}"
else
  ROLLOUT_TASK_TIMEOUT="${ROLLOUT_TASK_TIMEOUT:-600}"
fi
CFG_SRC="${OPENWEBRL_ROOT}/openwebrl/browser_training_config.yaml"
CFG_RUN="${OPENWEBRL_ROOT}/openwebrl/browser_training_config_repro.yaml"
cp "${CFG_SRC}" "${CFG_RUN}"
sed -i -e "s|^rollout_task_timeout_secs:.*|rollout_task_timeout_secs: ${ROLLOUT_TASK_TIMEOUT}.0|" "${CFG_RUN}"
if [ -n "${SLIME_BROWSER_ROLLOUT_CONCURRENCY:-}" ]; then
  # The config value wins over the env var (generate_browser.py:224), so it has to
  # be patched too, not just exported.
  sed -i -e "s|^browser_rollout_concurrency:.*|browser_rollout_concurrency: ${SLIME_BROWSER_ROLLOUT_CONCURRENCY}|" "${CFG_RUN}"
fi
echo "=== config diff (rollout_task_timeout only) ==="
diff "${CFG_SRC}" "${CFG_RUN}" || true

# ---- Patch the site-specific lines in a throwaway copy. It stays in scripts/
# so the launcher's REPO_ROOT=$(dirname)/.. math still resolves. ----
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
# ---- Stage 2 changes the LR mid-run (paper Table 7: 1e-6 then 5e-7) ----------
# Megatron's OptimizerParamScheduler asserts that the scheduler values in the
# checkpoint match the ones on the command line, so resuming stage 1 (1e-6) with
# --lr 5e-7 dies at MegatronTrainRayActor.init():
#   AssertionError: OptimizerParamScheduler: class input value 5e-07 and
#   checkpoint value 1e-06 for learning rate do not match
# --override-opt_param-scheduler tells it to use the command-line schedule and
# ignore the checkpoint's, which is exactly the intended stage-2 behaviour. The
# optimizer state itself (Adam moments) is still loaded.
if [ -n "${SLIME_LOAD_CHECKPOINT}" ]; then
  sed -i -e "s|   --lr-decay-style constant|   --lr-decay-style constant\n   --override-opt_param-scheduler|" "${RUN}"
  grep -n "override-opt_param-scheduler" "${RUN}" || { echo "ERROR: failed to add --override-opt_param-scheduler"; exit 7; }
fi

if [ -n "${RL_MAX_TOKENS_PER_GPU}" ]; then
  sed -i -e "s|# --use-dynamic-batch-size|--use-dynamic-batch-size|" \
         -e "s|# --max-tokens-per-gpu 8192|--max-tokens-per-gpu ${RL_MAX_TOKENS_PER_GPU}|" "${RUN}"
fi
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
echo "stage=${RL_STAGE}  num_rollout=${NUM_ROLLOUT}  max_steps=${BROWSER_MAX_STEPS}  lr=${RL_LR}  recompute=${RL_RECOMPUTE}  eval_interval=${EVAL_INTERVAL}  task_timeout=${ROLLOUT_TASK_TIMEOUT}  max_tokens_per_gpu=${RL_MAX_TOKENS_PER_GPU:-<static>}  sglang_attn=${SGLANG_ATTENTION_BACKEND:-<auto>}  cudagraph_off=${SGLANG_DISABLE_CUDA_GRAPH}  graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS:-<default>}"

if [ "${SGLANG_DISABLE_CUDA_GRAPH}" = "1" ]; then
  sed -i -e "s|   --sglang-log-level warning|   --sglang-disable-cuda-graph\n   --sglang-log-level warning|" "${RUN}"
  grep -n "sglang-disable-cuda-graph" "${RUN}" || { echo "ERROR: failed to inject --sglang-disable-cuda-graph"; exit 8; }
elif [ -n "${SGLANG_CUDA_GRAPH_MAX_BS}" ]; then
  sed -i -e "s|   --sglang-log-level warning|   --sglang-cuda-graph-max-bs ${SGLANG_CUDA_GRAPH_MAX_BS}\n   --sglang-log-level warning|" "${RUN}"
  grep -n "sglang-cuda-graph-max-bs" "${RUN}" || { echo "ERROR: failed to inject --sglang-cuda-graph-max-bs"; exit 8; }
fi

if [ -n "${SGLANG_ATTENTION_BACKEND}" ]; then
  sed -i -e "s|   --sglang-log-level warning|   --sglang-attention-backend ${SGLANG_ATTENTION_BACKEND}\n   --sglang-log-level warning|" "${RUN}"
  grep -n "sglang-attention-backend" "${RUN}" || { echo "ERROR: failed to inject sglang attention backend"; exit 7; }
fi

bash "${RUN}"
