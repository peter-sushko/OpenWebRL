#!/usr/bin/env bash
# ==============================================================================
# Local-process browser concurrency sweep, one 8-GPU node.
#
# Question: how many concurrent local Chromium envs can this node actually
# carry? 64 was never measured — it was picked to saturate an 80-episode smoke
# batch (upstream default is 16). This runs ONE paper-scale rollout
# (48 prompts x 5 samples = 240 episodes) at each cap and compares
# rollout/failed_ratio and perf/rollout_time.
#
# Launched by beaker/launch_rl_smoke.py --script beaker/run_env_concurrency_sweep.sh
# ==============================================================================
set -uo pipefail   # NOT -e: a failing arm must not skip the remaining arms
set -x

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"

CAPS="${CAPS:-64 128 240}"
SWEEP_ROOT="${SWEEP_ROOT:-${OPENWEBRL_ROOT}/outputs/env_concurrency_sweep}"
mkdir -p "${SWEEP_ROOT}"

# One paper-scale rollout per arm: 48 x 5 = 240 episodes, so the 64-cap arm
# queues ~4 waves and the 240-cap arm runs everything at once.
export SMOKE_NUM_ROLLOUT="${SMOKE_NUM_ROLLOUT:-1}"
export SMOKE_ROLLOUT_BATCH_SIZE="${SMOKE_ROLLOUT_BATCH_SIZE:-48}"
export SMOKE_N_SAMPLES="${SMOKE_N_SAMPLES:-5}"
export SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-8}"
export BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-8}"

for CAP in ${CAPS}; do
  ARM_ROOT="${SWEEP_ROOT}/cap_${CAP}"
  mkdir -p "${ARM_ROOT}"
  echo "===== ARM cap=${CAP} start $(date -u +%FT%TZ) ====="

  export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${CAP}"
  export SLIME_SAVE_ROOT="${ARM_ROOT}"
  export SLIME_OUTPUT_ROOT="${ARM_ROOT}/debug"

  # Sample host resources every 30s: free RAM, load, and live browser count.
  ( while true; do
      printf '%s mem_used_gb=%s load=%s chrome_procs=%s env_servers=%s\n' \
        "$(date -u +%FT%TZ)" \
        "$(free -g | awk 'NR==2{print $3}')" \
        "$(awk '{print $1}' /proc/loadavg)" \
        "$(pgrep -c -f 'chrome|chromium' || echo 0)" \
        "$(pgrep -c -f 'openwebrl.docker.env_server' || echo 0)"
      sleep 30
    done ) > "${ARM_ROOT}/resources.log" 2>&1 &
  SAMPLER_PID=$!

  bash "${OPENWEBRL_ROOT}/beaker/run_rl_smoke.sh" 2>&1 | tee "${SWEEP_ROOT}/cap_${CAP}.log"
  ARM_RC=${PIPESTATUS[0]}
  echo "===== ARM cap=${CAP} end rc=${ARM_RC} $(date -u +%FT%TZ) ====="

  kill "${SAMPLER_PID}" 2>/dev/null || true

  # Full teardown between arms: Ray/redis, then any leaked env servers/browsers.
  bash "${OPENWEBRL_ROOT}/scripts/clean_processes.sh" || true
  pkill -9 -f 'openwebrl.docker.env_server' || true
  pkill -9 -f 'chrome|chromium' || true
  sleep 20
done

# ---- Summary ----
echo "================ SWEEP SUMMARY ================"
for CAP in ${CAPS}; do
  LOG="${SWEEP_ROOT}/cap_${CAP}.log"
  [ -f "${LOG}" ] || continue
  echo "--- cap=${CAP} ---"
  grep -a -oE "'(rollout/(failed_ratio|raw_reward_mean|aborted_or_timeout_ratio|effective_batch_ratio)|perf/(rollout_time|wait_time_ratio))': [0-9.]+" "${LOG}" | sort -u
  echo "peak mem_used_gb: $(awk -F'mem_used_gb=' '{split($2,a," ");if(a[1]+0>m)m=a[1]+0}END{print m}' "${SWEEP_ROOT}/cap_${CAP}/resources.log" 2>/dev/null)"
  echo "peak chrome_procs: $(awk -F'chrome_procs=' '{split($2,a," ");if(a[1]+0>m)m=a[1]+0}END{print m}' "${SWEEP_ROOT}/cap_${CAP}/resources.log" 2>/dev/null)"
done
echo "==============================================="
