# OpenWebRL-4B RL reproduction — run record

Everything needed to re-run or audit the two-stage MM-GRPO reproduction of
arXiv 2606.02031, plus the eval numbers it produced. Branch `tier3-rl-smoke-fixes`
of `peter-sushko/OpenWebRL`.

## What was run

| stage | iterations | max browser steps | lr | experiment | wandb | wall clock |
|---|---|---|---|---|---|---|
| 1 | 90 (ids 0–89) | 15 | 1e-6 | `01M14WEQMWYPY3X454AH919GZG` | [080cqfaz](https://wandb.ai/ai2-llm/molmoweb/runs/080cqfaz) | 2026-08-28 19:12 → 08-30 16:25 UTC (45.2 h) |
| 2 | 50 (ids 90–139) | 30 | 5e-7 | `01M1AATTTQ68RTC9JNCXYTPE0K` | [nttwgibl](https://wandb.ai/ai2-llm/molmoweb/runs/nttwgibl) | 2026-08-30 22:13 → 08-31 18:38 UTC (20.4 h) |

Both on one 8×B300 node, `ai2/holmes`, workspace `ai2/oe-agents-holmes`, image
`peters/openwebrl-train-sm103f`. 65.6 h total, ~525 GPU-hours, against the paper's
~300 B200 GPU-hours.

## Launch commands

```bash
# stage 1
python beaker/launch_rl_full.py --stage 1 \
  --image peters/openwebrl-train-sm103f --cluster ai2/holmes \
  --workspace ai2/oe-agents-holmes --browser-concurrency 48 --eval-interval 15

# stage 2 — resumes stage 1's last checkpoint
python beaker/launch_rl_full.py --stage 2 \
  --image peters/openwebrl-train-sm103f --cluster ai2/holmes \
  --workspace ai2/oe-agents-holmes --browser-concurrency 48 --eval-interval 10 \
  --resume-from /weka/oe-training-default/new_peters/OpenWebRL/outputs/rl_full/openwebrl_4b_grpo_repro_s1 \
  --ckpt-step 89
```

Start checkpoint: `models/OpenWebRL/OpenWebRL-4B-SFT` (their released SFT model).
Hyperparameters follow paper Table 7; `beaker/run_rl_full.sh` sets them per stage and
patches a throwaway copy of the released launcher, so the original script stays
untouched.

## Checkpoints

| stage | save dir | checkpoints | size |
|---|---|---|---|
| 1 | `outputs/rl_full/openwebrl_4b_grpo_repro_s1` | 18, every 5 iterations, last `iter_0000089` | 1.1 T |
| 2 | `outputs/rl_full/openwebrl_4b_grpo_repro_s2` | 10, last `iter_0000139` | 584 G |

slime writes torch-dist format; the eval harness needs HF. Convert with:

```bash
python tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir outputs/rl_full/openwebrl_4b_grpo_repro_s2/iter_0000139 \
  --output-dir /weka/oe-training-default/new_peters/models/OpenWebRL-4B-RL-s2-iter139 \
  --origin-hf-dir /weka/oe-training-default/new_peters/models/OpenWebRL/OpenWebRL-4B-SFT --force
```

Converted HF checkpoints (8.3 G each): `models/OpenWebRL-4B-RL-s1-iter89`,
`models/OpenWebRL-4B-RL-s2-iter139`.

## Speed

| stage | rollout | train | iteration | throughput |
|---|---|---|---|---|
| 1 | 1104 s mean | 438 s mean | ~1540 s | 107 TFLOPS/GPU |
| 2 | 876 s mean | 332 s mean | ~1210 s | — |

An iteration is 48 prompts × 5 samples = 240 kept trajectories (~2200 turn-level
samples, ~17 M tokens), and ~18–20 optimizer steps at global batch 256 with PPO
epochs 2.

Two things set this speed:

* **Token-budget packing.** The released launcher ships `--micro-batch-size 1` with
  dynamic batching commented out, which measured 34 TFLOPS/GPU (~2% of a B300) and
  1909 s per train phase, over half of it ranks idling. `RL_MAX_TOKENS_PER_GPU=32768`
  (now the default in `run_rl_full.sh`) enables `--use-dynamic-batch-size` and cut the
  train phase to ~530 s at 107 TFLOPS — 3.3× from one flag. Partitioning is
  Karmarkar–Karp over sequence lengths, not first-fit.
* **Browser cold start.** `local_process` spawns a Chromium + env_server per task.
  Browsing is only 69 s per episode (p50 55, p90 126) but startup measured 296 s
  median at concurrency 48, which is most of the rollout phase. Concurrency 48 with
  `ROLLOUT_TASK_TIMEOUT=1800` and a 300 s startup timeout was the first setting with
  every infra ratio at 0.000; the paper's 90 pushed startup to 476–600 s and episodes
  hit the task timeout.

## Training signal

`rollout/raw_reward_mean` went 0.35 → ~0.47 over stage 1 (first ten iterations
averaged 0.39, last ten 0.47) and sat at 0.46 through stage 2. `grad_norm` 1.1–2.6,
`kl` ~0.002 with the KL coefficient at 0.0, `aborted_ratio` and `remove_sample_ratio`
0.000 throughout.

In-run `eval/webvoyager`, 70-task subset: 0.165 → 0.310 → 0.429 → 0.322 → 0.533 →
0.402 → 0.499 across stage 1, ending 0.476 in stage 2. It swings ±10 points between
passes and shares `--max-steps` with training, so treat it as a trend line only — not
comparable to the offline numbers below.

## Eval results

All offline evals: 8 GPUs (sglang TP=2 × DP=4), temp 0.6 / top-p 0.95 / top-k 20 /
repetition penalty 1.0 / 4096 max response / 32768 context / 30 steps / 1 context
screenshot / 3 judge screenshots / 120 s judge timeout / bfloat16, judges per protocol
(gpt-4o for WebVoyager, o4-mini AgentTrek for Online-Mind2Web). This is paper Table
8's "official" preset, selected by `--score-mode official`. Success is over all tasks
in the denominator, the paper's convention.

### Browser-Use, viewport pinned 1280x1000

| checkpoint | WebVoyager | Online-Mind2Web |
|---|---|---|
| OpenWebRL-4B-SFT (start) | 58.0 | 44.3 |
| **ours, stage 2 iter139** | **67.6** | **56.4** |
| released OpenWebRL-4B | 66.0 | 61.7 |
| paper's reported | 74.1 | 67.0 |

RL added +9.6 WebVoyager and +12.1 om2w over the SFT start. Against their released
checkpoint we are +1.6 on WebVoyager and −5.3 on om2w. Their checkpoint scores 8.1 and
5.3 below the paper's own numbers on our harness while the SFT baseline lands within
3 — that asymmetry is unexplained.

Stage 1's checkpoint (iter89) scored 62.8 on WebVoyager w/o aborted on local browsers;
stage 2 measured 60.6 in the same configuration, i.e. stage 2 added nothing detectable
(inside the ±5 retry noise measured previously).

Example eval launch:

```bash
python beaker/launch_eval.py \
  --ckpt /weka/oe-training-default/new_peters/models/OpenWebRL-4B-RL-s2-iter139 \
  --bench webvoyager --image peters/openwebrl-train-sm103f \
  --cluster ai2/jupiter --workspace ai2/oe-agents \
  --env-mode browser-use --n-parallel 20 --score-mode official --run-tag wv_off_ours
```

### Vendor comparison, same checkpoints and decoding

| checkpoint | bench | Browser-Use 1280x1000 stealth | Browserbase 1280x720 stealth | Browserbase 1280x1000 no stealth |
|---|---|---|---|---|
| OpenWebRL-4B-SFT | om2w | 44.3 | — | 40.3 (39 aborts, unreliable) |
| ours iter139 | WebVoyager | 67.6 | 51.3 | 58.1 |
| ours iter139 | om2w | 56.4 | 52.2 | 60.5 |
| released 4B | WebVoyager | 66.0 | 53.4 | — |
| released 4B | om2w | 61.7 | 56.3 | 63.8 |

Most of Browserbase's apparent deficit was viewport, not vendor: its advanced stealth
renders 1280x720 regardless of what is requested, and `set_viewport_size()` changes
`window.innerHeight` but *not* the screenshot, so the geometry silently disagreed.
`--browserbase-no-stealth` is the only way to get 1280x1000, at the cost of anti-bot
fingerprinting (more site blocks, visible in the WebVoyager aborts).
`browserbase_env.py` now scales clicks to the screenshot's real pixel size.

## Gotchas worth keeping

1. **`--num-rollout` is an absolute stop index.** A resumed run starts at
   `loaded_rollout_id + 1` (`slime/backends/megatron_utils/actor.py:400`), so stage 2
   needs 140, not 50. Passing 50 makes `range(90, 50)` empty and the job exits
   *succeeded* in 7 minutes having trained nothing. Check `rollout_id` in the logs,
   not the exit code.
2. **`--override-opt_param-scheduler`** is required on resume, or Megatron asserts
   that the checkpoint's lr (1e-6) equals the command line's (5e-7) and dies in
   `MegatronTrainRayActor.init()`.
3. **B300 needs `peters/openwebrl-train-sm103f`.** The stock image's `sgl_kernel` has
   no sm_103 cubins and its three overlapping kernel packages disagree; that image
   builds sgl-kernel 0.3.19 from sglang's own `v0.5.6.post2` tag with `sm_103a`
   enabled, and drops the mismatched `flashinfer-jit-cache` 0.6.3 so flashinfer 0.5.3
   JITs its own modules. Also export `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` —
   Triton's bundled 12.8 ptxas rejects `sm_103`. That fix is needed in `run_eval.sh`
   too, not just in training.
4. **Allocation.** holmes schedules instantly with `--min-runtime 2h` in
   `ai2/oe-agents-holmes`; priority alone still lands on preemptible backfill. On
   jupiter use workspace `ai2/oe-agents` instead — `oe-agents-holmes` holds no
   allocation there and launches are rejected outright.
5. **One Browser-Use eval at a time.** The concurrent-session cap is per account, so
   overlapping jobs produce `429: Too many concurrent active sessions` and 60–80%
   aborts. Four parallel runs at `--n-parallel 16` were all invalid. A single job at
   20–25 is clean. Browserbase has no such limit; 50 workers is fine.
6. **Judge parsing.** `reward_webvoyager.py` returns 0.0 if `"NOT SUCCESS"` appears
   anywhere in the judge's reply, so a response reasoning "the first attempt was NOT
   SUCCESS … Final verdict: SUCCESS" scores 0. It errs toward failure and affects
   every WebVoyager number equally. Unquantified, because judge text is not saved in
   the result records. Upstream issue #5 is the mirror image of this in the native
   judge, which our eval path does not use.
