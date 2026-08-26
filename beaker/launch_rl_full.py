"""
Launch the full OpenWebRL-4B MM-GRPO reproduction run on Beaker via Gantry.

One 8-GPU node, ~3-4 days at the paper's 100 rollout steps. The slime/Megatron/
SGLang stack comes from an existing Beaker image; the OpenWebRL repo is read
from the Weka mount. Container entry point is beaker/run_rl_full.sh.

Resume after preemption:
  python beaker/launch_rl_full.py --resume-from <save-dir>

Prereqs:
  - `beaker account login`
  - Secrets in the workspace: PS_OPENAI_API_KEY, PS_HF_TOKEN, PS_WANDB_API_KEY
"""
import argparse
import os
import shutil
import subprocess

OPENWEBRL_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL"


def sandbox_env_args(args):
    """Extra gantry args for Orchard sandbox mode; empty for local_process."""
    if args.env_mode != "sandbox":
        return []
    if not args.sandbox_url or not args.sandbox_image:
        raise SystemExit("sandbox mode needs --sandbox-url and --sandbox-image "
                         "(or SANDBOX_ORCHESTRATOR_URL / BROWSER_SANDBOX_IMAGE in the environment)")
    return [
        "--env", "SLIME_BROWSER_ENV_MODE=sandbox",
        "--env", "SANDBOX_ORCHESTRATOR_URL=" + args.sandbox_url,
        "--env", "BROWSER_SANDBOX_IMAGE=" + args.sandbox_image,
        "--secret-env", "SANDBOX_API_KEY=" + args.sandbox_key_secret,
    ]


def snapshot_script(script_rel, name):
    """Copy the entry script to an immutable per-run path and return it.

    The container executes the script straight off Weka, and bash reads a script
    lazily by byte offset -- so editing beaker/run_rl_full.sh while a job is
    running it can make bash execute garbage at the next read. Launching a
    snapshot instead means the shared file is always safe to edit.
    """
    src = os.path.join(OPENWEBRL_ROOT, script_rel)
    snap_dir = os.path.join(OPENWEBRL_ROOT, "outputs", "rl_full", "_launched")
    os.makedirs(snap_dir, exist_ok=True)
    dst = os.path.join(snap_dir, f"{name}_{os.path.basename(script_rel)}")
    shutil.copy2(src, dst)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", "peters/openwebrl-train"))
    p.add_argument("--name", default="openwebrl_rl_repro_4b")
    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="Paper schedule: stage 1 = 90 iters @ max 15 rollout steps, "
                        "stage 2 = 50 iters @ max 30, resumed from stage 1.")
    p.add_argument("--script", default="beaker/run_rl_full.sh")
    p.add_argument("--cluster", default="ai2/holmes",
                   help="B300 (268 GiB/GPU). The paper used B200 (180 GiB); H100 (80 GiB) OOMs.")
    p.add_argument("--budget", default="ai2/oe-omai")
    p.add_argument("--workspace", default="ai2/general-tool-use")
    p.add_argument("--gpus", type=int, default=8)
    # Multi-day run: urgent on jupiter to limit preemption.
    p.add_argument("--priority", default="urgent")
    p.add_argument("--resume-from", default="",
                   help="Existing SLIME_SAVE_DIR to resume from after preemption.")
    p.add_argument("--env-mode", default="local_process", choices=("local_process", "sandbox"),
                   help="Browser env mode. 'sandbox' uses Orchard pods (the paper's setting).")
    p.add_argument("--sandbox-url", default=os.environ.get("SANDBOX_ORCHESTRATOR_URL", ""),
                   help="Orchard orchestrator base URL (sandbox mode).")
    p.add_argument("--sandbox-image", default=os.environ.get("BROWSER_SANDBOX_IMAGE", ""),
                   help="browser-env image in a registry the Orchard cluster can pull.")
    p.add_argument("--sandbox-key-secret", default="PS_SANDBOX_API_KEY",
                   help="Beaker secret holding the Orchard API key.")
    p.add_argument("--browser-concurrency", type=int, default=0,
                   help="SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES (default 90, the paper's pool "
                        "size). The paper gives each browser a dedicated CPU in its own pod; "
                        "local_process shares one node, so 90 thrashes and env_server startup "
                        "blows past its 30 s budget. Lower it when running local_process.")
    p.add_argument("--eval-interval", type=int, default=0,
                   help="Iterations between WebVoyager eval passes (launcher default 5). "
                        "~80 min per pass; raising it only costs monitoring resolution.")
    p.add_argument("--recompute", action="store_true",
                   help="Enable Megatron activation recompute (RL_RECOMPUTE=1). Mathematically "
                        "identical, ~30-40%% slower; use it to fit H100 80 GiB.")
    p.add_argument("--mem-fraction", default="",
                   help="SGLANG_MEM_FRACTION_STATIC override. 0.5 on H100, launcher default 0.6 on Blackwell.")
    p.add_argument("--save-dir", default="",
                   help="Override SLIME_SAVE_DIR. Required when queueing the same stage on two "
                        "clusters as mutual backups -- two live runs sharing one save dir "
                        "would interleave checkpoint writes.")
    p.add_argument("--ckpt-step", type=int, default=0,
                   help="Iteration to resume from inside --resume-from (e.g. 90 for stage 2).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    command = [
        "gantry", "run",
        "--name", args.name,
        "--budget", args.budget,
        "--workspace", args.workspace,
        "--priority", args.priority,
        "--cluster", args.cluster,
        "--gpus", str(args.gpus),
        "--shared-memory", "100GiB",
        "--weka", "oe-training-default:/weka/oe-training-default",
        "--beaker-image", args.image,
        "--secret-env", "OPENAI_API_KEY=PS_OPENAI_API_KEY",
        "--secret-env", "HUGGINGFACE_HUB_TOKEN=PS_HF_TOKEN",
        "--secret-env", "WANDB_API_KEY=PS_WANDB_API_KEY",
        "--env", "OPENWEBRL_ROOT=" + OPENWEBRL_ROOT,
        "--env", "RL_STAGE=" + str(args.stage),
        *(["--env", "SLIME_SAVE_DIR=" + args.save_dir] if args.save_dir else []),
        *(["--env", "RL_RECOMPUTE=1"] if args.recompute else []),
        *(["--env", "EVAL_INTERVAL=" + str(args.eval_interval)] if args.eval_interval else []),
        *(["--env", "SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=" + str(args.browser_concurrency)]
          if args.browser_concurrency else []),
        *(["--env", "SGLANG_MEM_FRACTION_STATIC=" + args.mem_fraction] if args.mem_fraction else []),
        *sandbox_env_args(args),
    ]
    if args.resume_from:
        command += ["--env", "SLIME_LOAD_CHECKPOINT=" + args.resume_from]
    # Megatron wants the checkpoint ROOT in --load and the iteration in
    # --ckpt-step (run_browser_...sh:50). The launcher guards --ckpt-step on a
    # non-empty SLIME_CKPT_STEP, so omitting it falls back to Megatron's
    # latest_checkpointed_iteration.txt -- fine, but be explicit for stage 2.
    if args.ckpt_step:
        command += ["--env", "SLIME_CKPT_STEP=" + str(args.ckpt_step)]
    command += [
        "--no-python",
        "--allow-dirty",
        "--", "bash", snapshot_script(args.script, args.name),
    ]

    print("[launch]", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
