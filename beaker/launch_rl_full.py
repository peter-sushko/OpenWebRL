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
        *sandbox_env_args(args),
    ]
    if args.resume_from:
        command += ["--env", "SLIME_LOAD_CHECKPOINT=" + args.resume_from]
    command += [
        "--no-python",
        "--allow-dirty",
        "--", "bash", f"{OPENWEBRL_ROOT}/{args.script}",
    ]

    print("[launch]", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
