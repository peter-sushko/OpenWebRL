"""
Launch browser-benchmark evaluation on Beaker via Gantry.

One 8-GPU node per (checkpoint, benchmark) pair. The sglang/Megatron stack comes
from an existing Beaker image; the OpenWebRL repo is read from the Weka mount.
Container entry point is beaker/run_eval.sh.

Usage:
  # the authors' released post-RL checkpoint, all three benchmarks
  python beaker/launch_eval.py --ckpt OpenWebRL/OpenWebRL-4B --bench all

  # a local checkpoint, one benchmark
  python beaker/launch_eval.py --ckpt /weka/.../OpenWebRL-4B-SFT --bench om2w

Prereqs:
  - `beaker account login`
  - Secrets in the workspace: PS_OPENAI_API_KEY, PS_HF_TOKEN
"""
import argparse
import os
import re
import subprocess

OPENWEBRL_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL"
BENCHES = ("om2w", "webvoyager", "deepshop")


def slug(ckpt: str) -> str:
    """Beaker experiment names allow [a-zA-Z0-9_.-] only."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(ckpt.rstrip("/"))).lower()


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
    p.add_argument("--ckpt", required=True,
                   help="Local checkpoint dir or HuggingFace repo id.")
    p.add_argument("--bench", default="all", choices=BENCHES + ("all",))
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", "peters/openwebrl-train"))
    p.add_argument("--script", default="beaker/run_eval.sh")
    p.add_argument("--cluster", default="ai2/jupiter")
    p.add_argument("--budget", default="ai2/oe-omai")
    p.add_argument("--workspace", default="ai2/general-tool-use")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--priority", default="urgent")
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

    benches = BENCHES if args.bench == "all" else (args.bench,)

    for bench in benches:
        command = [
            "gantry", "run",
            "--name", f"owrl_eval_{slug(args.ckpt)}_{bench}",
            "--budget", args.budget,
            "--workspace", args.workspace,
            "--priority", args.priority,
            "--cluster", args.cluster,
            "--gpus", str(args.gpus),
            "--shared-memory", "100GiB",
            "--weka", "oe-training-default:/weka/oe-training-default",
            "--beaker-image", args.image,
            # Judge: o4-mini (om2w), gpt-4o (webvoyager/deepshop). HF for weights.
            "--secret-env", "OPENAI_API_KEY=PS_OPENAI_API_KEY",
            "--secret-env", "HUGGINGFACE_HUB_TOKEN=PS_HF_TOKEN",
            "--env", "OPENWEBRL_ROOT=" + OPENWEBRL_ROOT,
            "--env", "BENCH=" + bench,
            "--env", "CKPT=" + args.ckpt,
            *sandbox_env_args(args),
            "--no-python",
            "--allow-dirty",
            "--", "bash", f"{OPENWEBRL_ROOT}/{args.script}",
        ]
        print("[launch]", " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
