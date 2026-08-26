"""
Re-score finished eval trajectories with the FULL screenshot history.

Why: in turn-level eval, generate_browser.py:1857 stores
metadata["full_image_list"] from the *K-truncated* context (img_list, built
from mm_messages_filtered at :1743-1766), not the whole episode. With the
paper's context_num_screenshots=1 that leaves exactly one screenshot, so the
WebVoyager/DeepShop judges -- which do full_image_list[-30:] -- score each task
from a single image. The trajectory-level path (:1595) uses all_imgs and is
unaffected. metadata["messages"] IS stored unfiltered, so the real history is
recoverable from results already on disk and we can re-judge without re-running
any rollout.

This changes ONLY the screenshots handed to the judge. Same judge model, same
prompts, same answer extraction, same status gating.

Usage:
  python beaker/rejudge_full_images.py <eval_leaf_dir> --protocol webvoyager [--limit N]
"""
import argparse
import asyncio
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slime.utils.types import Sample


def images_from_messages(messages):
    """Recover every screenshot in the episode, in order, from the unfiltered messages."""
    out = []
    for m in messages or []:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if "image" not in str(part.get("type", "")).lower():
                continue
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str) and url:
                out.append(url)
    return out


class Args:
    def __init__(self, protocol, n_imgs, model, hf_checkpoint):
        self.judge_max_attached_imgs = n_imgs
        self.judge_api_model = model
        self.judge_api_mode = os.environ.get("JUDGE_API_MODE", "served")
        self.judge_timeout_secs = 120.0
        self.hf_checkpoint = hf_checkpoint
        self.judge_prompt_variant = "action_history"


def load_records(d, limit=None):
    files = sorted(glob.glob(os.path.join(d, "results_task_*.jsonl")))
    if limit:
        files = files[:limit]
    for f in files:
        try:
            first = open(f).read().strip().split("\n")[0]
            yield f, json.loads(first)
        except Exception as e:
            print(f"skip {os.path.basename(f)}: {e}", file=sys.stderr)


def to_sample(rec, full_images):
    s = Sample(prompt=rec.get("prompt", ""))
    s.response = rec.get("response", "")
    md = dict(rec.get("metadata") or {})
    md["full_image_list"] = full_images
    md.setdefault("intent", rec.get("intent", ""))
    md.setdefault("task_id", rec.get("task_id", ""))
    s.metadata = md
    st = str(rec.get("status", ""))
    # Only COMPLETED episodes reach the judge (see _score_single); everything
    # else is scored 0.0 by the protocol regardless of screenshots.
    s.status = Sample.Status.COMPLETED if st.endswith("COMPLETED") else Sample.Status.FAILED
    return s


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir")
    p.add_argument("--protocol", default="webvoyager", choices=["webvoyager", "deepshop", "online_mind2web"])
    p.add_argument("--n-imgs", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--judge-model", default="")
    p.add_argument("--hf-checkpoint", default="OpenWebRL/OpenWebRL-4B")
    p.add_argument("--out", default="")
    a = p.parse_args()

    if a.protocol == "webvoyager":
        from openwebrl.eval.reward_webvoyager import _score_single
        default_model = "gpt-4o"
    elif a.protocol == "deepshop":
        from openwebrl.eval.reward_deepshop import _score_single
        default_model = "gpt-4o"
    else:
        from openwebrl.eval.reward_online_mind2web import _score_single
        default_model = "o4-mini"

    args = Args(a.protocol, a.n_imgs, a.judge_model or default_model, a.hf_checkpoint)

    recs = list(load_records(a.eval_dir, a.limit or None))
    print(f"loaded {len(recs)} records from {a.eval_dir}")

    async def one(f, rec):
        stored = rec.get("metadata", {}).get("full_image_list") or []
        full = images_from_messages(rec.get("metadata", {}).get("messages"))
        # Fall back to whatever was stored if we cannot recover a longer history.
        if len(full) < len(stored):
            full = stored
        s = to_sample(rec, full)
        try:
            new = await _score_single(args, s)
        except Exception as e:
            print(f"judge error {os.path.basename(f)}: {type(e).__name__} {e}", file=sys.stderr)
            new = None
        old = rec.get("reward")
        return {
            "task_id": rec.get("task_id"),
            "old": old,
            "new": new,
            "n_imgs_old": len(stored),
            "n_imgs_new": len(full),
            "status": str(rec.get("status")),
        }

    results = await asyncio.gather(*[one(f, r) for f, r in recs])

    def rate(key):
        vals = [r[key] for r in results if isinstance(r[key], (int, float))]
        return (sum(1 for v in vals if v == 1.0) / len(results)) if results else 0.0

    n_judged = sum(1 for r in results if r["new"] is not None)
    flipped_up = sum(1 for r in results if r["old"] != 1.0 and r["new"] == 1.0)
    flipped_dn = sum(1 for r in results if r["old"] == 1.0 and r["new"] != 1.0)
    mean_imgs = sum(r["n_imgs_new"] for r in results) / max(1, len(results))

    print("=" * 60)
    print(f"protocol            : {a.protocol}  judge={args.judge_api_model}  n_imgs<={a.n_imgs}")
    print(f"records             : {len(results)}   judged_ok: {n_judged}")
    print(f"screenshots/task    : was {sum(r['n_imgs_old'] for r in results)/max(1,len(results)):.2f} -> now {mean_imgs:.2f}")
    print(f"success rate OLD    : {rate('old'):.4f} ({rate('old')*100:.2f}%)")
    print(f"success rate NEW    : {rate('new'):.4f} ({rate('new')*100:.2f}%)")
    print(f"flipped 0->1        : {flipped_up}")
    print(f"flipped 1->0        : {flipped_dn}")
    print("=" * 60)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    asyncio.run(main())
