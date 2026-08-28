"""
Paper vs replication table for Online-Mind2Web.

Only counts runs in the CORRECT configuration -- Browser-Use cloud browsers with
the viewport pinned to 1280x1000 DPR1, plus the Table 8 settings (30 steps,
o4-mini judge @3 imgs, judge timeout 120s, 16 parallel). Earlier local_process
and unpinned-viewport runs are deliberately excluded: they measured a different
environment, not a different model.

A run still in flight is reported as "running (n/300)" rather than as a score.
"""
import glob, json, os, sys

ROOT = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/eval"

# Paper Table 2, Online-Mind2Web official success rate.
# None = not recorded in the excerpts we have; do not invent it.
PAPER = {
    "Qwen3-VL-4B-Thinking": 32.0,
    "OpenWebRL-4B-SFT":     47.0,
    "OpenWebRL-4B":         67.0,
    "OpenWebRL-8B":         None,
}

# checkpoint -> directory holding its results, for the valid config only
DIRS = {
    "Qwen3-VL-4B-Thinking": f"{ROOT}/multi/om2w_vp1000_sweep/Qwen3-VL-4B-Thinking",
    "OpenWebRL-4B-SFT":     f"{ROOT}/multi/om2w_vp1000_sweep/OpenWebRL-4B-SFT",
    "OpenWebRL-8B":         f"{ROOT}/multi/om2w_vp1000_sweep/OpenWebRL-8B",
    "OpenWebRL-4B":         f"{ROOT}/OpenWebRL-4B/om2w_vp1000_noaborted",
}
OFFICIAL_4B = f"{ROOT}/OpenWebRL-4B/om2w_vp1000_official"
TOTAL = 300


def tally(d):
    if not os.path.isdir(d):
        return None
    rows = []
    for f in glob.glob(d + "/**/results_task_*.jsonl", recursive=True):
        try:
            rows.append(json.loads(open(f).read().strip().split("\n")[0]))
        except Exception:
            pass
    ab = len(glob.glob(d + "/**/abnormal_task_*.jsonl", recursive=True))
    if not rows and not ab:
        return None
    done = len(rows) + ab
    return {"done": done, "pass": sum(1 for r in rows if r.get("reward") == 1.0)}


def cell(t):
    if t is None:
        return "not started"
    if t["done"] < TOTAL:
        return f"running ({t['done']}/{TOTAL})"
    return f"{100*t['pass']/TOTAL:.1f}%"


rows = []
for name in ("Qwen3-VL-4B-Thinking", "OpenWebRL-4B-SFT", "OpenWebRL-4B", "OpenWebRL-8B"):
    t = tally(DIRS[name])
    paper = PAPER[name]
    ours = cell(t)
    if t and t["done"] >= TOTAL and paper is not None:
        delta = f"{100*t['pass']/TOTAL - paper:+.1f}"
    else:
        delta = "—"
    rows.append((name, "—" if paper is None else f"{paper:.1f}", ours, delta))

w = max(len(r[0]) for r in rows) + 2
print(f"{'Checkpoint':<{w}} {'Paper':>7} {'Ours':>18} {'Delta':>7}")
print("-" * (w + 36))
for n, p, o, d in rows:
    print(f"{n:<{w}} {p:>7} {o:>18} {d:>7}")

t = tally(OFFICIAL_4B)
print()
print("Same config but official/stochastic decoding (temp 0.6 / top-p 0.95 / top-k 20):")
print(f"  OpenWebRL-4B: {cell(t)}   (paper 67.0)")
