"""Compare our eval runs against the OpenWebRL paper's Table 2 numbers."""
import json, glob, os, sys

# Paper Table 2 (official success rate: Browser-Use Stealth + stochastic decoding)
PAPER = {
    ("Qwen3-VL-4B-Thinking", "webvoyager"): 52.6,
    ("Qwen3-VL-4B-Thinking", "om2w"):       32.0,
    ("OpenWebRL-4B-SFT",     "webvoyager"): 60.2,
    ("OpenWebRL-4B-SFT",     "om2w"):       47.0,
    ("OpenWebRL-4B",         "webvoyager"): 74.1,
    ("OpenWebRL-4B",         "om2w"):       67.0,
}
NTASK = {"om2w": 300, "webvoyager": 595, "deepshop": 150}
base = sys.argv[1] if len(sys.argv) > 1 else "outputs/eval"

rows = []
for ck in sorted(os.listdir(base)) if os.path.isdir(base) else []:
    for bench in sorted(os.listdir(f"{base}/{ck}")):
        best = None
        for leaf in glob.glob(f"{base}/{ck}/{bench}/*/eval_*"):
            n = succ = ab = 0; imgs = []
            for f in glob.glob(leaf + "/results_task_*.jsonl"):
                try: r = json.loads(open(f).read().strip().split("\n")[0])
                except Exception: continue
                n += 1
                if str(r.get("status", "")).endswith("ABORTED"): ab += 1
                if r.get("reward") == 1.0: succ += 1
                imgs.append(len(r.get("metadata", {}).get("full_image_list") or []))
            if n and (best is None or n > best[0]):
                best = (n, ab, succ, sum(imgs)/len(imgs))
        if best:
            n, ab, succ, mi = best
            rows.append((ck, bench, n, ab, succ, 100*succ/n, 100*succ/(n-ab) if n > ab else 0, mi))

hdr = f"{'checkpoint':<24}{'bench':<11}{'done':>9}{'abort':>7}{'incl_ab':>9}{'w/o_ab':>8}{'paper':>7}{'Δ':>7}{'imgs':>6}"
print(hdr); print("-"*len(hdr))
for ck, bench, n, ab, succ, incl, wo, mi in rows:
    exp = NTASK.get(bench, 0)
    p = PAPER.get((ck, bench))
    d = f"{wo-p:+.1f}" if p else "-"
    print(f"{ck:<24}{bench:<11}{f'{n}/{exp}':>9}{ab:>7}{incl:>9.1f}{wo:>8.1f}{(p if p else 0):>7.1f}{d:>7}{mi:>6.1f}")
print("\npaper = Table 2 official score (Browser-Use Stealth + temp 0.6); ours = deterministic, local_process.")
print("w/o_ab = success rate excluding aborted tasks (the paper's reproduction-friendly metric).")
