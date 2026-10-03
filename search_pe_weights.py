"""Micro-grid search over fusion weights, guided by eval_pe_bench slices.

Grid: audio_w x vision_w -> build variant (fuse_pe_single.py) -> eval slice.
Keeps the best variant dir; prints a comparison table.

Usage:
  python3 search_pe_weights.py --out results/pe_bench/search.json
Run AFTER baseline so you can compare against anchor/fused numbers.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
    print("\n".join(tail), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-w", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--vision-w", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    ap.add_argument("--n-coco", type=int, default=64)
    ap.add_argument("--n-esc", type=int, default=100)
    ap.add_argument("--workdir", default="results/pe_bench/search")
    ap.add_argument("--out", default="results/pe_bench/search.json")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    summary = {}
    for aw in args.audio_w:
        for vw in args.vision_w:
            tag = f"a{aw}_v{vw}"
            mdir = os.path.join(args.workdir, tag)
            jpath = os.path.join(args.workdir, f"{tag}.json")
            if not os.path.exists(os.path.join(mdir, "model.safetensors")):
                run([sys.executable, "fuse_pe_single.py", "--out", mdir,
                     "--audio-weight", str(aw), "--vision-weight", str(vw)])
            run([sys.executable, "eval_pe_bench.py", "--models", mdir,
                 "--n-coco", str(args.n_coco), "--n-esc", str(args.n_esc),
                 "--out", jpath])
            rep = json.load(open(jpath))["models"][mdir]
            score = (rep["i2t"]["R@1"] + rep["t2i"]["R@1"]) / 2 + rep["esc_acc"]
            summary[tag] = {**rep, "search_score": score, "dir": mdir}
            print(f"[{tag}] search_score={score:.4f} {rep}", flush=True)

    best = max(summary, key=lambda t: summary[t]["search_score"])
    # keep best, drop the rest to save disk
    for tag, info in summary.items():
        if tag != best and os.path.isdir(info["dir"]):
            shutil.rmtree(info["dir"], ignore_errors=True)
    summary["best"] = best
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"BEST: {best} score={summary[best]['search_score']:.4f} "
          f"kept at {summary[best]['dir']}")
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
