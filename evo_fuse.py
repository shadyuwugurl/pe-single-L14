"""Evolution-based PE fusion (tournament GA, no new deps).

Genome = [audio_w, core_w, spatial_w] in [0,1]^3.
  audio tower: (1-aw)*anchor + aw*aframe
  video trunk: (1-cw-sw)*anchor + cw*core + sw*spatial (renormalized if cw+sw>1)
Fitness = micro-slice bench (COCO t2i/i2t R@1 + ESC acc). Text tower is never
touched by blends, so text embeds are encoded ONCE and reused.

Usage:
  python3 evo_fuse.py --pop 6 --gens 3 --out results/pe_bench/evo.json
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def load_all():
    from transformers import AutoModel, AutoProcessor
    from huggingface_hub import hf_hub_download
    from fuse_pe_single import build_vision_donor_map
    anchor_id, af_id = "facebook/pe-av-base", "facebook/pe-a-frame-base"
    print("loading anchor/aframe...", flush=True)
    anchor = AutoModel.from_pretrained(anchor_id, trust_remote_code=True,
                                       torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(anchor_id, trust_remote_code=True)
    af = AutoModel.from_pretrained(af_id, trust_remote_code=True,
                                   torch_dtype=torch.float32).state_dict()
    anchor_params = {k: v.cpu().clone() for k, v in anchor.named_parameters()}
    donors = {}
    for repo, fn in (("facebook/PE-Core-L14-336", "PE-Core-L14-336.pt"),
                     ("facebook/PE-Spatial-L14-448", "PE-Spatial-L14-448.pt")):
        p = hf_hub_download(repo_id=repo, filename=fn)
        raw = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        donors[repo] = {k.replace("module.", ""): v for k, v in raw.items()}
        print(f"donor {repo}: {len(raw)} tensors", flush=True)
    timm_keys = [k for k in anchor_params if ".timm_model.blocks." in k]
    assert timm_keys, "no short-prefix timm blocks in named_parameters"
    prefix = timm_keys[0].split(".timm_model.blocks.")[0] + ".timm_model."
    core_map, _ = build_vision_donor_map(donors["facebook/PE-Core-L14-336"], prefix, anchor_params)
    spat_map, _ = build_vision_donor_map(donors["facebook/PE-Spatial-L14-448"], prefix, anchor_params)
    print(f"mapped core={len(core_map)} spatial={len(spat_map)} prefix={prefix}", flush=True)
    return anchor, proc, anchor_params, af, core_map, spat_map


def apply_genome(model, anchor_params, af_sd, core_map, spat_map, g):
    """Blend directly into live params via copy_ (short param names).

    Raw load_state_dict silently no-ops on PeAudioVideoModel's dual-prefix
    key layout (verified: 0 missing/0 unexpected yet values unchanged), while
    param.copy_ by name works. anchor_params: name->pristine clone.
    core/spat_map keys are short-prefix timm keys matching named_parameters.
    Returns nothing; model is modified in place (callers re-apply per genome).
    """
    import torch
    aw, cw, sw = [float(np.clip(x, 0, 1)) for x in g]
    s = cw + sw
    if s > 1:
        cw, sw = cw / s, sw / s
    live = dict(model.named_parameters())
    # Audio: blend ONLY the standalone audio_model tower. The joint
    # audio_video_encoder copy has same-shaped but semantically different
    # tensors — blending donor weights there destroys audio-text alignment
    # (measured: ESC 0.94 -> 0.04). Donor side likewise restricted to its
    # audio tower.
    donor_audio = {k: v for k, v in af_sd.items()
                   if k.startswith("audio_model.") or ".audio_model." in k}
    with torch.no_grad():
        blended = set()
        for name, p in live.items():
            if not (name.startswith("audio_model.") or name.startswith("audio_plus_text_head")):
                continue
            base = anchor_params[name]
            hit = donor_audio.get(name)
            if hit is None:
                # fall back to suffix match WITHIN audio towers only
                for ak, av in donor_audio.items():
                    if ak.split(".")[-2:] == name.split(".")[-2:] and tuple(av.shape) == tuple(p.shape):
                        hit = av
                        break
            if hit is not None and tuple(hit.shape) == tuple(p.shape):
                p.copy_(((1 - aw) * base.float() + aw * hit.float()).to(p.dtype))
                blended.add(name)
            else:
                p.copy_(base)
        for full, dc in core_map.items():
            if full not in live:
                continue
            ds = spat_map.get(full)
            a = anchor_params[full]
            w0 = 1 - cw - (sw if ds is not None else 0)
            fused = w0 * a.float() + cw * dc.float()
            if ds is not None:
                fused = fused + sw * ds.float()
            live[full].copy_(fused.to(live[full].dtype))
            blended.add(full)
        for full, ds in spat_map.items():
            if full in core_map or full not in live:
                continue
            a = anchor_params[full]
            live[full].copy_((((1 - sw) * a.float()) + sw * ds.float()).to(live[full].dtype))
            blended.add(full)
        # restore untouched params to pristine (guards blend-of-blend drift)
        for name, p in live.items():
            if name in blended:
                continue
            if not torch.equal(p.data, anchor_params[name].data):
                p.copy_(anchor_params[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-coco", type=int, default=32)
    ap.add_argument("--n-esc", type=int, default=50)
    ap.add_argument("--out", default="results/pe_bench/evo.json")
    ap.add_argument("--save-best", default="fused_pe_evo")
    args = ap.parse_args()
    import os as _os, torch as _t
    _n = _os.cpu_count() or 10
    _t.set_num_threads(_n); _t.set_num_interop_threads(_n)
    rng = np.random.RandomState(args.seed)

    anchor, proc, anchor_params, af_sd, core_map, spat_map = load_all()
    sys.path.insert(0, ".")
    from eval_pe_bench import encode_images, encode_audios, encode_texts_video, encode_texts_audio

    rows = json.load(open("bench_data/coco150.json"))[:args.n_coco]
    img_paths = [f"bench_data/coco_img/{r['img'].split('/')[-1]}" for r in rows]
    caps = [r["pos"] for r in rows]
    from datasets import load_dataset
    esc = load_dataset("ashraq/esc50", split="train", streaming=True)
    clips, labels = [], []
    for ex in esc:
        s = ex["audio"].get_all_samples()
        arr = np.asarray(s.data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        r = torch.nn.functional.interpolate(
            t, size=int(round(len(arr) * 48000 / 44100)), mode="linear", align_corners=False)
        clips.append(r[0, 0].numpy())
        labels.append(ex["category"])
        if len(clips) >= args.n_esc:
            break
    srs = [48000] * len(clips)
    cats = sorted(set(labels))
    prompts = [f"the sound of {c}" for c in cats]
    y = np.array([cats.index(c) for c in labels])

    print("pre-encoding text sides (constant across genomes)...", flush=True)
    with torch.no_grad():
        T = encode_texts_video(anchor, proc, caps).numpy()
        P = encode_texts_audio(anchor, proc, prompts).numpy()

    def fitness(g):
        apply_genome(anchor, anchor_params, af_sd, core_map, spat_map, g)
        with torch.no_grad():
            V = encode_images(anchor, proc, img_paths).numpy()
            A = encode_audios(anchor, proc, clips, srs).numpy()
        S = V @ T.T
        N = S.shape[0]
        oi = np.argsort(-S, axis=1)
        ot = np.argsort(-S.T, axis=1)
        i2t1 = sum(1 for i in range(N) if i in oi[i, :1]) / N
        t2i1 = sum(1 for i in range(N) if i in ot[i, :1]) / N
        acc = float(((A @ P.T).argmax(axis=1) == y).mean())
        return (i2t1 + t2i1) / 2 + acc, {"i2t1": i2t1, "t2i1": t2i1, "esc": acc}

    pop = [np.array([0.25, 0.0, 0.0])]
    while len(pop) < args.pop:
        pop.append(rng.rand(3) * np.array([1.0, 0.5, 0.5]))
    hist = []
    for gen in range(args.gens):
        scored = []
        for g in pop:
            f, m = fitness(g)
            scored.append((f, g, m))
            print(f"gen{gen} g={np.round(g,3).tolist()} fit={f:.4f} {m}", flush=True)
        scored.sort(key=lambda x: -x[0])
        hist.append({"gen": gen,
                     "best": {"g": scored[0][1].tolist(), "fit": scored[0][0], **scored[0][2]}})
        elites = [s[1] for s in scored[:2]]
        nxt = list(elites)
        while len(nxt) < args.pop:
            a, b = scored[rng.randint(len(scored) // 2)][1], scored[rng.randint(len(scored) // 2)][1]
            mask = rng.rand(3) < 0.5
            child = np.where(mask, a, b) + rng.randn(3) * 0.08
            nxt.append(np.clip(child, 0, 1))
        pop = nxt

    best = hist[-1]["best"]
    # re-apply best in place and save (disk round-trip path is proven)
    apply_genome(anchor, anchor_params, af_sd, core_map, spat_map, np.array(best["g"]))
    anchor.save_pretrained(args.save_best)
    try:
        from transformers import AutoTokenizer, AutoProcessor as AP
        AutoTokenizer.from_pretrained("facebook/pe-av-base", trust_remote_code=True).save_pretrained(args.save_best)
        AP.from_pretrained("facebook/pe-av-base", trust_remote_code=True).save_pretrained(args.save_best)
    except Exception:
        pass
    json.dump({"history": hist, "best": best, "dir": args.save_best},
              open(args.out, "w"), indent=2)
    print(f"EVO BEST {best} saved -> {args.save_best}")


if __name__ == "__main__":
    sys.exit(main())
