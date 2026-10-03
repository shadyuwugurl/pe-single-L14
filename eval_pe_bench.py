"""Paper-aligned retrieval bench for PE-family single models (CPU-friendly slices).

Benches (real datasets, MMEB-style global-gallery protocol, dot-product):
  COCO  (image<->text): N images from bench_data/coco150.json (MMEB MSCOCO_i2t
        rows; positives = tgt_text[0], verified: row0 = moped guy). Metrics
        i2t/t2i R@1/5/10 over the NxN slice gallery.
  ESC-50 (audio->text): M clips from ashraq/esc50 streaming, 50 prompts
        "the sound of {category}". Metric: zero-shot accuracy.

Usage:
  python3 eval_pe_bench.py --models facebook/pe-av-base ./fused_pe_single \
      --n-coco 150 --n-esc 200 --out results/pe_bench/baseline.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def load_model(mid):
    from transformers import AutoModel, AutoProcessor
    m = AutoModel.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.float32).eval()
    try:
        p = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    except Exception:
        p = AutoProcessor.from_pretrained("facebook/pe-av-base", trust_remote_code=True)
    return m, p


@torch.no_grad()
def encode_images(model, proc, img_paths, batch=8):
    """Single-frame video embeds -> (N, D)."""
    embs = []
    for i in range(0, len(img_paths), batch):
        pix = []
        for pth in img_paths[i:i + batch]:
            img = Image.open(pth).convert("RGB")
            out = proc(videos=[[np.array(img)]], return_tensors="pt")
            pix.append(out["pixel_values_videos"][0])  # (1,C,H,W)
        pv = torch.stack(pix, dim=0)  # (B,1,C,H,W)
        b = len(pix)
        o = model(pixel_values_videos=pv, **_dummy_text(model, proc, b),
                  **_dummy_audio(proc, b))
        embs.append(o.video_embeds.cpu())
    return torch.cat(embs, dim=0)


def _dummy_text(model, proc, b):
    # joint forward needs >=2 modalities; pair video with a throwaway caption
    t = proc.tokenizer(["a photo"] * b, return_tensors="pt", padding=True)
    return {"input_ids": t["input_ids"], "attention_mask": t.get("attention_mask")}


@torch.no_grad()
def encode_texts_video(model, proc, texts, batch=64):
    embs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        t = proc.tokenizer(chunk, return_tensors="pt", padding=True)
        o = model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"),
                  **_dummy_video(proc, len(chunk)), **_dummy_audio(proc, len(chunk)))
        embs.append(o.text_video_embeds.cpu())
    return torch.cat(embs, dim=0)


def _dummy_video(proc, b):
    import numpy as np
    # mid-gray still (in-distribution-ish neutral context; zeros distort text path)
    frames = (np.full((1, 336, 336, 3), 128, dtype=np.uint8))
    out = proc(videos=[frames] * b, return_tensors="pt")
    return {"pixel_values_videos": out["pixel_values_videos"]}


@torch.no_grad()
def encode_audios(model, proc, waves, srs, batch=4):
    embs = []
    for i in range(0, len(waves), batch):
        chunk_w, chunk_s = waves[i:i + batch], srs[i:i + batch]
        outs = [proc(audio=w, sampling_rate=s, return_tensors="pt")
                for w, s in zip(chunk_w, chunk_s)]
        iv = torch.stack([o["input_values"][0, 0] for o in outs], dim=0)
        L = iv.shape[1]
        iv = iv.unsqueeze(1)  # (B,1,T)
        pm = torch.ones((len(chunk_w), L), dtype=torch.int32)
        t = proc.tokenizer(["a sound"] * len(chunk_w), return_tensors="pt", padding=True)
        o = model(input_values=iv, padding_mask=pm, input_ids=t["input_ids"],
                  attention_mask=t.get("attention_mask"),
                  **_dummy_video(proc, len(chunk_w)))
        embs.append(o.audio_embeds.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_texts_audio(model, proc, texts, batch=64):
    embs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        t = proc.tokenizer(chunk, return_tensors="pt", padding=True)
        o = model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"),
                  **_dummy_audio(proc, len(chunk)), **_dummy_video(proc, len(chunk)))
        embs.append(o.text_audio_embeds.cpu())
    return torch.cat(embs, dim=0)


def _dummy_audio(proc, b):
    import numpy as np
    w = np.zeros(48000, dtype=np.float32)
    out = proc(audio=[w] * b, sampling_rate=48000, return_tensors="pt")
    iv = out["input_values"]
    if iv.dim() == 2:
        iv = iv.unsqueeze(1)
    return {"input_values": iv, "padding_mask": torch.ones((b, iv.shape[-1]), dtype=torch.int32)}


def recall_at(S, Ks=(1, 5, 10)):
    # S: (Nq, Ng), correct = diagonal
    order = np.argsort(-S, axis=1)
    N = S.shape[0]
    out = {}
    for k in Ks:
        hits = sum(1 for i in range(N) if i in order[i, :k])
        out[f"R@{k}"] = hits / N
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-coco", type=int, default=150)
    ap.add_argument("--n-esc", type=int, default=200)
    ap.add_argument("--batch-img", type=int, default=8)
    ap.add_argument("--batch-aud", type=int, default=4)
    ap.add_argument("--out", default="results/pe_bench/baseline.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import os as _os, torch as _t
    _n = _os.cpu_count() or 10
    _t.set_num_threads(_n); _t.set_num_interop_threads(_n)
    rng = np.random.RandomState(args.seed)

    rows = json.load(open("bench_data/coco150.json"))[:args.n_coco]
    img_paths = [f"bench_data/coco_img/{r['img'].split('/')[-1]}" for r in rows]
    caps = [r["pos"] for r in rows]
    assert all(os.path.exists(p) for p in img_paths), "missing coco images"

    from datasets import load_dataset
    esc = load_dataset("ashraq/esc50", split="train", streaming=True)
    clips, labels = [], []
    for ex in esc:
        s = ex["audio"].get_all_samples()
        arr = np.asarray(s.data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        clips.append(arr)
        labels.append(ex["category"])
        if len(clips) >= args.n_esc:
            break
    srs = [44100] * len(clips)
    # PE audio tower is fixed at 48kHz: resample once (identical inputs for all models)
    clips48 = []
    for arr in clips:
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        r = torch.nn.functional.interpolate(t, size=int(round(len(arr) * 48000 / 44100)),
                                            mode="linear", align_corners=False)
        clips48.append(r[0, 0].numpy())
    clips, srs = clips48, [48000] * len(clips48)
    cats = sorted(set(labels))
    prompts = [f"the sound of {c}" for c in cats]
    y = np.array([cats.index(c) for c in labels])

    report = {"n_coco": len(rows), "n_esc": len(clips), "n_classes": len(cats), "models": {}}
    for mid in args.models:
        t0 = time.time()
        model, proc = load_model(mid)
        V = encode_images(model, proc, img_paths, args.batch_img).numpy()
        T = encode_texts_video(model, proc, caps).numpy()
        S = V @ T.T
        i2t = recall_at(S)
        t2i = recall_at(S.T)
        A = encode_audios(model, proc, clips, srs, args.batch_aud).numpy()
        P = encode_texts_audio(model, proc, prompts).numpy()
        pred = (A @ P.T).argmax(axis=1)
        acc = float((pred == y).mean())
        rep = {"i2t": i2t, "t2i": t2i, "esc_acc": acc,
               "mean_r1": (i2t["R@1"] + t2i["R@1"]) / 2,
               "secs": round(time.time() - t0, 1)}
        report["models"][mid] = rep
        print(f"{mid}: i2t {i2t} t2i {t2i} esc {acc:.4f} ({rep['secs']}s)", flush=True)
        del model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
