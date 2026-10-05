"""GPU alignment fine-tune for pe-single-L14 (PE-style, heads-first).

Freezes both towers; trains joint contrastive heads + logit scales with
InfoNCE over (video,text) + (audio,text) batches.

Data (all public, downloaded on demand into ./tune_data):
  - COCO train2014 slice: captions_train2014.json + N images from CDN
  - ESC-50 folds 1-4 (audio-text pairs via "the sound of {category}" prompts)

Run (single GPU, e.g. RunPod A100/L4):
  pip install datasets torchcodec soundfile librosa
  python3 tune_pe.py --model MC7ever/pe-single-L14 --out ./pe-single-L14-tuned \
    --n-coco-train 4000 --n-esc-train 1600 --epochs 2 --lr 1e-5 --batch 32

Then: eval_pe_bench.py --models ./pe-single-L14-tuned (full slices).
"""

import argparse
import json
import os
import urllib.request

import torch


def download_coco_train(n, cache="tune_data/coco_train"):
    os.makedirs(f"{cache}/img", exist_ok=True)
    ann = f"{cache}/captions_train2014.json"
    if not os.path.exists(ann):
        print("downloading COCO train captions (~240MB)...", flush=True)
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            f"{cache}/ann.zip")
        import zipfile
        zipfile.ZipFile(f"{cache}/ann.zip").extractall(cache)
        os.rename(f"{cache}/annotations/captions_train2014.json", ann)
    data = json.load(open(ann))
    img_by_id = {i["id"]: i["file_name"] for i in data["images"]}
    pairs = []
    for c in data["annotations"]:
        fn = img_by_id[c["image_id"]]
        dst = f"{cache}/img/{fn}"
        if not os.path.exists(dst):
            try:
                urllib.request.urlretrieve(
                    f"http://images.cocodataset.org/train2014/{fn}", dst)
            except Exception:
                continue
        pairs.append((dst, c["caption"]))
        if len(pairs) >= n:
            break
    print(f"coco train pairs: {len(pairs)}", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MC7ever/pe-single-L14")
    ap.add_argument("--out", default="./pe-single-L14-tuned")
    ap.add_argument("--n-coco-train", type=int, default=4000)
    ap.add_argument("--n-esc-train", type=int, default=1600)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="default: cuda if available else mps if available else cpu")
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="save resume checkpoint every N steps (crash-safe)")
    ap.add_argument("--audio-every", type=int, default=0,
                    help="every Nth batch is audio<->text (ESC pairs); 0 = video only")
    ap.add_argument("--amp", action="store_true",
                    help="autocast-bf16 (proven MPS-safe, cos 0.9999 vs fp32)")
    ap.add_argument("--resume", default=None,
                    help="resume from a step checkpoint dir")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.device:
        dev = args.device
    else:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = (dev == "cuda") or args.amp
    print("device:", dev, "amp:", use_amp, flush=True)

    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.float32).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # freeze towers; train heads + scales + projs only.
    # NOTE: 'norm' deliberately NOT trainable — tower LayerNorms matching it
    # would force full-graph backward (all activations retained) and OOM/swap
    # the machine. Heads-only keeps backward inside the head subgraph.
    train_sub = ("head", "proj", "logit_scale", "logit_bias", "data_proj")
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        if any(s in name for s in train_sub):
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    print(f"trainable {n_train/1e6:.1f}M / frozen {n_frozen/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    if dev == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    if args.resume and os.path.exists(os.path.join(args.resume, "opt.pt")):
        # NOTE: raw load_state_dict silently no-ops on PeAudioVideoModel's
        # dual-prefix layout (verified) — restore via copy_ by param name.
        ck = torch.load(os.path.join(args.resume, "model_ckpt.pt"),
                        map_location=dev, weights_only=True)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in ck and tuple(ck[name].shape) == tuple(p.shape):
                    p.copy_(ck[name])
        opt.load_state_dict(torch.load(os.path.join(args.resume, "opt.pt"),
                                       map_location=dev, weights_only=True))
        start_step = int(open(os.path.join(args.resume, "step.txt")).read())
        print(f"resumed at step {start_step}", flush=True)
    else:
        start_step = 0

    import numpy as np
    from PIL import Image
    coco = download_coco_train(args.n_coco_train)

    from datasets import load_dataset
    esc_cache_npy = "../tune_data/esc1600.npy"
    esc_cache_lab = "../tune_data/esc1600_labels.json"
    esc = []
    if os.path.exists(esc_cache_npy) and os.path.exists(esc_cache_lab):
        _cm = np.load(esc_cache_npy, mmap_mode="r")
        _labels = json.load(open(esc_cache_lab))
        for c, e in zip(_cm, _labels):
            cat = e["cat"] if isinstance(e, dict) else str(e)
            fold = e.get("fold", 1) if isinstance(e, dict) else 1
            if fold == 5:
                continue
            esc.append((np.asarray(c, dtype=np.float32),
                        f"the sound of {cat}"))
        esc = esc[:args.n_esc_train]
        print(f"esc train pairs: {len(esc)} (disk cache, mmap)", flush=True)
    else:
        esc_all = load_dataset("ashraq/esc50", split="train", streaming=True)
        _cc, _ll = [], []
        for ex in esc_all:
            s = ex["audio"].get_all_samples()
            arr = np.asarray(s.data, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            r = torch.nn.functional.interpolate(
                t, size=int(round(len(arr) * 48000 / 44100)),
                mode="linear", align_corners=False)
            clip = r[0, 0].numpy()
            _cc.append(clip)
            _ll.append({"cat": ex["category"], "fold": int(ex["fold"])})
            if int(ex["fold"]) != 5:
                esc.append((clip, f"the sound of {ex['category']}"))
            if len(_cc) >= 2000:
                break
        os.makedirs("../tune_data", exist_ok=True)
        L = max(len(c) for c in _cc)
        _mat = np.zeros((len(_cc), L), dtype=np.float32)
        for i, c in enumerate(_cc):
            _mat[i, :len(c)] = c
        np.save(esc_cache_npy, _mat)  # plain fast write, mmap-readable
        json.dump(_ll, open(esc_cache_lab, "w"))
        esc = esc[:args.n_esc_train]
        print(f"esc train pairs: {len(esc)} (fold 5 held out, cached)", flush=True)
    print(f"esc train pairs: {len(esc)} (fold 5 held out)", flush=True)

    def info_nce(a, b):
        a = torch.nn.functional.normalize(a.float(), dim=-1)
        b = torch.nn.functional.normalize(b.float(), dim=-1)
        s = a @ b.T * np.exp(4.0)
        tgt = torch.arange(s.shape[0], device=s.device)
        return (torch.nn.functional.cross_entropy(s, tgt)
                + torch.nn.functional.cross_entropy(s.T, tgt)) / 2

    model.train()
    step = 0
    n_coco_steps = (len(coco) + args.batch - 1) // args.batch
    esc_idx = 0
    for ep in range(args.epochs):
        rng = np.random.RandomState(args.seed + ep)
        order = rng.permutation(len(coco))
        for bi in range(n_coco_steps):
            use_audio = (args.audio_every > 0 and args.n_esc_train > 0
                         and (bi + 1) % args.audio_every == 0 and esc)
            if use_audio:
                chunk = [esc[(esc_idx + j) % len(esc)] for j in range(args.batch)]
                esc_idx += args.batch
                ivs = []
                for w, _ in chunk:
                    o = proc(audio=w, sampling_rate=48000, return_tensors="pt")
                    v = o["input_values"]
                    if v.dim() == 2:
                        v = v.unsqueeze(1)
                    ivs.append(v[0, 0])
                L = max(v.shape[0] for v in ivs)
                iv = torch.stack([torch.nn.functional.pad(
                    v, (0, L - v.shape[0])) for v in ivs], dim=0).unsqueeze(1)
                pm = torch.ones((len(chunk), L), dtype=torch.int32)
                texts = [c for _, c in chunk]
                t = proc.tokenizer(texts, return_tensors="pt", padding=True,
                                   truncation=True)
                fr = np.full((1, 336, 336, 3), 128, dtype=np.uint8)
                pv = proc(videos=[fr] * len(chunk),
                          return_tensors="pt")["pixel_values_videos"]
                inp = {"input_values": iv, "padding_mask": pm,
                       "input_ids": t["input_ids"],
                       "attention_mask": t.get("attention_mask"),
                       "pixel_values_videos": pv}
                is_audio = True
            else:
                i = bi * args.batch
                chunk = [coco[j] for j in order[i:i + args.batch]]
                frames = [np.array(Image.open(p).convert("RGB"))
                          for p, _ in chunk]
                texts = [c for _, c in chunk]
                pv = proc(videos=[[f] for f in frames],
                          return_tensors="pt")["pixel_values_videos"]
                t = proc.tokenizer(texts, return_tensors="pt", padding=True,
                                   truncation=True)
                aw = proc(audio=[np.zeros(48000, dtype=np.float32)] * len(chunk),
                          sampling_rate=48000, return_tensors="pt")
                iv = aw["input_values"]
                if iv.dim() == 2:
                    iv = iv.unsqueeze(1)
                inp = {"pixel_values_videos": pv, "input_ids": t["input_ids"],
                       "attention_mask": t.get("attention_mask"),
                       "input_values": iv,
                       "padding_mask": torch.ones((len(chunk), iv.shape[-1]),
                                                  dtype=torch.int32)}
            inp = {k: (v.to(dev) if hasattr(v, "to") else v)
                   for k, v in inp.items()}
            with torch.autocast(dev, dtype=torch.bfloat16,
                                enabled=use_amp):
                o = model(**{k: v for k, v in inp.items()
                              if k in ("input_ids", "attention_mask",
                                       "pixel_values_videos", "input_values",
                                       "padding_mask")})
                loss = (info_nce(o.audio_embeds, o.text_audio_embeds)
                        if use_audio else
                        info_nce(o.video_embeds, o.text_video_embeds))
            # NOTE: audio batches reuse the same loop shape in v2; v1 trains
            # video<->text alignment (the regression axis). Audio-text pairs
            # (esc) enter via --epochs mixing in v2 after video converges.
            (loss / args.accum).backward()
            n_done = bi + 1 + ep * n_coco_steps
            if n_done % args.accum == 0:
                if dev == "cuda" and use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
            step += 1
            if step <= start_step:
                opt.zero_grad()
                continue
            if step % 20 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f}", flush=True)
            if step % args.ckpt_every == 0:
                os.makedirs(args.out + "_ckpt", exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(args.out + "_ckpt", "model_ckpt.pt"))
                torch.save(opt.state_dict(),
                           os.path.join(args.out + "_ckpt", "opt.pt"))
                open(os.path.join(args.out + "_ckpt", "step.txt"), "w").write(str(step))
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    import sys
    sys.exit(main())
