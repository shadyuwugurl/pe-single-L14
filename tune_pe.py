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
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)

    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # freeze towers; train heads + scales + projs only
    train_sub = ("head", "proj", "logit", "data_proj", "norm")
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
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    import numpy as np
    from PIL import Image
    coco = download_coco_train(args.n_coco_train)

    from datasets import load_dataset
    esc_all = load_dataset("ashraq/esc50", split="train", streaming=True)
    esc = []
    for ex in esc_all:
        if int(ex["fold"]) == 5:
            continue  # hold out fold 5 for eval
        s = ex["audio"].get_all_samples()
        arr = np.asarray(s.data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        r = torch.nn.functional.interpolate(
            t, size=int(round(len(arr) * 48000 / 44100)),
            mode="linear", align_corners=False)
        esc.append((r[0, 0].numpy(), f"the sound of {ex['category']}"))
        if len(esc) >= args.n_esc_train:
            break
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
    for ep in range(args.epochs):
        rng = np.random.RandomState(args.seed + ep)
        order = rng.permutation(len(coco))
        for i in range(0, len(coco), args.batch):
            chunk = [coco[j] for j in order[i:i + args.batch]]
            frames = [np.array(Image.open(p).convert("RGB")) for p, _ in chunk]
            texts = [c for _, c in chunk]
            inp = proc(videos=[[f] for f in frames], text=texts,
                       return_tensors="pt")
            inp = {k: (v.to(dev) if hasattr(v, "to") else v)
                   for k, v in inp.items()}
            with torch.autocast(dev, dtype=torch.bfloat16,
                                enabled=(dev == "cuda")):
                o = model(**{k: v for k, v in inp.items()
                              if k in ("input_ids", "attention_mask",
                                       "pixel_values_videos", "input_values",
                                       "padding_mask")})
                loss = info_nce(o.video_embeds, o.text_video_embeds)
            (loss / args.accum).backward()
            if (i // args.batch + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f}", flush=True)
    model.save_pretrained(args.out)
    proc.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    import sys
    sys.exit(main())
