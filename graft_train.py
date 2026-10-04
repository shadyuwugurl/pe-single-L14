"""VLM graft: frozen PE encoder + trainable projector + frozen LFM2.5-2.6B.

Only the projector trains (~10M params): pooled 1024-d media embeds ->
K prefix tokens in LLM space (2048-d). Captioning loss on COCO-train
(image->caption) + ESC audio (clip->"the sound of X").

  python3 graft_train.py --enc pe_single_github/pe-single-L14-tuned \\
      --llm LiquidAI/LFM2.5-2.6B --out ./pe-graft-lfm --n-coco 4000 \
      --epochs 2 --batch 8 --lr 1e-4 --device mps --k-tokens 4

Inference: graft_generate.py --graft ./pe-graft-lfm --image foo.jpg
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Projector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=2048, k=4):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim * k))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        b = x.shape[0]
        t = self.net(x.float()).view(b, self.k, -1)
        return self.norm(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="pe_single_github/pe-single-L14-tuned")
    ap.add_argument("--llm", default="LiquidAI/LFM2.5-2.6B")
    ap.add_argument("--out", default="./pe-graft-lfm")
    ap.add_argument("--n-coco", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k-tokens", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = (args.device or ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu"))
    print("device:", dev, flush=True)

    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from transformers import AutoModelForCausalLM
    enc = AutoModel.from_pretrained(args.enc, trust_remote_code=True,
                                    torch_dtype=torch.float32).to(dev).eval()
    for p in enc.parameters():
        p.requires_grad = False
    eproc = AutoProcessor.from_pretrained("facebook/pe-av-base",
                                          trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad = False
    tok = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    wte = llm.get_input_embeddings()
    proj = Projector(1024, llm.config.hidden_size, args.k_tokens).to(dev)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr)
    start = 0
    if args.resume and os.path.exists(os.path.join(args.resume, "proj.pt")):
        proj.load_state_dict(torch.load(os.path.join(args.resume, "proj.pt"),
                                        map_location=dev, weights_only=True))
        opt.load_state_dict(torch.load(os.path.join(args.resume, "opt.pt"),
                                       map_location=dev, weights_only=True))
        start = int(open(os.path.join(args.resume, "step.txt")).read())
        print("resumed at", start, flush=True)

    # data: reuse COCO-train cache from tune phase when present
    sys.path.insert(0, "pe_single_github")
    from tune_pe import download_coco_train
    coco = download_coco_train(args.n_coco, cache="tune_data/coco_train")

    @torch.no_grad()
    def media_embed(kind, payload):
        if kind == "img":
            out = eproc(videos=[[payload]], return_tensors="pt")
            pv = out["pixel_values_videos"].to(dev)
            b = 1
            t = eproc.tokenizer(["a photo"], return_tensors="pt",
                                padding=True)
            aw = eproc(audio=[np.zeros(48000, dtype=np.float32)],
                       sampling_rate=48000, return_tensors="pt")
            iv = aw["input_values"]
            if iv.dim() == 2:
                iv = iv.unsqueeze(1)
            o = enc(pixel_values_videos=pv,
                    input_ids=t["input_ids"].to(dev),
                    attention_mask=t.get("attention_mask").to(dev),
                    input_values=iv.to(dev),
                    padding_mask=torch.ones(
                        (b, iv.shape[-1]), dtype=torch.int32).to(dev))
            return o.video_embeds
        else:
            aw = eproc(audio=payload, sampling_rate=48000, return_tensors="pt")
            iv = aw["input_values"]
            if iv.dim() == 2:
                iv = iv.unsqueeze(1)
            iv = iv.to(dev)
            t = eproc.tokenizer(["a sound"], return_tensors="pt", padding=True)
            import numpy as _np
            fr = _np.full((1, 336, 336, 3), 128, dtype=_np.uint8)
            pv = eproc(videos=[fr], return_tensors="pt")[
                "pixel_values_videos"].to(dev)
            o = enc(input_values=iv,
                    padding_mask=torch.ones((1, iv.shape[-1]),
                                            dtype=torch.int32).to(dev),
                    input_ids=t["input_ids"].to(dev),
                    attention_mask=t.get("attention_mask").to(dev),
                    pixel_values_videos=pv)
            return o.audio_embeds

    prompt_img = "Describe the image: "
    step = 0
    proj.train()
    for ep in range(args.epochs):
        rng = np.random.RandomState(args.seed + ep)
        order = rng.permutation(len(coco))
        for i in range(0, len(coco), args.batch):
            chunk = [coco[j] for j in order[i:i + args.batch]]
            embs, ids_list, masks = [], [], []
            for pth, cap in chunk:
                img = np.array(Image.open(pth).convert("RGB"))
                with torch.no_grad():
                    me = media_embed("img", img)
                pre = proj(me)
                pin = tok(prompt_img, return_tensors="pt")
                cap_in = tok(cap + tok.eos_token, return_tensors="pt")
                pe = wte(pin["input_ids"].to(dev))
                ce = wte(cap_in["input_ids"].to(dev))
                full = torch.cat([pre.to(ce.dtype), pe, ce], dim=1)
                lab = torch.cat([
                    torch.full((1, pre.shape[1] + pe.shape[1]), -100,
                               dtype=torch.long),
                    cap_in["input_ids"]], dim=1).to(dev)
                attn = torch.ones((1, full.shape[1]),
                                  dtype=torch.long).to(dev)
                embs.append(full)
                ids_list.append(lab)
                masks.append(attn)
            # pad batch (prefix len constant; text varies)
            L = max(f.shape[1] for f in embs)
            B = len(embs)

            def padR(t, L, v, dev):
                if t.shape[1] >= L:
                    return t
                pad = torch.full((t.shape[0], L - t.shape[1]) + t.shape[2:],
                                 v, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=1)

            Ein = torch.cat([padR(f, L, 0.0, dev) for f in embs], dim=0)
            Lab = torch.cat([padR(l, L, -100, dev) for l in ids_list], dim=0)
            Att = torch.cat([padR(m_, L, 0, dev) for m_ in masks], dim=0)
            out = llm(inputs_embeds=Ein, attention_mask=Att)
            logits = out.logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                Lab[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step <= start:
                continue
            if step % 20 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f}",
                      flush=True)
            if step % args.ckpt_every == 0:
                os.makedirs(args.out + "_ckpt", exist_ok=True)
                torch.save(proj.state_dict(),
                           os.path.join(args.out + "_ckpt", "proj.pt"))
                torch.save(opt.state_dict(),
                           os.path.join(args.out + "_ckpt", "opt.pt"))
                open(os.path.join(args.out + "_ckpt", "step.txt"),
                     "w").write(str(step))
    os.makedirs(args.out, exist_ok=True)
    torch.save({"proj": proj.state_dict(), "k": args.k_tokens,
                "enc": args.enc, "llm": args.llm},
               os.path.join(args.out, "graft.pt"))
    print("saved", args.out)


if __name__ == "__main__":
    sys.exit(main())
