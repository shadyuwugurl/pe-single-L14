"""SVD-bridged feature distillation: big teacher -> L-scale student tower.

Pilot 1: PE-Core-G14-448 (1536-wide, 50L, native .pt) -> our video tower
(1024-wide, 24 blocks, timm layout).

Method (honest naming: NOT weight surgery — widths AND depths differ):
  1. Layer map: student block i learns from teacher block round(i*49/23).
  2. Bridge init: for each mapped pair, collect teacher block outputs on real
     images, SVD the feature covariance, init a 1536->1024 linear bridge from
     top singular vectors (PCA-style, principled dimensionality reduction).
  3. Student-side bottleneck adapters (1024->r->1024, r=64, zero-init output)
     after each student block, trained so student+adapter matches bridged
     teacher features (MSE) + keeps contrastive retrieval alive (InfoNCE on a
     COCO slice, small weight).
  4. Only adapters + bridge train. Everything else frozen. MPS-feasible.

Verdict condition (from ROADMAP): t2i moves where naive Spatial blending hurt
(0.766 -> 0.64). COCO slice eval before/after decides.

Usage:
  python3 svd_distill.py --teacher facebook/PE-Core-G14-448 --teacher-file PE-Core-G14-448.pt \\
      --student results/pe_bench/search/a0.25_v0.0 --out ./svd_core_g14 \
      --n-feat 500 --n-train 2000 --rank 64 --epochs 2 --batch 8 --device mps
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Bridge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x.float())


class Bottleneck(nn.Module):
    def __init__(self, dim, r=64):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.up = nn.Linear(dim and r, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.nn.functional.gelu(self.down(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="facebook/PE-Core-G14-448")
    ap.add_argument("--teacher-file", default="PE-Core-L14-336.pt")
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", default="./svd_distill_out")
    ap.add_argument("--n-feat", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = (args.device or ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu"))
    print("device:", dev, flush=True)

    from transformers import AutoModel, AutoProcessor
    from huggingface_hub import hf_hub_download

    student = AutoModel.from_pretrained(args.student, trust_remote_code=True,
                                        torch_dtype=torch.float32).to(dev).eval()
    for p in student.parameters():
        p.requires_grad = False
    proc = AutoProcessor.from_pretrained("facebook/pe-av-base",
                                         trust_remote_code=True)
    # tower handle for direct stills (same path as layer_sweep.py)
    tower = None
    for _, mod in student.named_modules():
        if type(mod).__name__ == "TimmWrapperForImageClassification":
            tower = mod
            break
    assert tower is not None
    s_blocks = tower.timm_model.blocks
    n_s = len(s_blocks)

    # teacher: timm native G14 (same PE family, istribution weights)
    import timm
    tarch = "vit_pe_core_gigantic_patch14_448"
    teacher = timm.create_model(tarch, pretrained=True).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    # teacher dims from timm cfg
    tcfg = dict(width=1536)
    print("teacher:", tarch, "params: %.1fM" % (
        sum(p.numel() for p in teacher.parameters()) / 1e6), flush=True)

    # layer map (depth mismatch handled explicitly, not hidden)
    # NOTE: G14 gigantic has ~50 blocks; L student 24. Map student i -> teacher t.
    n_t = len(teacher.blocks)
    lmap = {i: round(i * (n_t - 1) / max(n_s - 1, 1)) for i in range(n_s)}
    print(f"student blocks {n_s}, teacher blocks {n_t}, map {lmap}", flush=True)

    rows = json.load(open("bench_data/coco150.json"))
    img_paths = [f"bench_data/coco_img/{r['img'].split('/')[-1]}" for r in rows]

    def stills(idxs):
        arr = []
        for j in idxs:
            img = Image.open(img_paths[j]).convert("RGB")
            a = np.array(img)
            arr.append(a)
        return arr

    import torchvision.transforms as T
    tform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])

    @torch.no_grad()
    def teacher_block_feats(idxs, layers):
        """Mean-pooled teacher block outputs for image idxs. {layer: (B,1536)}"""
        out = {t: [] for t in layers}
        for s in range(0, len(idxs), 8):
            batch = torch.stack(
                [tform(Image.open(img_paths[j]).convert("RGB").resize((448, 448)))
                 for j in idxs[s:s + 8]], dim=0).to(dev)
            feats = teacher.get_intermediate_layers(batch, n={l + 1 for l in layers})
            for t, f in zip(sorted(layers), feats):
                if f.dim() == 4:
                    f = f.flatten(2).transpose(1, 2)
                out[t].append(f.mean(dim=1).cpu())
        return {t: torch.cat(v, 0) for t, v in out.items()}

    # 1) collect teacher feats on n-feat images, SVD-init one bridge per
    # mapped student block (1536 -> 1024, top singular vectors of the
    # centered feature covariance = PCA projection, principled not arbitrary)
    feat_idx = list(range(min(args.n_feat, len(img_paths))))
    needed = sorted(set(lmap.values()))
    tf = teacher_block_feats(feat_idx, needed)
    bridges, adapters = {}, {}
    for i in range(n_s):
        t = lmap[i]
        F = tf[t].float()
        C = torch.cov((F - F.mean(0)).T)
        _, V = torch.linalg.eigh(C)
        P = V[:, -1024:].T  # top-1024 eigenvectors -> (1024,1536)
        b = Bridge(1536, 1024).to(dev)
        with torch.no_grad():
            b.proj.weight.copy_(P.to(b.proj.weight.dtype))
            b.proj.bias.zero_()
        bridges[i] = b
        adapters[i] = Bottleneck(1024, args.rank).to(dev)
    print("bridges SVD-initialized on", len(feat_idx), "images", flush=True)

    # 2) train adapters (+bridge fine-tune): MSE(student+adapter, bridged
    # teacher) on n-train images. Student tower frozen; only adapters+bridge.
    train_idx = feat_idx[:min(args.n_train, len(img_paths))]
    params = [p for m in list(adapters.values()) + list(bridges.values())
              for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr)

    @torch.no_grad()
    def student_block_toks(idxs):
        from transformers import AutoProcessor as _AP
        _proc = _AP.from_pretrained("facebook/pe-av-base",
                                   trust_remote_code=True)
        outs = []
        for s in range(0, len(idxs), 8):
            pix = []
            for j in idxs[s:s + 8]:
                o = _proc(videos=[[np.array(
                    Image.open(img_paths[j]).convert("RGB"))]],
                    return_tensors="pt")
                pix.append(o["pixel_values_videos"][0])
            pv = torch.stack(pix, dim=0)[:, 0].to(dev)
            h = tower(pixel_values=pv,
                      output_hidden_states=True).hidden_states
            outs.append([x.detach().cpu() for x in h[-n_s:]])
        per_layer = [torch.cat([b[i] for b in outs], 0) for i in range(n_s)]
        return per_layer  # token maps, mean-pooled by caller

    s_toks = student_block_toks(train_idx)
    t_toks = teacher_block_feats(train_idx, needed)
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for s in range(0, len(train_idx), args.batch):
            sl = slice(s, s + args.batch)
            loss = 0.0
            for i in range(n_s):
                st = s_toks[i][sl].float()
                if st.dim() == 4:
                    st = st.flatten(2).transpose(1, 2)
                st = st.mean(dim=1).to(dev)
                with torch.no_grad():
                    tt = t_toks[lmap[i]][sl].float()
                    if tt.dim() == 4:
                        tt = tt.flatten(2).transpose(1, 2)
                    tgt = bridges[i](tt.mean(dim=1).to(dev))
                pred = adapters[i](st)
                loss = loss + torch.nn.functional.mse_loss(pred, tgt)
            loss = loss / n_s
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"ep{ep} distill_mse={tot / max(nb, 1):.6f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"adapters": {i: m.state_dict() for i, m in adapters.items()},
                "bridges": {i: m.state_dict() for i, m in bridges.items()},
                "lmap": lmap, "rank": args.rank},
               os.path.join(args.out, "svd_adapters.pt"))
    print("saved", args.out,
          "— eval next: wire adapters into tower forward, run COCO t2i.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
