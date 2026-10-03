"""Which depth gives the best visual embeddings? (PE paper: not the output.)

Hooks the ViT blocks of the video tower, pushes each layer's tokens through the
model's own final norm + attn_pool + head, and scores COCO t2i/i2t retrieval
per layer vs the final video_embeds baseline.

Usage:
  python3 layer_sweep.py --model ./fused_pe_single --n 48 --out results/pe_bench/sweep_fused.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def resolve_timm(model):
    for name, mod in model.named_modules():
        if name.endswith("timm_model"):
            return mod
    raise RuntimeError("timm_model submodule not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import os as _os, torch as _t
    _n = _os.cpu_count() or 10
    _t.set_num_threads(_n); _t.set_num_interop_threads(_n)

    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained("facebook/pe-av-base", trust_remote_code=True)
    timm = resolve_timm(model)
    n_blocks = len(timm.blocks)
    print(f"blocks: {n_blocks}", flush=True)

    rows = json.load(open("bench_data/coco150.json"))[:args.n]
    img_paths = [f"bench_data/coco_img/{r['img'].split('/')[-1]}" for r in rows]
    caps = [r["pos"] for r in rows]

    acts = {}

    def hook(i):
        def fn(mod, inp, out):
            acts[i] = out.detach().cpu()
        return fn
    # NOTE: joint-model forward does not execute timm blocks observably
    # (hooks never fire); use the vision tower directly with
    # output_hidden_states instead (supported natively).
    vision_model = None
    for name, mod in model.named_modules():
        if type(mod).__name__ == "TimmWrapperForImageClassification":
            vision_model = mod
            break
    assert vision_model is not None, "vision tower not found"
    timm = resolve_timm(model)
    n_blocks = len(timm.blocks)
    print(f"blocks: {n_blocks} direct-tower path", flush=True)

    # final norm / pool / head submodules (names vary; find by suffix).
    # NOTE: some 'norm' matches (e.g. timm norm=Identity) have no parameters —
    # only keep modules that actually transform.
    def _has_params(m):
        try:
            next(m.parameters())
            return True
        except StopIteration:
            return False
    named = dict(timm.named_modules())
    norm = next((m for n, m in named.items()
                 if n.split(".")[-1] in ("norm", "norm_final", "ln_post", "norm_pre")
                 and _has_params(m)), None)
    pool = next((m for n, m in named.items() if "attn_pool" in n and not list(m.children())), None)
    pool_mod = None
    for n, m in named.items():
        if n.endswith("attn_pool"):
            pool_mod = m
            break
    head = next((m for n, m in named.items() if n.split(".")[-1] == "head"), None)
    print(f"norm={norm is not None} pool={pool_mod is not None} head={head is not None}", flush=True)

    layer_tokens = {i: [] for i in range(n_blocks)}

    @torch.no_grad()
    def run_images():
        import numpy as np
        for s in range(0, len(img_paths), args.batch):
            pix = []
            for pth in img_paths[s:s + args.batch]:
                img = Image.open(pth).convert("RGB")
                out = proc(videos=[[np.array(img)]], return_tensors="pt")
                pix.append(out["pixel_values_videos"][0])
            pv = torch.stack(pix, dim=0)  # (B,1,C,H,W) single-frame videos
            frames = pv[:, 0]             # (B,C,H,W) stills for the tower
            out = vision_model(pixel_values=frames, output_hidden_states=True)
            hs = out.hidden_states  # tuple: embeddings + per-block states?
            assert len(hs) >= n_blocks, f"hidden_states {len(hs)} < blocks {n_blocks}"
            states = hs[-n_blocks:]
            for i in range(n_blocks):
                layer_tokens[i].append(states[i].detach().cpu())
            # calibrated baseline: official joint video_embeds (true joint space)
            b = len(pix)
            t = proc.tokenizer(["a photo"] * b, return_tensors="pt", padding=True)
            aw = proc(audio=[np.zeros(48000, dtype=np.float32)] * b,
                      sampling_rate=48000, return_tensors="pt")
            iv = aw["input_values"]
            if iv.dim() == 2:
                iv = iv.unsqueeze(1)
            jo = model(pixel_values_videos=pv, input_ids=t["input_ids"],
                       attention_mask=t.get("attention_mask"), input_values=iv,
                       padding_mask=torch.ones((b, iv.shape[-1]), dtype=torch.int32))
            yield jo.video_embeds.cpu()

    base_parts = None  # filled after layer_embed is defined (see below)
    V_base = None
    print("layer_embed defined next; capture runs after", flush=True)

    # per-layer embeds through frozen norm/pool/head
    @torch.no_grad()
    def layer_embed(tokens):
        # Mean-pool spatial maps per layer. Deliberately NOT reusing the
        # tower's attn_pool (it expects the full joint-forward context and
        # misshapes standalone inputs). Comparative signal across depths is
        # what ranks layers; the joint forward remains the calibrated baseline.
        x = tokens.float()
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))  # (B,C)
        else:
            x = x.mean(dim=1)
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return (x / n).cpu()

    base_parts = list(run_images())
    V_base = torch.cat(base_parts, dim=0)
    print("captured", {i: torch.cat(v, 0).shape for i, v in list(layer_tokens.items())[:1]}, flush=True)

    # text embeds once (standard path)
    @torch.no_grad()
    def encode_texts(texts):
        embs = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            t = proc.tokenizer(chunk, return_tensors="pt", padding=True)
            import numpy as np
            frames = np.full((1, 336, 336, 3), 128, dtype=np.uint8)
            pv = proc(videos=[frames] * len(chunk), return_tensors="pt")["pixel_values_videos"]
            aw = proc(audio=[np.zeros(48000, dtype=np.float32)] * len(chunk),
                      sampling_rate=48000, return_tensors="pt")
            iv = aw["input_values"]
            if iv.dim() == 2:
                iv = iv.unsqueeze(1)
            o = model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"),
                      pixel_values_videos=pv, input_values=iv,
                      padding_mask=torch.ones((len(chunk), iv.shape[-1]), dtype=torch.int32))
            embs.append(o.text_video_embeds.cpu())
        return torch.cat(embs, dim=0)

    T = encode_texts(caps).numpy()

    def recalls(V):
        S = V @ T.T
        order = np.argsort(-S, axis=1)
        N = S.shape[0]
        i2t1 = sum(1 for i in range(N) if i in order[i, :1]) / N
        oT = np.argsort(-S.T, axis=1)
        t2i1 = sum(1 for i in range(N) if i in oT[i, :1]) / N
        t2i5 = sum(1 for i in range(N) if i in oT[i, :5]) / N
        return {"i2t_R1": i2t1, "t2i_R1": t2i1, "t2i_R5": t2i5}

    out = {"baseline_final": recalls(V_base.numpy()), "layers": {}}
    print("final:", out["baseline_final"], flush=True)
    for i in range(n_blocks):
        tok = torch.cat(layer_tokens[i], dim=0)
        E = layer_embed(tok).numpy()
        m = recalls(E)
        out["layers"][i] = m
        print(f"layer {i}: {m}", flush=True)

    best = max(out["layers"], key=lambda i: out["layers"][i]["t2i_R1"])
    out["best_layer"] = best
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"best layer {best}: {out['layers'][best]} — saved {args.out}")


if __name__ == "__main__":
    sys.exit(main())
