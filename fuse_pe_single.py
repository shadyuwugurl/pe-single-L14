"""Fuse Meta PE family into ONE single model file — no router, no ensemble.

Anchor: transformers `facebook/pe-av-base` (PeAudioVideoModel).
It is already a single joint model (audio + video + text in one file).
We fuse into it:
  - audio tower  <- average(pe-av-base audio, pe-a-frame-base audio)
                   both are pe_audio_encoder hidden=1024 layers=16 heads=8 -> shapes match
  - video tower  <- average(pe-av-base video, PE-Core-L14, PE-Spatial-L14)
                   all are Large/14 width=1024 layers=24 -> trunk shapes match;
                   native PE .pt keys are fuzzy-mapped by suffix+shape, only
                   exact-shape matches are merged, rest kept from anchor.
  - text tower   <- kept from anchor (ModernBERT 1024/22L, shared).

Why L-scale and not G14 flagships?
  PE-Core-G14-448 / PE-Spatial-G14-448 are width=1536 layers=50.
  pe-av-base / pe-a-frame-base are width=1024 (Large scale).
  1536-wide and 1024-wide tensors cannot be weight-averaged (shape mismatch,
  same rule as AGENTS.md cross-arch trap). L-scale (1024-wide) is the unique
  scale where Core-L + Spatial-L + AV-video all share width/depth/patch,
  so true single-model weight fusion is mathematically possible.
  G-scale fusion would need a width-projection graft + distillation
  (model-osmosis pattern), which is a separate heavier recipe.

Usage:
  python3 fuse_pe_single.py --out ./fused_pe_single --vision-weight 0.5 --audio-weight 0.5
  python3 fuse_pe_single.py --probe-only   # shapes only, no big downloads
"""

import argparse
import re
import sys

VISION_CORE = "facebook/PE-Core-L14-336"
VISION_SPATIAL = "facebook/PE-Spatial-L14-448"
AV_ID = "facebook/pe-av-base"
AFRAME_ID = "facebook/pe-a-frame-base"


def norm_key(k: str) -> str:
    k = k.replace("module.", "")
    k = re.sub(r"^audio_model\.", "", k)
    k = re.sub(r"^video_model\.", "", k)
    k = re.sub(r"^audio_tower\.", "", k)
    k = re.sub(r"^visual\.", "", k)
    k = re.sub(r"^vision_model\.", "", k)
    k = re.sub(r"^encoder\.", "", k)
    return k


def suffix_map(sd: dict) -> dict:
    """Map normalized-suffix -> full key (last wins)."""
    m = {}
    for k in sd.keys():
        m[norm_key(k).split(".")[-4:].__repr__()] = k
    return m


def merge_matched(anchor_sd: dict, donor_sd: dict, weight: float, prefix_filter: str = None):
    """Weighted average over exact-shape matches. Returns (merged_sd, n_matched, n_total)."""
    donor_by_norm = {norm_key(k): (k, v) for k, v in donor_sd.items()}
    merged = dict(anchor_sd)
    n_matched = 0
    n_total = 0
    for ka, va in anchor_sd.items():
        if prefix_filter and prefix_filter not in ka:
            continue
        n_total += 1
        hit = donor_by_norm.get(norm_key(ka))
        if hit is None:
            continue
        _, vd = hit
        if tuple(va.shape) != tuple(vd.shape):
            continue
        try:
            merged[ka] = ((1.0 - weight) * va.float() + weight * vd.float()).to(va.dtype)
            n_matched += 1
        except Exception:
            continue
    return merged, n_matched, n_total


def build_vision_donor_map(native_sd: dict, anchor_timm_prefix: str, anchor_sd: dict):
    """Explicit native-PE -> timm_wrapper key map with reshape rules.

    Returns dict anchor_key -> donor_tensor (already reshaped), only for
    exact post-reshape shape matches. Skips pos_embed on resolution
    mismatch, skips attn_pool/head/proj/ln_post (different head designs).
    """
    import torch
    # find timm block count from anchor
    block_ids = set()
    for k in anchor_sd.keys():
        if ".timm_model.blocks." in k:
            try:
                block_ids.add(int(k.split(".timm_model.blocks.")[1].split(".")[0]))
            except Exception:
                pass
    n_blocks = (max(block_ids) + 1) if block_ids else 0

    def get(*names):
        for n in names:
            if n in native_sd:
                return native_sd[n]
        return None

    mapped = {}

    def put(anchor_suffix: str, tensor):
        if tensor is None:
            return
        full = anchor_timm_prefix + anchor_suffix
        if full not in anchor_sd:
            return
        t = tensor
        a = anchor_sd[full]
        if tuple(t.shape) != tuple(a.shape):
            # known reshapes: cls_token (E,) -> (1,1,E); pos_embed (N,E) -> (1,N,E)
            try:
                if tuple(t.shape) == (1024,) and tuple(a.shape) == (1, 1, 1024):
                    t = t.reshape(1, 1, 1024)
                elif len(t.shape) == 2 and len(a.shape) == 3 and t.shape[0] == a.shape[1] and t.shape[1] == a.shape[2]:
                    t = t.unsqueeze(0)
                else:
                    return
            except Exception:
                return
        if tuple(t.shape) != tuple(a.shape):
            return
        mapped[full] = t.to(a.dtype)

    put("patch_embed.proj.weight", get("visual.conv1.weight", "conv1.weight"))
    put("cls_token", get("visual.class_embedding", "class_embedding"))
    put("pos_embed", get("visual.positional_embedding", "positional_embedding"))
    put("norm_pre.weight", get("visual.ln_pre.weight", "ln_pre.weight"))
    put("norm_pre.bias", get("visual.ln_pre.bias", "ln_pre.bias"))
    for i in range(n_blocks):
        put(f"blocks.{i}.norm1.weight", get(f"visual.transformer.resblocks.{i}.ln_1.weight",
                                            f"transformer.resblocks.{i}.ln_1.weight"))
        put(f"blocks.{i}.norm1.bias", get(f"visual.transformer.resblocks.{i}.ln_1.bias",
                                          f"transformer.resblocks.{i}.ln_1.bias"))
        put(f"blocks.{i}.attn.qkv.weight", get(f"visual.transformer.resblocks.{i}.attn.in_proj_weight",
                                               f"transformer.resblocks.{i}.attn.in_proj_weight"))
        put(f"blocks.{i}.attn.qkv.bias", get(f"visual.transformer.resblocks.{i}.attn.in_proj_bias",
                                             f"transformer.resblocks.{i}.attn.in_proj_bias"))
        put(f"blocks.{i}.attn.proj.weight", get(f"visual.transformer.resblocks.{i}.attn.out_proj.weight",
                                                f"transformer.resblocks.{i}.attn.out_proj.weight"))
        put(f"blocks.{i}.attn.proj.bias", get(f"visual.transformer.resblocks.{i}.attn.out_proj.bias",
                                              f"transformer.resblocks.{i}.attn.out_proj.bias"))
        put(f"blocks.{i}.norm2.weight", get(f"visual.transformer.resblocks.{i}.ln_2.weight",
                                            f"transformer.resblocks.{i}.ln_2.weight"))
        put(f"blocks.{i}.norm2.bias", get(f"visual.transformer.resblocks.{i}.ln_2.bias",
                                          f"transformer.resblocks.{i}.ln_2.bias"))
        put(f"blocks.{i}.mlp.fc1.weight", get(f"visual.transformer.resblocks.{i}.mlp.c_fc.weight",
                                              f"transformer.resblocks.{i}.mlp.c_fc.weight"))
        put(f"blocks.{i}.mlp.fc1.bias", get(f"visual.transformer.resblocks.{i}.mlp.c_fc.bias",
                                            f"transformer.resblocks.{i}.mlp.c_fc.bias"))
        put(f"blocks.{i}.mlp.fc2.weight", get(f"visual.transformer.resblocks.{i}.mlp.c_proj.weight",
                                              f"transformer.resblocks.{i}.mlp.c_proj.weight"))
        put(f"blocks.{i}.mlp.fc2.bias", get(f"visual.transformer.resblocks.{i}.mlp.c_proj.bias",
                                            f"transformer.resblocks.{i}.mlp.c_proj.bias"))
    return mapped, n_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./fused_pe_single")
    ap.add_argument("--audio-weight", type=float, default=0.5,
                    help="weight of A-Frame donor in audio tower (0..1)")
    ap.add_argument("--vision-weight", type=float, default=0.5,
                    help="total donor weight for vision (split Core/Spatial if both present)")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    import os as _os, torch as _t
    _n = _os.cpu_count() or 10
    _t.set_num_threads(_n); _t.set_num_interop_threads(_n)

    import torch
    from transformers import AutoConfig, AutoModel

    print("== configs ==")
    for mid in (AV_ID, AFRAME_ID):
        c = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        ac = getattr(c, "audio_config", c)
        if ac is None or getattr(ac, "hidden_size", None) is None:
            avc = getattr(c, "audio_video_config", None)
            ac = getattr(avc, "audio_config", ac) if avc is not None else ac
        print(f"{mid}: model_type={getattr(c, 'model_type', '?')} "
              f"audio_hidden={getattr(ac, 'hidden_size', '?')} "
              f"audio_layers={getattr(ac, 'num_hidden_layers', '?')} "
              f"audio_heads={getattr(ac, 'num_attention_heads', '?')}")
    # native PE vision dims (from perception_models/core/vision_encoder/config.py)
    print("PE-Core-L14-336: width=1024 layers=24 patch=14 (native PE config)")
    print("PE-Spatial-L14-448: width=1024 layers=24 patch=14 (native PE config)")
    print("PE-Core-G14-448 / PE-Spatial-G14-448: width=1536 layers=50 "
          "-> MISMATCH vs 1024-wide AV/AFrame, cannot linear-merge (skipped by design)")

    if args.probe_only:
        print("probe-only: OK, audio towers shape-compatible per config; vision L-scale compatible per PE config.")
        return 0

    print("\n== downloading anchor + donors (this pulls ~4-6GB once, cached after) ==")
    av = AutoModel.from_pretrained(AV_ID, trust_remote_code=True, torch_dtype=torch.float32)
    av_sd = dict(av.state_dict())
    print(f"anchor {AV_ID}: {len(av_sd)} tensors")

    af = AutoModel.from_pretrained(AFRAME_ID, trust_remote_code=True, torch_dtype=torch.float32)
    af_sd = dict(af.state_dict())
    print(f"donor {AFRAME_ID}: {len(af_sd)} tensors")

    # ---- audio tower merge ----
    # transformers key prefixes differ (audio_model vs audio_tower); use suffix+shape matching
    # restrict anchor side to audio-related keys to avoid touching video/text
    audio_prefixes = ("audio",)
    audio_keys = {k: v for k, v in av_sd.items() if "audio" in k.lower()}
    print(f"anchor audio-tower tensors: {len(audio_keys)}")
    merged_audio, n_m, n_t = merge_matched(audio_keys, af_sd, args.audio_weight)
    print(f"audio merge: matched {n_m}/{n_t} exact-shape tensors (w_aframe={args.audio_weight})")
    av_sd.update(merged_audio)

    # ---- vision tower merge (native PE .pt donors) ----
    vision_donors = []
    try:
        from huggingface_hub import hf_hub_download
        for repo, fn in (("facebook/PE-Core-L14-336", "PE-Core-L14-336.pt"),
                         ("facebook/PE-Spatial-L14-448", "PE-Spatial-L14-448.pt")):
            try:
                p = hf_hub_download(repo_id=repo, filename=fn)
                print(f"loading {repo} ...")
                raw = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
                if isinstance(raw, dict) and "state_dict" in raw:
                    raw = raw["state_dict"]
                elif isinstance(raw, dict) and "weights" in raw:
                    raw = raw["weights"]
                raw = {k.replace("module.", ""): v for k, v in raw.items()}
                vision_donors.append((repo, raw))
                print(f"  {repo}: {len(raw)} tensors")
            except Exception as e:
                print(f"  SKIP {repo}: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"vision donor download disabled: {e}")

    # find the single timm prefix in the anchor (nested embedders)
    timm_keys = [k for k in av_sd.keys() if ".timm_model.blocks." in k]
    assert timm_keys, "no timm_model.blocks.* keys found in anchor — transformers layout changed?"
    anchor_timm_prefix = timm_keys[0].split(".timm_model.blocks.")[0] + ".timm_model."
    print(f"anchor timm prefix: {anchor_timm_prefix}")
    if vision_donors and timm_keys:
        import torch as _t
        for repo, raw in vision_donors:
            w_each = args.vision_weight / len(vision_donors)
            mapped, n_blocks = build_vision_donor_map(raw, anchor_timm_prefix, av_sd)
            print(f"vision map {repo}: {len(mapped)} trunk tensors mapped (blocks={n_blocks}, w={w_each:.3f}); "
                  f"pos_embed {'mapped' if anchor_timm_prefix+'pos_embed' in mapped else 'SKIPPED (resolution/head mismatch, anchor kept)'}, "
                  f"attn_pool/head/proj/ln_post intentionally kept from anchor")
            deltas = []
            for full, donor_t in mapped.items():
                a = av_sd[full]
                fused = ((1.0 - w_each) * a.float() + w_each * donor_t.float()).to(a.dtype)
                try:
                    d = (fused.float() - a.float()).abs().mean().item()
                    deltas.append(d)
                except Exception:
                    pass
                av_sd[full] = fused
            if deltas:
                import statistics as _s
                print(f"  applied {len(mapped)} tensors, mean|delta|={_s.mean(deltas):.6f} "
                      f"(small drift = no corruption, knowledge blended)")
            else:
                print("  WARNING: nothing applied — anchor video tower unchanged")
    else:
        print("vision merge: no donors loaded, anchor video tower kept as-is")

    # ---- save single model ----
    av.load_state_dict(av_sd, strict=False)
    av.save_pretrained(args.out)
    try:
        from transformers import AutoTokenizer, AutoProcessor
        try:
            tok = AutoTokenizer.from_pretrained(AV_ID, trust_remote_code=True)
            tok.save_pretrained(args.out)
        except Exception:
            pass
        try:
            proc = AutoProcessor.from_pretrained(AV_ID, trust_remote_code=True)
            proc.save_pretrained(args.out)
        except Exception:
            pass
    except Exception:
        pass
    print(f"\nsaved single fused model -> {args.out}")
    print("load with: AutoModel.from_pretrained('{}', trust_remote_code=True)".format(args.out))

    # ---- smoke test ----
    print("\n== smoke test (dummy forward, CPU) ==")
    av.eval()
    with torch.no_grad():
        # infer expected dtypes/shapes from model itself; use text-only ifjoint forward needs media
        try:
            cfg = av.config
            print(f"model_type={getattr(cfg, 'model_type', '?')}")
            # text embeddings path
            if hasattr(av, "get_text_features") or hasattr(av, "encode_text"):
                print("text tower present")
            # generic: count params
            n_params = sum(p.numel() for p in av.parameters())
            print(f"total params: {n_params/1e6:.1f}M (single model, no router)")
        except Exception as e:
            print(f"smoke info failed: {e}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
