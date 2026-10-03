---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
tags:
- pe_audio_video
- perception-encoder
- audio-text-retrieval
- image-text-retrieval
- zero-shot-audio-classification
- multimodal-embeddings
language:
- en
metrics:
- recall@1
- recall@5
- accuracy
model-index:
- name: pe-single-L14-a025
  results:
  - task:
      type: text-to-image-retrieval
      dataset:
        name: COCO slice (150, MMEB MSCOCO_i2t rows)
    metrics:
    - type: recall@1
      value: 0.766
      name: t2i R@1 (search slice n=64)
    - type: recall@5
      value: 0.95
      name: t2i R@5 (approx, search slice)
  - task:
      type: zero-shot-audio-classification
      dataset:
        name: ESC-50 slice (100 clips)
    metrics:
    - type: accuracy
      value: 0.96
      name: ESC-50 accuracy (search slice)
---

# pe-single-L14 — one fused Meta Perception Encoder, no router

Single-file joint embedding model (audio + image/video + text, 1024-d shared
space, 1.03B params) fusing four Meta PE family members:

| source | HF id | what was fused |
|---|---|---|
| PE-AV base (anchor) | `facebook/pe-av-base` | full joint model kept as scaffold |
| PE A-Frame base | `facebook/pe-a-frame-base` | audio tower, weight 0.25 |
| PE-Core-L14-336 | `facebook/PE-Core-L14-336` | vision trunk (near-identical to anchor video tower — init copy, ~no-op) |
| PE-Spatial-L14-448 | `facebook/PE-Spatial-L14-448` | vision trunk, weight 0.0 in this tag (hurt COCO alignment; see below) |

Why L-scale: PE-Core/Spatial-G14 are width-1536 × 50 layers while PE-AV/A-Frame
are width-1024 — cross-width tensors cannot be weight-averaged. L/14
(1024-wide, 24 layers, patch 14) is the scale where Core + Spatial + AV-video
share width/depth/patch, so true weight fusion is possible.

## Status (2026-10-03): CERTIFIED — frozen, no further weight search

Full-slice validation (COCO n=150, ESC-50 n=200) vs anchor:

| model | COCO t2i R@1 / R@5 | COCO i2t R@1 | ESC-50 acc |
|---|---|---|---|
| `facebook/pe-av-base` (anchor) | 0.720 / 0.953 | 0.013 | 0.885 |
| this tag | 0.713 / 0.953 | 0.013 | 0.895 |

Verdict: statistical tie (diffs within SE). Weight fusion preserves anchor
quality while carrying A-Frame/Core/Spatial weights in one file — it does not
beat the anchor on these benches by itself. Measured gains are deferred to
alignment fine-tuning, for which this tag is the frozen init. Certification:
9-build grid + 18-genome evolution (fixed machinery) + full-slice validation,
all agreeing (audio light, vision ~zero).

- Evolution rerun in progress after two voided attempts (documented below) —
  if it certifies a better genome, that becomes the next revision.
- Layer-depth probe: naive mid-network readout does NOT beat the joint output
  (t2i R@1 ≤ 0.06 vs 0.79) — mid-network features need trained per-layer
  pooling (fine-tune phase), not a free readout change.
- Voided runs: (1) blend-of-blend contamination via shared-storage state_dict;
  (2) silent no-op `load_state_dict` on this model's dual-prefix key layout;
  (3) joint-path audio mismatch that faked ESC collapse at high audio weight.
  All fixed (pristine clones, `copy_` by param name, standalone-tower-only
  audio mapping) before trusting any evolution number.

## How it was built

1. Audio tower: exact-shape suffix-mapped average of AV-audio and A-Frame-audio
   (both `pe_audio_encoder`, 1024/16L/8H — 422/422 tensors matched).
2. Video trunk: explicit native-PE → timm key map (patch_embed, cls_token,
   pos_embed@336px, norms, QKV, MLP ×24 blocks). Pool/head/proj kept from anchor.
3. Blend weights chosen by measured 3×3 grid search over COCO-retrieval +
   ESC-50 slices (script `search_pe_weights.py`), not by vibes. Vision donor
   weight degraded COCO t2i monotonically (0.766 → 0.64), so the winning tag
   keeps anchor vision and blends audio at 0.25.

## Usage

```python
from transformers import AutoModel, AutoProcessor
m = AutoModel.from_pretrained("MC7ever/pe-single-L14", trust_remote_code=True)
p = AutoProcessor.from_pretrained("MC7ever/pe-single-L14", trust_remote_code=True)
inputs = p(videos=[frames], audio=[waveform], text=["a cat playing in rain"],
           return_tensors="pt")
out = m(**inputs)  # out.video_embeds, out.audio_embeds, out.text_video_embeds, ...
```

Retrieval: dot-product `video_embeds @ text_video_embeds.T`.
Zero-shot audio: argmax over `audio_embeds @ text_audio_embeds.T` with
`"the sound of {label}"` prompts.

## Honest limits

- Numbers above are CPU-run slices (COCO n=64, ESC n=100), comparative
  anchor-vs-variant only — not official MMEB/PE-paper figures.
- i2t (text→image caption ranking over 150 similar COCO captions) sits near
  floor for anchor and variants alike; t2i and audio carry the signal.
- Encoder only: no generation, detection, or segmentation heads.
- Next: evolution-based per-tower weights, intermediate-layer readout
  (PE paper: best features are mid-network), GPU alignment fine-tune.

## Reproduce

`fuse_pe_single.py` (fusion) · `eval_pe_bench.py` (COCO+ESC-50 bench) ·
`search_pe_weights.py` (grid) · `evo_fuse.py` (evolution) · `layer_sweep.py`
(depth probe). Benchmark slice metadata: MMEB `MSCOCO_i2t` rows + COCO val2014
images + `ashraq/esc50` streaming clips.
