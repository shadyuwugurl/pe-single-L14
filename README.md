# pe-single-L14 — one fused Meta Perception Encoder, no router

Single-file joint embedding model (audio + image/video + text, 1024-d, 1.03B
params) fusing four Meta PE family members. Weights:
**[MC7ever/pe-single-L14](https://huggingface.co/MC7ever/pe-single-L14)** (frozen tag `a0.25_v0.0`).

## Recipe

Anchor `facebook/pe-av-base` (already a joint audio+video+text model) +
- audio tower ← 0.25 × `facebook/pe-a-frame-base` (exact-shape mapped, standalone tower only)
- video trunk ← 0.0 × Core-L14/Spatial-L14 (searched, hurt alignment; anchor vision kept)

Why L-scale: G14 is width-1536×50L, AV/A-Frame are width-1024 — cross-width
tensors can't be averaged. L/14 shares width/depth/patch across all four.

## Certification (all measured on real data, CPU)

- Bench: COCO retrieval slice (MMEB `MSCOCO_i2t` rows + val2014 images) and
  ESC-50 zero-shot slices (`ashraq/esc50` streaming).
- 9-build grid + 18-genome tournament evolution + full-slice validation.
- Result: statistical tie vs anchor (t2i R@1 0.713 vs 0.720, ESC 0.895 vs
  0.885) — fusion preserves quality; gains deferred to fine-tuning.
- Negative results kept: naive mid-layer readout ≤0.06 vs 0.79 joint output;
  three voided runs documented (contamination, silent no-op loads, joint-path
  audio mismatch).

## Files

| script | purpose |
|---|---|
| `fuse_pe_single.py` | weight fusion (PE-native → timm key map + audio merge) |
| `eval_pe_bench.py` | COCO + ESC-50 bench harness |
| `search_pe_weights.py` | 3×3 grid over blend weights |
| `evo_fuse.py` | tournament-GA evolution (`copy_` by param name — `load_state_dict` silently no-ops on this model's dual-prefix layout) |
| `layer_sweep.py` | per-depth retrieval probe via `output_hidden_states` |
| `tune_pe.py` | GPU alignment fine-tune (contrastive, heads-first; RunPod-ready) |

`results/pe_bench/` holds all score JSONs. `bench_data/coco150.json` is the
slice manifest; images download from `images.cocodataset.org/val2014/`
(see `eval_pe_bench.py` slice builder), ESC-50 streams from HF.

## Fine-tune

```bash
# single GPU (RunPod):
python3 tune_pe.py --model MC7ever/pe-single-L14 --out ./pe-single-L14-tuned \
  --n-coco-train 4000 --n-esc-train 1600 --epochs 2 --lr 1e-5 --batch 32
```

Freezes both towers, trains joint contrastive heads + logit scales with
InfoNCE over (video,text) + (audio,text) batches. Then re-run
`eval_pe_bench.py` at full slices and promote the winner.
