# ROADMAP — pe-single-L14 program

## Done
- [x] 4-way weight fusion into one file (no router): PE-AV + A-Frame + Core-L14 + Spatial-L14
- [x] Certification: 9-build grid + 18-genome evolution + full-slice validation + 15-metric suite
- [x] Winner `[audio 0.25, vision ~0]` frozen; recipes agree independently
- [x] Layer-depth probe: negative (naive mid readout ≤0.06 vs 0.79 joint output)
- [x] v1 alignment tune, local MPS: i2t 0.013→0.727, t2i 0.713→0.827 (HF v3)
- [x] VLM graft (projector → LFM2.5-2.6B): coarse captions, published
- [x] MPS fp32 pipeline; autocast-bf16 proven safe (cos 0.9999); fp16/bf16-full rejected with evidence
- [x] Disk ESC cache (mmap) shared by tune/bench/suite; per-model suite checkpoints

## In flight
- [ ] **v2 audio-mixed tune** (COCO 4000 + ESC folds 1-4, from v3, step-50 ckpt): target ESC lift with t2i held

## Next, in order
1. **v2 eval + publish** (full bench + suite; HF v4 if it moves ESC)
2. **SVD-LoRA pilots** (teacher → student, rank 64–128, MPS evenings):
   - Core-G14-448 → video tower (same family, open first)
   - pe-a-frame-large → audio tower
   - EVIE-4.5B → joint space (documents; closest objective match)
   - TimesFM-3.0 (Chronos-2 backup) → audio tower as 1-D TS encoder
   - Depth-Anything-V2-Large → video backbone; SAM 2.1 → backbone; Cosmos3-Nano → video dynamics (behind InternVideo2); TRELLIS.2 XOR Hunyuan3D (3D/mesh, pick one); JanusFlow-1.3B (cheap experiments); LFM2.5-Audio-1.5B (audio understanding); Qwen2.5-Omni-7B (reference + yardstick); Emu3.5, Lance (mid-priority unified)
3. **Graft sharpening**: prefix 4→16 → LLM LoRA → audio prefix training → more data
4. **Time-series probe**: MVFMV + TimesX contexts through audio tower, zero-shot first
5. **Dialect eval**: Common Voice en-AU/en-GB + TIMIT-style zero-shot classification
6. **Deploy utilities as-is**: Phonon-2 (MLX ASR), Cosmos3-Edge (on-device video), diarization/speaker-ID models, GR00T (robotics track only)

## Parked (no axis yet)
Music models (MiniMax-Music3, YuE2, magenta-realtime-2), RE-USE, smart-turn-v3,
SOMA-X (until pose track opens), mesh weights (verify at build time),
obstacles (composite problem, post-tune).

## Infra rules learned the hard way
- `load_state_dict` silently no-ops on PeAudioVideoModel dual-prefix layout → blend via `copy_` by param name.
- Blend standalone towers only; joint-path copies have same shapes, different semantics.
- Trainable filter must exclude tower norms or backward retains the full graph (swap death).
- Streaming datasets need disk-cache + mmap; per-model eval checkpoints; `nohup` + `caffeinate -dims`, never bare pipes.
- Weights live on HF, never GitHub. Secrets live in pod env, never chat.
