"""Caption with the graft: PE encoder -> projector -> LFM2.5-2.6B.

Usage:
  python3 graft_generate.py --graft ./pe-graft-lfm --image bench_data/coco_img/<f>.jpg
  python3 graft_generate.py --graft ./pe-graft-lfm --audio clip.wav --sr 44100
"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graft", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = (args.device or ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu"))
    sys.path.insert(0, ".")
    from graft_train import Projector
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from transformers import AutoModelForCausalLM

    g = torch.load(f"{args.graft}/graft.pt", map_location="cpu",
                   weights_only=False)
    enc = AutoModel.from_pretrained(g["enc"], trust_remote_code=True,
                                    torch_dtype=torch.float32).to(dev).eval()
    eproc = AutoProcessor.from_pretrained("facebook/pe-av-base",
                                          trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(g["llm"], trust_remote_code=True,
                                               torch_dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(g["llm"], trust_remote_code=True)
    proj = Projector(1024, llm.config.hidden_size, g["k"]).to(dev)
    proj.load_state_dict(g["proj"])
    proj.eval()
    wte = llm.get_input_embeddings()

    with torch.no_grad():
        if args.image:
            img = np.array(Image.open(args.image).convert("RGB"))
            out = eproc(videos=[[img]], return_tensors="pt")
            pv = out["pixel_values_videos"].to(dev)
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
                        (1, iv.shape[-1]), dtype=torch.int32).to(dev))
            me = o.video_embeds
            prompt = args.prompt or "Describe the image: "
        else:
            import soundfile as sf
            wav, sr = sf.read(args.audio)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if sr != 48000:
                wav = np.interp(
                    np.linspace(0, len(wav), int(len(wav) * 48000 / sr)),
                    np.arange(len(wav)), wav).astype(np.float32)
            aw = eproc(audio=wav, sampling_rate=48000, return_tensors="pt")
            iv = aw["input_values"]
            if iv.dim() == 2:
                iv = iv.unsqueeze(1)
            iv = iv.to(dev)
            t = eproc.tokenizer(["a sound"], return_tensors="pt",
                                padding=True)
            fr = np.full((1, 336, 336, 3), 128, dtype=np.uint8)
            pv = eproc(videos=[fr], return_tensors="pt")[
                "pixel_values_videos"].to(dev)
            o = enc(input_values=iv,
                    padding_mask=torch.ones((1, iv.shape[-1]),
                                            dtype=torch.int32).to(dev),
                    input_ids=t["input_ids"].to(dev),
                    attention_mask=t.get("attention_mask").to(dev),
                    pixel_values_videos=pv)
            me = o.audio_embeds
            prompt = args.prompt or "Describe the sound: "
        pre = proj(me).to(wte.weight.dtype)
        pin = tok(prompt, return_tensors="pt")
        pe = wte(pin["input_ids"].to(dev))
        inputs_embeds = torch.cat([pre, pe], dim=1)
        attn = torch.ones((1, inputs_embeds.shape[1]),
                          dtype=torch.long).to(dev)
        out = llm.generate(inputs_embeds=inputs_embeds,
                           attention_mask=attn, max_new_tokens=args.max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id)
        print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    sys.exit(main())
