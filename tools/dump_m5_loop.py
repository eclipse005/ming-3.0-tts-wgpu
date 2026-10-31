# Dump the M5 decode-loop goldens for the A1_bare case (TTS, no prompt, no
# speaker embedding).  Captures the token layout, the initial embeddings, every
# decode step's hidden state / stop probability / sampled latent, the exact
# torch.randn draws (so the Rust port can replay the identical RNG stream), and
# the final waveform.
import json
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS

G = "D:/Ming-3.0-tts/wgpu/goldens/m5"
import os
os.makedirs(G, exist_ok=True)

TEXT = "今天天气真不错，我们一起去公园散步吧。"

torch.manual_seed(1895)
tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model

cap = {"steps": [], "noise": []}

# --- RNG spy: torch.randn draws, in order -------------------------------------
orig_randn = torch.randn


def spy_randn(*a, **kw):
    t = orig_randn(*a, **kw)
    cap["noise"].append(t.detach().float().cpu().numpy())
    return t


torch.randn = spy_randn

# --- input assembly -----------------------------------------------------------
orig_prepare = m.prepare_input_embed


def spy_prepare(**kw):
    ids, emb = orig_prepare(**kw)
    cap["input_ids"] = ids[0].detach().cpu().numpy().astype(np.int64)
    cap["inputs_embeds"] = emb[0].detach().float().cpu().numpy()
    return ids, emb


m.prepare_input_embed = spy_prepare

# --- per-step forward ---------------------------------------------------------
orig_sample = m.flowloss.sample


def spy_flow(z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
             temperature=0, steps=10):
    cap["cfg"] = cfg
    cap["patch_size"] = patch_size
    cap["sigma"] = sigma
    cap["temperature"] = temperature
    cap["steps_n"] = steps
    cap["lh_before"] = latent_history[0].detach().float().cpu().numpy().copy()
    out = orig_sample(z, latent_history, cfg, patch_size, sigma=sigma,
                      temperature=temperature, steps=steps)
    cap["steps"].append({
        "z": z[0, 0].detach().float().cpu().numpy(),
        "latent": out[0].detach().float().cpu().numpy(),
    })
    return out


m.flowloss.sample = spy_flow

orig_decode = m.audio.decode


def spy_decode(latent, **kw):
    cap["decode_latent"] = latent[0].detach().float().cpu().numpy()
    out = orig_decode(latent, **kw)
    cap["waveform"] = out[0][0, 0].detach().float().cpu().numpy()
    return out


m.audio.decode = spy_decode

with torch.inference_mode():
    wav = tts.synthesize(task="TTS", text=TEXT, output_path=None)

torch.randn = orig_randn

print("input_ids", cap["input_ids"].shape, "embeds", cap["inputs_embeds"].shape)
print("steps", len(cap["steps"]), "randn draws", len(cap["noise"]),
      [n.shape for n in cap["noise"][:3]])
print("decode latent", cap["decode_latent"].shape, "waveform", cap["waveform"].shape)

np.save(f"{G}/input_ids.npy", np.ascontiguousarray(cap["input_ids"]))
np.save(f"{G}/inputs_embeds.npy", np.ascontiguousarray(cap["inputs_embeds"]))
np.save(f"{G}/decode_latent.npy", np.ascontiguousarray(cap["decode_latent"]))
np.save(f"{G}/waveform.npy", np.ascontiguousarray(cap["waveform"]))
np.save(f"{G}/z_steps.npy", np.ascontiguousarray(np.stack([s["z"] for s in cap["steps"]])))
np.save(f"{G}/latent_steps.npy", np.ascontiguousarray(np.stack([s["latent"] for s in cap["steps"]])))
np.save(f"{G}/lh_first.npy", np.ascontiguousarray(cap["lh_before"]))
for i, n in enumerate(cap["noise"]):
    np.save(f"{G}/noise_{i:03d}.npy", np.ascontiguousarray(n))
meta = {
    "n_steps": len(cap["steps"]),
    "n_noise": len(cap["noise"]),
    "noise_shapes": [list(n.shape) for n in cap["noise"]],
    "cfg": cap["cfg"], "patch_size": cap["patch_size"],
    "sigma": cap["sigma"], "temperature": cap["temperature"],
    "steps_n": cap["steps_n"],
    "text": TEXT,
}
with open(f"{G}/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print("done", meta["n_steps"], "steps")
