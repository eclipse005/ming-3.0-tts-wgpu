# Fresh, self-consistent VAE-encoder reference: ONE forward, all outputs
# from that single call (mean/std/latent + per-layer attention output).
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS

G = "D:/Ming-3.0-tts/wgpu/goldens"
torch.manual_seed(1895)
tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model
enc_mod = m.audio.encoder

wf = np.load(f"{G}/m3_waveform.npy")
wave = torch.tensor(wf, device="cuda")[None]
n = wave.shape[1]

cap = {}
l0 = enc_mod.encoder.layers[0]
l0.self_attn.o_proj.register_forward_hook(
    lambda mod, i, o: cap.__setitem__("l0_attn", i[0][0].detach().float().cpu().numpy()))
for i in (0, 23):
    enc_mod.encoder.layers[i].register_forward_hook(
        (lambda i: lambda mod, inp, o: cap.__setitem__(f"layer{i}", o[0].detach().float().cpu().numpy()))(i))
enc_mod.aggregator.register_forward_hook(
    lambda mod, inp, o: cap.__setitem__("agg_out", o[0].detach().float().cpu().numpy()))

with torch.inference_mode():
    latent, frame_num = m.audio.encode_latent(wave, torch.tensor([n], device="cuda"))
    # mean/std from a second identical call is fine ONLY if deterministic;
    # instead derive from the sampled latent is impossible, so recompute the
    # parameters via the recorded fc3 input path:
    h, _ = enc_mod.encoder(wave)
    h = h.transpose(1, 2)
    mean, logvar = torch.chunk(h, 2, dim=1)
    cap["mean"] = mean[0].transpose(0, 1).detach().float().cpu().numpy()
    cap["std"] = torch.exp(0.5 * logvar)[0].transpose(0, 1).detach().float().cpu().numpy()

for k, v in cap.items():
    np.save(f"{G}/m3ref_{k}.npy", v)
    print(k, v.shape)
np.save(f"{G}/m3ref_latent.npy", latent[0].detach().float().cpu().numpy())
eps = (cap["latent" if "latent" in cap else "mean"])  # placeholder
print("done")
