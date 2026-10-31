# Dump every sub-operator of the DiT's FIRST block. Two traps: the sampling loop
# is CUDA graphed (so the eager path must be forced by calling cfm.sample
# directly), and the aggregator has its own rotary embedding (so captures must be
# windowed to the first dit.forward call).
import sys
sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch
import fm.modules as fmmod

from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens"
TEXT = "今天天气真不错，我们一起去公园散步吧。"
PROMPT_WAV = "D:/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

torch.manual_seed(1895)
tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model
dit = m.flowloss.cfm.model
w = {"capturing": False, "done": False, "rope": 0}
cap = {}

def save(k, t):
    a = np.ascontiguousarray(t.detach().float().cpu().numpy())
    cap[k] = a
    print("  %-10s %-16s absmax=%.4g" % (k, str(a.shape), float(np.abs(a).max())), flush=True)

orig_sample = m.flowloss.sample
def sample_spy(z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
               temperature=0, steps=10):
    noise = torch.randn((z.shape[0], patch_size, m.flowloss.z_channels),
                        device=z.device, dtype=z.dtype)
    sde = torch.randn((steps, *noise.shape), device=z.device, dtype=z.dtype)
    return m.flowloss.cfm.sample(noise=noise, c=z, latent_history=latent_history,
                                 sde_noise=sde, cfg_scale=cfg, patch_size=patch_size,
                                 sigma=sigma, temperature=temperature)
m.flowloss.sample = sample_spy

orig_fwd = dit.forward
def fwd_spy(x, t, c, latent_history, mask=None):
    opening = not w["done"]
    if opening:
        w["capturing"] = True
        w["rope"] = 0
        save("in", x[0])
    out = orig_fwd(x, t, c, latent_history, mask)
    if opening:
        w["capturing"] = False
        w["done"] = True
    return out
dit.forward = fwd_spy

b0 = dit.blocks[0]
for name, mod in (("norm1", b0.norm1), ("q", b0.attn.to_q), ("k", b0.attn.to_k),
                  ("v", b0.attn.to_v), ("attn", b0.attn), ("norm2", b0.norm2),
                  ("ff_pre", b0.mlp.ff[0][0]), ("gelu", b0.mlp.ff[0][1]),
                  ("ff_out", b0.mlp.ff[2]), ("out", b0)):
    mod.register_forward_hook(
        (lambda k: lambda mod, i, o: save(k, o[0] if torch.is_tensor(o) else o)
         if w["capturing"] else None)(name))

orig_rope = fmmod.apply_rotary_pos_emb
def rope_spy(t, freqs, scale=1.0):
    out = orig_rope(t, freqs, scale)
    if w["capturing"] and w["rope"] < 2:
        save("q_rope" if w["rope"] == 0 else "k_rope", out[0])
    w["rope"] += 1
    return out
fmmod.apply_rotary_pos_emb = rope_spy

with torch.inference_mode():
    tts.synthesize(task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV,
                   prompt_text=PROMPT_TEXT, use_spk_emb=True, output_path=None)
fmmod.apply_rotary_pos_emb = orig_rope

for k, v in cap.items():
    np.save(OUT + "/b0_" + k + ".npy", v)
print("saved", len(cap), "arrays")