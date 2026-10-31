# Dump one dit_forward call's intermediates via forward hooks (real batched path).
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

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

cap = {}
first = {"done": False}

def save(k, t):
    a = t.detach().float().cpu().numpy()
    np.save(f"{OUT}/m2bisect_{k}.npy", a)
    print(f"  {k}: {a.shape} absmax={np.abs(a).max():.4f}")

# first flowloss.sample inputs
orig_sample = m.flowloss.sample
def sample_spy(z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
               temperature=0, steps=10):
    if not first["done"]:
        save("z", z[0, 0])
        save("lh", latent_history[0, 0])
        save("cfg", torch.tensor(float(cfg)))
    noise = torch.randn((z.shape[0], patch_size, m.flowloss.z_channels),
                        device=z.device, dtype=z.dtype)
    sde = torch.randn((steps, *noise.shape), device=z.device, dtype=z.dtype)
    return m.flowloss.cfm.sample(noise=noise, c=z, latent_history=latent_history,
                                 sde_noise=sde, cfg_scale=cfg, patch_size=patch_size,
                                 sigma=sigma, temperature=temperature)
m.flowloss.sample = sample_spy

orig_fwd = dit.forward
def fwd_spy(x, t, c, latent_history, mask=None):
    if not first["done"]:
        save("x", x[0, 0])
        save("c", c[0, 0])
        save("lh", latent_history[0, 0])
        save("t_scalar", t.detach().float().reshape(-1)[:1])
        out = orig_fwd(x, t, c, latent_history, mask)
        first["done"] = True
        return out
    return orig_fwd(x, t, c, latent_history, mask)
dit.forward = fwd_spy

def hook_of(key):
    def hook(mod, inp, out):
        if not first["done"]:
            save(key, out[0])
    return hook

dit.t_embedder.time_mlp.register_forward_hook(hook_of("t_mlp"))
dit.c_embedder.cond_embedder.register_forward_hook(hook_of("c_emb"))
xs = {"n": 0}
def x_hook(mod, inp, out):
    if not first["done"]:
        save(f"x_emb_call{xs['n']}", out[0, 0])
        xs["n"] += 1
dit.x_embedder.register_forward_hook(x_hook)
dit.blocks[0].register_forward_hook(hook_of("block0"))
dit.blocks[7].register_forward_hook(hook_of("block7"))
dit.final_layer.register_forward_hook(hook_of("final"))

with torch.inference_mode():
    tts.synthesize(task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV,
                   prompt_text=PROMPT_TEXT, use_spk_emb=True, output_path=None)
print("done")
