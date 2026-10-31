# Dump VAE-decoder goldens for M4: latent -> fc1 -> x4 upsample -> 24-layer
# causal stack -> ISTFT head -> waveform.  Injects the M3 latent so the encoder
# is not involved; every capture comes from ONE real decode() call (hooks only,
# no hand-rolled forward path).
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS

G = "D:/Ming-3.0-tts/wgpu/goldens"
TEXT = "今天天气真不错，我们一起去公园散步吧。"
PROMPT_WAV = "D:/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

torch.manual_seed(1895)
tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model
dec = m.audio.decoder

latent = torch.from_numpy(np.load(f"{G}/m3_latent.npy")).float().unsqueeze(0).cuda()
print("latent", tuple(latent.shape))

cap = {}


def hook(name, squeeze=True):
    def f(mod, inp, out):
        t = out
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        elif isinstance(t, tuple):
            t = t[0]
        t = t.detach().float().cpu().numpy()
        if t.ndim == 3:
            t = t[0]
        cap[name] = t
    return f


hooks = [
    dec.fc1.register_forward_hook(hook("fc1_out")),
    dec.upsampling.upsampler.register_forward_hook(hook("up_out")),
    dec.decoder.register_forward_hook(hook("dec_out")),
    dec.head.out.register_forward_hook(hook("head_lin")),
]
for i in (0, 1, 2, 8, 23):
    hooks.append(dec.decoder.layers[i].register_forward_hook(hook(f"d{i}")))

with torch.inference_mode():
    wav, _state, _pkv = m.audio.decode(latent)

for h in hooks:
    h.remove()

cap["latent"] = latent[0].detach().float().cpu().numpy()
cap["up_out"] = np.ascontiguousarray(cap["up_out"].T)   # [C, T] -> [T, C]
cap["waveform"] = wav[0, 0].detach().float().cpu().numpy()

for k, v in cap.items():
    print(f"  {k}: {v.shape}")
    np.save(f"{G}/m4_{k}.npy", v)
print("done")
