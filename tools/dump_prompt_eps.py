import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
import numpy as np, torch
from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens/m5_params"
PROMPT_WAV = "data/voice_presets/voice_audio/lingguang.wav"

orig = torch.randn
draws = []
def spy(*a, **kw):
    t = orig(*a, **kw); draws.append(t.detach().float().cpu().numpy()); return t

torch.manual_seed(1895)
tts = MingTTS("Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model
waveform, spk = tts._load_prompt([PROMPT_WAV])

# capture mean/std of the posterior
cap = {}
orig_enc = m.audio.encoder
def hook(mod, inp, out):
    cap["fc3"] = out[0].detach().float().cpu().numpy() if not isinstance(out, tuple) else None
m.audio.encoder.fc3.register_forward_hook(
    lambda mod, i, o: cap.__setitem__("fc3_out", o.detach().float().cpu().numpy()))

torch.randn = spy
latent = m.encode_prompt_latent(waveform)
torch.randn = orig

print("draws:", [(d.shape, d.size) for d in draws])
np.save(f"{OUT}/prompt_eps_raw.npy", np.ascontiguousarray(draws[0].reshape(-1)))
fc3 = cap["fc3_out"][0]                      # [T, 128]
mean = fc3[:, :64]
logvar = fc3[:, 64:]
std = np.exp(0.5 * logvar)
np.save(f"{OUT}/prompt_mean.npy", np.ascontiguousarray(mean))
np.save(f"{OUT}/prompt_std.npy", np.ascontiguousarray(std))
eps_py = draws[0].reshape(64, -1).T          # [T, 64]
print("fc3", fc3.shape, "eps", eps_py.shape)
np.save(f"{OUT}/prompt_eps_T64.npy", np.ascontiguousarray(eps_py))
recon = mean + std * eps_py
print("recon vs latent rel:", float(np.linalg.norm(recon - latent[0].float().cpu().numpy()) / np.linalg.norm(latent[0].float().cpu().numpy())))
