# Per-step trace of the clone case (A3): hidden state, latent, stop decision.
import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
import numpy as np, torch
from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens/m5_params"
PROMPT_WAV = "data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"
TEXT = "今天天气真不错，我们一起去公园散步吧。"

torch.manual_seed(1895)
tts = MingTTS("Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model

steps = []
orig = m.flowloss.sample
def spy(z, lh, cfg=1.0, patch_size=1, sigma=0.25, temperature=0, steps_n=10):
    out = orig(z, lh, cfg, patch_size, sigma=sigma, temperature=temperature, steps=steps_n)
    steps.append((z[0, 0].detach().float().cpu().numpy().copy(),
                  out[0].detach().float().cpu().numpy().copy(),
                  lh[0].detach().float().cpu().numpy().copy()))
    return out
m.flowloss.sample = spy

emb = {}
orig_prep = m.prepare_input_embed
def spy_prep(**kw):
    ids, e = orig_prep(**kw)
    emb["emb"] = e[0].detach().float().cpu().numpy()
    return ids, e
m.prepare_input_embed = spy_prep

with torch.inference_mode():
    wav = tts.synthesize(task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV,
                         prompt_text=PROMPT_TEXT, use_spk_emb=True, output_path=None)

np.save(f"{OUT}/a3_embeds.npy", np.ascontiguousarray(emb["emb"]))
np.save(f"{OUT}/a3_z_steps.npy", np.ascontiguousarray(np.stack([s[0] for s in steps])))
np.save(f"{OUT}/a3_latents.npy", np.ascontiguousarray(np.stack([s[1] for s in steps])))
np.save(f"{OUT}/a3_lh0.npy", np.ascontiguousarray(steps[0][2]))
print("steps", len(steps), "latent shape", steps[0][1].shape, "lh0", steps[0][2].shape)
print("lh0", steps[0][2].ravel()[:4])
print("prompt latent last row", np.load(f"{OUT}/prompt_latent.npy")[-1][:4])
