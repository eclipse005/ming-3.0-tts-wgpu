# Reference per-step trace for C6 (single fragment, ids verified identical).
# Where does our loop leave the reference trajectory?
import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
import numpy as np, torch
from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens/m5_params"
TEXT = "今天晚上吃火锅还是烧烤呢？"

torch.manual_seed(1895)
tts = MingTTS("Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model

steps = []
orig = m.flowloss.sample
def spy(z, lh, cfg=1.0, patch_size=1, sigma=0.25, temperature=0, steps_n=10):
    out = orig(z, lh, cfg, patch_size, sigma=sigma, temperature=temperature, steps=steps_n)
    steps.append((z[0, 0].detach().float().cpu().numpy().copy(),
                  out[0].detach().float().cpu().numpy().copy()))
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
    wav = tts.synthesize(task="EMOTION", text=TEXT, controls={"emotion": "惊讶"},
                         output_path=None)

np.save(OUT + "/c6_embeds.npy", np.ascontiguousarray(emb["emb"]))
np.save(OUT + "/c6_z_steps.npy", np.ascontiguousarray(np.stack([s[0] for s in steps])))
np.save(OUT + "/c6_latents.npy", np.ascontiguousarray(np.stack([s[1] for s in steps])))
print("steps", len(steps), "embeds", emb["emb"].shape, "wav", len(wav[0]) if wav.dim()>1 else len(wav))
