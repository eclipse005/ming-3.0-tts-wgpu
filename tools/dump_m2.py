# Dump flowloss.sample goldens for M2 parity: per decode step, record
# (z_diff, latent_history, cfg) inputs, the exact noise/sde_noise draws, and
# the sampled latent output. RNG order is preserved (the spy re-implements
# FlowLoss.sample's two randn calls in the same order, then delegates).
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

rec = []

def sample_spy(z, latent_history, cfg=1.0, patch_size=1, sigma=0.25,
               temperature=0, steps=10):
    noise = torch.randn((z.shape[0], patch_size, m.flowloss.z_channels),
                        device=z.device, dtype=z.dtype)
    sde_noise = torch.randn((steps, *noise.shape), device=z.device, dtype=z.dtype)
    y = m.flowloss.cfm.sample(
        noise=noise, c=z, latent_history=latent_history, sde_noise=sde_noise,
        cfg_scale=cfg, patch_size=patch_size, sigma=sigma, temperature=temperature)
    rec.append(dict(
        z=z[0, 0].float().cpu().numpy(),
        lh=latent_history[0, 0].float().cpu().numpy(),
        noise=noise[0, 0].float().cpu().numpy(),
        sde=sde_noise[:, 0, 0, :].float().cpu().numpy(),
        y=y[0, 0].float().cpu().numpy(),
        cfg=float(cfg),
    ))
    return y

m.flowloss.sample = sample_spy

euler_x, pred_r, null_r, t_r = [], [], [], []
orig_dit_fwd = m.flowloss.cfm.model.forward
def dit_fwd_spy(x, t, c, latent_history, mask=None):
    euler_x.append(x[0, 0].float().cpu().numpy())
    t_r.append(float(t.reshape(-1)[0]))
    out = orig_dit_fwd(x, t, c, latent_history, mask)
    if len(euler_x) == 1:
        with torch.inference_mode():
            re = orig_dit_fwd(x.clone(), t.clone(), c.clone(), latent_history.clone(), mask)
        print("INPROC recompute diff:", (re[0, 0] - out[0, 0]).abs().max().item())
        print("INPROC out00:", out[0, 0, :3].tolist())
    pred_r.append(out[0, 0].float().cpu().numpy())
    null_r.append(out[1, 0].float().cpu().numpy())
    return out
m.flowloss.cfm.model.forward = dit_fwd_spy

with torch.inference_mode():
    tts.synthesize(task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV,
                   prompt_text=PROMPT_TEXT, use_spk_emb=True, output_path=None)

print(f"steps recorded: {len(rec)}  z_channels={m.flowloss.z_channels}")
np.save(f"{OUT}/m2_z.npy", np.stack([r["z"] for r in rec]))
np.save(f"{OUT}/m2_lh.npy", np.stack([r["lh"] for r in rec]))
np.save(f"{OUT}/m2_noise.npy", np.stack([r["noise"] for r in rec]))
np.save(f"{OUT}/m2_sde.npy", np.stack([r["sde"] for r in rec]))
np.save(f"{OUT}/m2_y.npy", np.stack([r["y"] for r in rec]))
np.save(f"{OUT}/m2_cfg.npy", np.array([r["cfg"] for r in rec], dtype=np.float32))
np.save(f"{OUT}/m2_euler_x.npy", np.stack(euler_x))
np.save(f"{OUT}/m2_pred.npy", np.stack(pred_r))
np.save(f"{OUT}/m2_null.npy", np.stack(null_r))
np.save(f"{OUT}/m2_t.npy", np.array(t_r, dtype=np.float32))
print("t[0:12] =", t_r[:12])
print(f"saved {len(rec)} steps, {len(euler_x)} euler calls -> {OUT}/m2_*.npy  (y[0][:4]={rec[0]['y'][:4]})")

from safetensors import safe_open
with safe_open("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/model.safetensors", framework="numpy") as f:
    keys = list(f.keys())
    for pat in ("flowloss.cfm.model.", "linear_proj_audio."):
        ks = sorted(k for k in keys if k.startswith(pat))
        print(f"--- {pat} ({len(ks)} keys)")
        for k in ks[:10]:
            print("  ", k)
        print("   ...")
print("done")
