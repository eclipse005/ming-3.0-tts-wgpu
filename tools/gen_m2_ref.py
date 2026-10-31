# Recompute the flowloss reference latents from the recorded inputs, fresh and
# independent per step (flow_sample is memoryless given its inputs).
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS

G = "D:/Ming-3.0-tts/wgpu/goldens"
torch.manual_seed(1234)  # irrelevant: no RNG in cfm.sample at temperature 0

tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model
cfm = m.flowloss.cfm

z = np.load(f"{G}/m2_z.npy")
lh = np.load(f"{G}/m2_lh.npy")
noise = np.load(f"{G}/m2_noise.npy")
sde = np.load(f"{G}/m2_sde.npy")
cfg = np.load(f"{G}/m2_cfg.npy")

# direct-call probe: identical nominal inputs to euler-0, outside the sampler
probe_preds = []
orig_fwd0 = cfm.model.forward
probe_ts = []
def probe_rec(x, t, c, lh_, mask=None):
    out = orig_fwd0(x, t, c, lh_, mask)
    probe_ts.append(float(t.reshape(-1)[0]))
    probe_preds.append(out[0, 0].float().cpu().numpy())
    return out
cfm.model.forward = probe_rec
with torch.inference_mode():
    _direct = cfm.sample(
        noise=torch.tensor(noise[0], device="cuda")[None, None],
        c=torch.tensor(z[0], device="cuda")[None, None],
        latent_history=torch.tensor(lh[0], device="cuda")[None, None],
        sde_noise=torch.tensor(sde[0], device="cuda").reshape(10, 1, 1, 64),
        cfg_scale=float(cfg[0]),
    )
print("PROBE euler-0 t =", probe_ts[0], "pred[:3] =", probe_preds[0][:3])
cfm.model.forward = orig_fwd0

ys = []
pred_ref = []
null_ref = []
orig_fwd = cfm.model.forward
def fwd_rec(x, t, c, lh_, mask=None):
    out = orig_fwd(x, t, c, lh_, mask)
    pred_ref.append(out[0, 0].float().cpu().numpy())
    null_ref.append(out[1, 0].float().cpu().numpy())
    return out
cfm.model.forward = fwd_rec
with torch.inference_mode():
    for i in range(z.shape[0]):
        y = cfm.sample(
            noise=torch.tensor(noise[i], device="cuda")[None, None],
            c=torch.tensor(z[i], device="cuda")[None, None],
            latent_history=torch.tensor(lh[i], device="cuda")[None, None],
            sde_noise=torch.tensor(sde[i], device="cuda").reshape(10, 1, 1, 64),
            cfg_scale=float(cfg[i]),
        )
        ys.append(y[0, 0].float().cpu().numpy())
ys = np.stack(ys)
np.save(f"{G}/m2_y_ref.npy", ys)
np.save(f"{G}/m2_pred_ref.npy", np.stack(pred_ref))
np.save(f"{G}/m2_null_ref.npy", np.stack(null_ref))
print(f"pred_ref saved: {len(pred_ref)}; pred_ref[0][:3]={pred_ref[0][:3]}")
print(f"y_ref saved: {ys.shape} y_ref[0][:4]={ys[0][:4]}")
