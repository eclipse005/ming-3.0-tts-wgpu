import torch, numpy as np, math
from safetensors.torch import load_file
sd = load_file("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/model.safetensors")
a = sd["flowloss.cfm.model.rotary_embed.inv_freq"].float().numpy()
b = sd["linear_proj_audio.rotary_embed.inv_freq"].float().numpy()
print("dit inv_freq == aggregator inv_freq:", np.array_equal(a, b))
print("0.75^i 与 checkpoint 的最大偏差: %.3e" % float(np.abs(a - 0.75 ** np.arange(32)).max()))

G = "D:/Ming-3.0-tts/wgpu/goldens/"
q_pre = np.load(G + "b0_q.npy").astype(np.float64).reshape(3, 16, 64)
q_ref = np.load(G + "b0_q_rope.npy").astype(np.float64).transpose(1, 0, 2)
k_pre = np.load(G + "b0_k.npy").astype(np.float64).reshape(3, 16, 64)
k_ref = np.load(G + "b0_k_rope.npy").astype(np.float64).transpose(1, 0, 2)

def rope(x, pos, inv):
    out = np.empty_like(x)
    for i in range(32):
        ang = pos * inv[i]
        c, s = math.cos(ang), math.sin(ang)
        A, B = x[..., 2 * i], x[..., 2 * i + 1]
        out[..., 2 * i] = A * c - B * s
        out[..., 2 * i + 1] = A * s + B * c
    return out

for tag, inv in (("theta=1e4 (ours)", np.array([1e4 ** (-2.0 * i / 64.0) for i in range(32)])),
                 ("checkpoint buffer", a.astype(np.float64))):
    eq = [np.linalg.norm(rope(q_pre[p], p, inv) - q_ref[p]) / np.linalg.norm(q_ref[p]) for p in range(3)]
    ek = [np.linalg.norm(rope(k_pre[p], p, inv) - k_ref[p]) / np.linalg.norm(k_ref[p]) for p in range(3)]
    print(f"{tag:20s} q: {eq[0]:.2e} {eq[1]:.2e} {eq[2]:.2e}   k: {ek[0]:.2e} {ek[1]:.2e} {ek[2]:.2e}")
