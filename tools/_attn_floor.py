# Is the attention's 500x amplification a defect or f32 cancellation?
import numpy as np, math
G = "D:/Ming-3.0-tts/wgpu/goldens/"
q = np.load(G + "b0_q_rope.npy").astype(np.float64)   # [16, 3, 64] head-major
k = np.load(G + "b0_k_rope.npy").astype(np.float64)
v = np.load(G + "b0_v.npy").astype(np.float32).reshape(3, 16, 64).transpose(1, 0, 2)
attn_ref = np.load(G + "b0_attn.npy").astype(np.float64)   # [3, 1024] (o_proj output)

qw = np.load(G + "b0_q.npy").astype(np.float64)            # pre-rope, [3,1024]
scale = 1.0 / math.sqrt(64)
scores = np.einsum("hsd,htd->hst", q, k) * scale
print("score 范围: min %.1f max %.1f   |q| ~ %.1f" % (scores.min(), scores.max(), np.abs(q).max()))
print("若用 f32 存 score，1 ulp ≈ %.2e" % (np.spacing(np.float32(scores.max()))))

def attn_out(qq):
    s = np.einsum("hsd,htd->hst", qq, k) * scale
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s); p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("hst,htd->hsd", p, v)          # [16,3,64]

base = attn_out(q)
# perturb q by exactly 1 ulp at f32, like a different summation order would
q_ulp = np.nextafter(q.astype(np.float32), np.float32(np.inf)).astype(np.float64)
pert = attn_out(q_ulp)
rel = np.linalg.norm(pert - base) / np.linalg.norm(base)
print("给 q 加 1 ulp 后，注意力输出相对变化 = %.3e" % rel)
print("实测我们 vs Python 的 attn 误差     = 1.098e-04")
print()
print("=> 若 1 ulp 就能造成 ~%.0e 的输出变化，说明 1.1e-4 是 f32 消去的物理下限。" % rel)
