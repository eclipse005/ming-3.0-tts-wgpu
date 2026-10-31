import numpy as np, math
G = "D:/Ming-3.0-tts/wgpu/goldens/"
q_pre = np.load(G + "b0_q.npy").astype(np.float64).reshape(3, 16, 64)   # our pre-rope layout [seq,head,dim]
q_ref = np.load(G + "b0_q_rope.npy").astype(np.float64)                  # torch [head,seq,dim]
q_ref = q_ref.transpose(1, 0, 2)                                         # -> [seq,head,dim]

theta = 1e4
def rope_interleaved(x, pos):
    out = np.empty_like(x)
    for i in range(32):
        ang = pos * (theta ** (-2.0 * i / 64.0))
        c, s = math.cos(ang), math.sin(ang)
        a, b = x[..., 2 * i], x[..., 2 * i + 1]
        out[..., 2 * i] = a * c - b * s
        out[..., 2 * i + 1] = a * s + b * c
    return out

def rope_half(x, pos):
    out = np.empty_like(x)
    for i in range(32):
        ang = pos * (theta ** (-2.0 * i / 64.0))
        c, s = math.cos(ang), math.sin(ang)
        a, b = x[..., i], x[..., i + 32]
        out[..., i] = a * c - b * s
        out[..., i + 32] = a * s + b * c
    return out

for name, fn in (("interleaved", rope_interleaved), ("rotate-half", rope_half)):
    tot = []
    for pos in range(3):
        ours = fn(q_pre[pos], pos)
        ref = q_ref[pos]
        tot.append(np.linalg.norm(ours - ref) / np.linalg.norm(ref))
    print(f"{name:12s}: pos0={tot[0]:.2e} pos1={tot[1]:.2e} pos2={tot[2]:.2e}")

# per-dim error profile of the interleaved variant at pos=2
ours = rope_interleaved(q_pre[2], 2)
err = np.abs(ours - q_ref[2]).max(axis=(0, 1))
print("\ninterleaved 逐维最大误差（pos=2）:")
print("  d 0..15 :", np.round(err[:16], 5))
print("  d 16..31:", np.round(err[16:32], 5))
print("  d 32..47:", np.round(err[32:48], 5))
print("  d 48..63:", np.round(err[48:], 5))
