import math
import numpy as np

M0 = np.uint64(0xD2511F53); M1 = np.uint64(0xCD9E8D57)
W0 = 0x9E3779B9; W1 = 0xBB67AE85; M32 = np.uint64(0xFFFFFFFF)


def philox(ctr, key, rounds=10):
    c = [np.uint32(x) for x in ctr]; k = [np.uint32(x) for x in key]
    for _ in range(rounds):
        p0 = M0 * np.uint64(int(c[0])); p1 = M1 * np.uint64(int(c[2]))
        c = [np.uint32(int(p1 >> np.uint64(32)) ^ int(c[1]) ^ int(k[0])),
             np.uint32(int(p1) & int(M32)),
             np.uint32(int(p0 >> np.uint64(32)) ^ int(c[3]) ^ int(k[1])),
             np.uint32(int(p0) & int(M32))]
        k = [np.uint32((int(k[0]) + W0) & 0xFFFFFFFF), np.uint32((int(k[1]) + W1) & 0xFFFFFFFF)]
    return [int(x) for x in c]


def uu(x): return (x + 1.0) * 2.3283064365386963e-10


def normal_at(k, seed=1895):
    r = philox([0, 0, k, 0], [seed & 0xFFFFFFFF, seed >> 32])
    u1, u2 = uu(r[0]), uu(r[1])
    return math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)


n0 = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_000.npy").reshape(-1)
n1 = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_001.npy").reshape(-1)
g0 = np.array([normal_at(k) for k in range(n0.size)])
g1 = np.array([normal_at(n0.size + k) for k in range(n1.size)])
print("draw0 max_abs", float(np.abs(g0 - n0).max()), "n", n0.size)
print("draw1 max_abs", float(np.abs(g1 - n1).max()), "n", n1.size)
print("draw1 got ", np.round(g1[:5], 6))
print("draw1 want", np.round(n1[:5], 6))
# verify a few more draws: cumulative offset
off = n0.size + n1.size
for i in (2, 3, 4):
    a = np.load(f"D:/Ming-3.0-tts/wgpu/goldens/m5/noise_{i:03d}.npy").reshape(-1)
    g = np.array([normal_at(off + k) for k in range(a.size)])
    print(f"draw{i} max_abs", float(np.abs(g - a).max()), "n", a.size)
    off += a.size
