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


def quad(k, seed=1895):
    r = philox([0, 0, k, 0], [seed & 0xFFFFFFFF, seed >> 32])
    def bm(x, y):
        u1, u2 = uu(x), uu(y)
        rr = math.sqrt(-2.0 * math.log(u1)); th = 2.0 * math.pi * u2
        return rr * math.sin(th), rr * math.cos(th)
    a, b = bm(r[0], r[1]); c, d = bm(r[2], r[3])
    return [a, b, c, d]


n1 = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_001.npy").reshape(-1)
want = n1[0]
print("want n1[0] =", round(float(want), 6))
hits = []
for m in range(0, 4096):
    for slot, v in enumerate(quad(m)):
        if abs(v - want) < 1e-5:
            hits.append((m, slot))
print("hits:", hits[:10])
for m, slot in hits[:3]:
    pred = [quad(m + i)[slot] for i in range(5)]
    print(f"  m={m} slot={slot} -> {np.round(pred,6)} vs want {np.round(n1[:5],6)}")
