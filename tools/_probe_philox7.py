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


def sin_bm(ctr, seed=1895):
    r = philox(ctr, [seed & 0xFFFFFFFF, seed >> 32])
    u1, u2 = uu(r[0]), uu(r[1])
    return math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)


n1 = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_001.npy").reshape(-1)
print("want n1[:5]", np.round(n1[:5], 6))
for off in list(range(0, 6)) + [32, 64, 128, 256, 512, 576, 640]:
    v = [sin_bm([off, 0, k, 0]) for k in range(5)]
    ok = abs(v[0] - n1[0]) < 1e-5
    print(f"off={off:4d} ctr=(off,0,k,0): {np.round(v,6)} {'<== MATCH' if ok else ''}")
