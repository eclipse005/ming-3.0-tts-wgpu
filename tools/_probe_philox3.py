import math
import numpy as np

TARGET = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_000.npy").reshape(-1)
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
def quartet(ctr, key=[1895, 0]):
    r = philox(list(ctr), key)
    def bm(x, y):
        u1, u2 = uu(x), uu(y)
        rr = math.sqrt(-2.0 * math.log(u1)); th = 2.0 * math.pi * u2
        return rr * math.sin(th), rr * math.cos(th)
    a, b = bm(r[0], r[1]); c, d = bm(r[2], r[3])
    return [a, b, c, d]


want = TARGET[:4]
print("want", np.round(want, 6))
hits = []
for n in range(0, 300):
    for slot, val in enumerate(quartet([n, 0, 0, 0])):
        if abs(val - want[1]) < 1e-5:
            hits.append(("ctr.x", n, slot, round(val, 6)))
print("matches for target[1]:", hits[:8])
# where does target[2] come from?
hits2 = []
for n in range(0, 300):
    for slot, val in enumerate(quartet([n, 0, 0, 0])):
        if abs(val - want[2]) < 1e-5:
            hits2.append(("ctr.x", n, slot, round(val, 6)))
print("matches for target[2]:", hits2[:8])
# try element index in ctr.y / ctr.w
for field in (1, 2, 3):
    hits3 = []
    for n in range(0, 8):
        c = [0, 0, 0, 0]; c[field] = n
        for slot, val in enumerate(quartet(c)):
            if abs(val - want[1]) < 1e-5:
                hits3.append((field, n, slot))
    if hits3: print("ctr field", field, "hits", hits3)
