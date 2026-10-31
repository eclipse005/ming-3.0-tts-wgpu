import math
import numpy as np

TARGET = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_000.npy").reshape(-1)
SAMPLES = TARGET.size
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


got = np.array([normal_at(k) for k in range(SAMPLES)])
err = np.abs(got - TARGET)
print("max_abs_error =", float(err.max()))
print("got[:6]", np.round(got[:6], 6))
print("want[:6]", np.round(TARGET[:6], 6))

# second draw of the same randn call sequence, to confirm the offset keeps counting
n2 = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_001.npy").reshape(-1)
got2 = np.array([normal_at(k) for k in range(SAMPLES)])   # placeholder: check whether the offset advances
print("noise_001 shape", n2.shape, "first", np.round(n2[:3], 6))
# hypothesis: the generator offset advances by 64 after the first draw
got_b = np.array([normal_at(SAMPLES + k) for k in range(SAMPLES)])
print("offset+64 max_abs =", float(np.abs(got_b - n2).max()))
