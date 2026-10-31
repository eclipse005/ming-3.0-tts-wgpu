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


C2P32 = np.float32(2.3283064e-10)
C2P33 = np.float32(1.1641532e-10)


def uniforms32(ctr, seed=1895):
    r = philox(ctr, [seed & 0xFFFFFFFF, seed >> 32])
    return np.float32(np.float32(r[0]) * C2P32 + C2P33), np.float32(np.float32(r[1]) * C2P32 + C2P33)


def normal_f32(draw, k, seed=1895):
    u1, u2 = uniforms32([draw, 0, k, 0], seed)
    r = np.sqrt(np.float32(-2.0) * np.log(u1))
    th = np.float32(2.0 * math.pi) * u2
    return np.float32(r * np.sin(th))


n = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_000.npy")
off = 0
worst = 0.0
for d in range(0, 6):
    a = np.load(f"D:/Ming-3.0-tts/wgpu/goldens/m5/noise_{d:03d}.npy").reshape(-1)
    got = np.array([normal_f32(d if d else 0, k) for k in range(a.size)], dtype=np.float32)
    # draw index d corresponds to the call counter; verify
    err = float(np.abs(got - a).max())
    worst = max(worst, err)
    print(f"draw{d}: n={a.size:4d} max_abs={err:.3e}")
print("worst", worst)
