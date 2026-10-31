# Reverse-engineer torch CUDA randn (Philox4x32-10 + curand Box-Muller) against
# the recorded first 64 draws for seed 1895.
import itertools, math
import numpy as np

TARGET = np.load("D:/Ming-3.0-tts/wgpu/goldens/m5/noise_000.npy").reshape(-1)
print("target[:6]", np.round(TARGET[:6], 6))

M0 = np.uint64(0xD2511F53)
M1 = np.uint64(0xCD9E8D57)
W0 = np.uint32(0x9E3779B9)
W1 = np.uint32(0xBB67AE85)
M32 = np.uint64(0xFFFFFFFF)


def philox4x32_10(ctr, key, rounds=10):
    c = [np.uint32(x) for x in ctr]
    k = [np.uint32(x) for x in key]
    for _ in range(rounds):
        p0 = M0 * np.uint64(c[0] & 0xFFFFFFFF)
        p1 = M1 * np.uint64(c[2] & 0xFFFFFFFF)
        nc = [
            np.uint32((p1 >> np.uint64(32)) ^ np.uint64(c[1]) ^ np.uint64(k[0])),
            np.uint32(p1 & M32),
            np.uint32((p0 >> np.uint64(32)) ^ np.uint64(c[3]) ^ np.uint64(k[1])),
            np.uint32(p0 & M32),
        ]
        c = nc
        k = [np.uint32((int(k[0]) + int(W0)) & 0xFFFFFFFF),
             np.uint32((int(k[1]) + int(W1)) & 0xFFFFFFFF)]
    return [int(x) for x in c]


def u_curand(x):          # curand_uniform: (0,1]
    return (x + 1.0) * 2.3283064365386963e-10


def u_plain(x):
    return x * 2.3283064365386963e-10


def bm(u1, u2, first="sin"):
    r = math.sqrt(-2.0 * math.log(u1))
    th = 2.0 * math.pi * u2
    a, b = r * math.cos(th), r * math.sin(th)
    return (a, b) if first == "cos" else (b, a)


for umap, first, scheme in itertools.product(
        ["curand", "plain"], ["cos", "sin"], ["idx", "stream"]):
    name = umap
    vals = []
    if scheme == "idx":   # one stream per element: ctr=(i,0,0,0)
        for i in range(TARGET.size):
            r = philox4x32_10([i, 0, 0, 0], [1895, 0])
            f = u_curand if umap == "curand" else u_plain
            vals.append(bm(f(r[0]), f(r[1]), first)[0])
    else:                 # single stream, 4 uniforms per call
        got = []
        i = 0
        while len(got) < TARGET.size:
            r = philox4x32_10([i, 0, 0, 0], [1895, 0])
            f = u_curand if umap == "curand" else u_plain
            a, b = bm(f(r[0]), f(r[1]), first)
            c, d = bm(f(r[2]), f(r[3]), first)
            got += [a, b, c, d]
            i += 1
        vals = got[:TARGET.size]
    v = np.array(vals, dtype=np.float64)
    err = float(np.abs(v - TARGET).max())
    print(f"{name:7s} {first} {scheme:7s} max_abs={err:.6f} first3={np.round(v[:3],6)}")
    if err < 1e-4:
        print("*** MATCH", name, first, scheme)
