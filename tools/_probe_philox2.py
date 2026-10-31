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


def uu(x):  return (x + 1.0) * 2.3283064365386963e-10
def bm_pair(x, y):
    u1, u2 = uu(x), uu(y)
    r = math.sqrt(-2.0 * math.log(u1)); th = 2.0 * math.pi * u2
    return r * math.sin(th), r * math.cos(th)


def normals_from(ctr, key, n_calls):
    out = []
    for call in range(n_calls):
        c = list(ctr); c[0] = np.uint32((int(c[0]) + call) & 0xFFFFFFFF)
        r = philox(c, key)
        a, b = bm_pair(r[0], r[1])
        c2, d2 = bm_pair(r[2], r[3])
        out += [a, b, c2, d2]
    return out


print("target[:8]", np.round(TARGET[:8], 6))
# L1: element i -> its own stream, first normal of the pair
v = [normals_from([i, 0, 0, 0], [1895, 0], 1)[0] for i in range(8)]
print("L1 own-stream/first :", np.round(v, 6))
# L2: 2 elements per stream (a,b)
v = []
for i in range(4):
    pair = normals_from([i, 0, 0, 0], [1895, 0], 1)
    v += pair[:2]
print("L2 2-per-stream      :", np.round(v, 6))
# L3: 4 elements per stream (from one call: a,b,c,d)
v = []
for i in range(2):
    v += normals_from([i, 0, 0, 0], [1895, 0], 1)
print("L3 4-per-stream      :", np.round(v, 6))
# L4: 4 elements per stream using two calls: a,b then a,b
v = []
for i in range(2):
    p = normals_from([i, 0, 0, 0], [1895, 0], 2)
    v += [p[0], p[1], p[4], p[5]]
print("L4 4-per-stream/2call:", np.round(v, 6))
# L5: single global stream, one element per pair (skip cos)
v = normals_from([0, 0, 0, 0], [1895, 0], 1) + normals_from([1, 0, 0, 0], [1895, 0], 1)
print("L5 stream pairs      :", np.round(v, 6))
