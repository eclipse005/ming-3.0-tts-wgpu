# Compare two rendered utterances: sample-level agreement plus the
# perception-relevant summaries (duration, log-mel correlation).
import sys

import numpy as np
import soundfile as sf


def logmel(x, sr, n_mels=80, n_fft=1024, hop=256):
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    if n <= 0:
        return np.zeros((n_mels, 0))
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, n_fft), strides=(x.strides[0] * hop, x.strides[0]))
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # mel filterbank (HTK-ish)
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)
    mel = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mel) / sr).astype(int)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    out = spec @ fb.T
    return np.log(np.maximum(out, 1e-8)).T


def main(a, b):
    x, sr1 = sf.read(a, dtype="float32")
    y, sr2 = sf.read(b, dtype="float32")
    print(f"A {a}: {x.shape} @ {sr1}")
    print(f"B {b}: {y.shape} @ {sr2}")
    if sr1 != sr2:
        print("RATE MISMATCH")
        return
    print(f"duration A={len(x)/sr1:.3f}s B={len(y)/sr2:.3f}s "
          f"ratio={len(y)/len(x):.4f}")
    n = min(len(x), len(y))
    d = x[:n] - y[:n]
    print(f"aligned samples={n} rel_l2={np.linalg.norm(d)/max(np.linalg.norm(x[:n]),1e-12):.3e} "
          f"maxabs={np.abs(d).max():.3e}")
    print(f"peak A={np.abs(x).max():.4f} B={np.abs(y).max():.4f}")
    mx = logmel(x, sr1)
    my = logmel(y, sr2)
    k = min(mx.shape[1], my.shape[1])
    if k > 4:
        ca = mx[:, :k] - mx[:, :k].mean()
        cb = my[:, :k] - my[:, :k].mean()
        corr = float((ca * cb).sum() / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
        print(f"log-mel correlation (first {k} frames) = {corr:.6f}")
        print(f"log-mel mean abs diff = {np.abs(mx[:, :k]-my[:, :k]).mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
