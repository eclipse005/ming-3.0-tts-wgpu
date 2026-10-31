# Where exactly does the flagged pair differ? Spread-out difference = another
# valid rendition of the same words; a concentrated hotspot = a real defect.
import numpy as np, soundfile as sf

L = "D:/Ming-3.0-tts/output/listen/"
CASES = [("01","A1_bare","control"), ("17","C6_surprised","flagged"), ("30","G2_empress","flagged")]

def logmel(x, sr, n_mels=80, n_fft=1024, hop=256):
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(x, (n, n_fft), (x.strides[0]*hop, x.strides[0]))
    spec = np.abs(np.fft.rfft(fr*win, axis=1))**2
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    h2m = lambda f: 2595.0*np.log10(1.0+f/700.0); m2h = lambda m: 700.0*(10**(m/2595.0)-1.0)
    mel = np.linspace(h2m(0), h2m(sr/2), n_mels+2)
    bins = np.floor((n_fft+1)*m2h(mel)/sr).astype(int)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels+1):
        l,c,r = bins[m-1], bins[m], bins[m+1]
        for k in range(l,c): fb[m-1,k] = (k-l)/(c-l) if c>l else 0
        for k in range(c,r): fb[m-1,k] = (r-k)/(r-c) if r>c else 0
    return np.log(np.maximum(spec@fb.T, 1e-8)).T

for idx, name, tag in CASES:
    a, sr = sf.read(L+idx+"_"+name+"__ref.wav", dtype="float32")
    b, _  = sf.read(L+idx+"_"+name+"__rust.wav", dtype="float32")
    print("=== %s %s  ref=%.2fs rust=%.2fs (%s)" % (idx, name, len(a)/sr, len(b)/sr, tag))
    n = min(len(a), len(b))
    A, B = logmel(a, sr), logmel(b, sr)
    k = min(A.shape[1], B.shape[1])
    d = np.abs(A[:, :k]-B[:, :k]).mean(axis=0)          # per-frame mel distance
    print("   log-mel 帧级差: mean=%.3f  p50=%.3f  p90=%.3f  max=%.3f (帧数 %d)" %
          (d.mean(), np.percentile(d,50), np.percentile(d,90), d.max(), k))
    # where is the top-10% of the difference?
    thr = np.percentile(d, 90)
    hot = np.where(d >= thr)[0]
    print("   最大差异出现在 %.2fs ~ %.2fs（占时长 %.0f%%）" %
          (hot[0]*256/sr, hot[-1]*256/sr, 100.0*len(hot)/k))
    # energy envelope agreement
    hop = sr//50
    ea = np.array([np.sqrt((a[i:i+hop]**2).mean()) for i in range(0, n-hop, hop)])
    eb = np.array([np.sqrt((b[i:i+hop]**2).mean()) for i in range(0, n-hop, hop)])
    m = min(len(ea), len(eb))
    print("   能量包络相关 = %.4f | 峰值 ref %.3f rust %.3f | 静音帧占比 ref %.2f rust %.2f" %
          (np.corrcoef(ea[:m], eb[:m])[0,1], np.abs(a).max(), np.abs(b).max(),
           (ea[:m] < 1e-3).mean(), (eb[:m] < 1e-3).mean()))
    print()
