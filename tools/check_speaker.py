# Speaker-identity check: does the Rust clone actually reproduce the prompt
# voice?  Compares CAMPlus embeddings of the prompt, the Python reference and
# the Rust output.
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

SESS = ort.InferenceSession("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/campplus.onnx",
                           providers=["CPUExecutionProvider"])


def embed(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    x = torch.from_numpy(wav).float().unsqueeze(0)
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    f = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, dither=0.0,
                                         sample_frequency=16000)
    f = f - f.mean(dim=0, keepdim=True)
    out = SESS.run(None, {"input": f.unsqueeze(0).numpy()})[0]
    return out[0], len(wav) / sr


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


paths = {
    "prompt(lingguang)": "D:/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav",
    "ref A3 (python)": "D:/Ming-3.0-tts/output/params_test/A3_clone.wav",
    "rust A3": "D:/Ming-3.0-tts/output/rust/A3_clone.wav",
    "ref A1 (no prompt)": "D:/Ming-3.0-tts/output/params_test/A1_bare.wav",
    "rust A1": "D:/Ming-3.0-tts/output/rust/A1_bare.wav",
}
vecs = {}
for name, p in paths.items():
    try:
        v, dur = embed(p)
        vecs[name] = v
        print(f"{name:20s} {dur:5.2f}s")
    except Exception as e:
        print(f"{name:20s} ERROR {e}")

if len(vecs) >= 3:
    print()
    for a in vecs:
        for b in vecs:
            if a < b:
                print(f"cos({a:18s}, {b:18s}) = {cos(vecs[a], vecs[b]):+.4f}")
