import sys
sys.path.insert(0, 'D:/Ming-3.0-tts')
import torch, torchaudio, os
print("torchaudio", torchaudio.__version__)
print("backend list loader", torchaudio.list_audio_backends())
os.makedirs('D:/Ming-3.0-tts/wgpu/goldens/m5_audio/_probe', exist_ok=True)
x = torch.tensor([[0.0, 0.5, -0.25, 1.0, -1.0, 0.1, 0.2, 0.3]])
p = 'D:/Ming-3.0-tts/wgpu/goldens/m5_audio/_probe/probe_44100.wav'
torchaudio.save(p, x, sample_rate=44100)
b = open(p,'rb').read()
print("bytes", len(b))
print(b[:96].hex(' '))
print("data chunk offset search:", b.find(b'data'))
