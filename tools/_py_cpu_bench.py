import sys, time, torch
sys.path.insert(0, "D:/Ming-3.0-tts")
from ming_tts import MingTTS
t0=time.time()
m=MingTTS("Ming-3.0-tts-0.5b", device="cpu", text_normalization=False)
print("load %.1fs"%(time.time()-t0), flush=True)
for i in range(2):
    t=time.time()
    w=m.synthesize(task="TTS", text="今天天气真不错，我们一起去公园散步吧。", max_length=24, output_path=None)
    dt=time.time()-t; dur=w.shape[-1]/m.sample_rate
    print("run%d synth=%.1fs audio=%.2fs rtf=%.2f"%(i,dt,dur,dt/dur), flush=True)
