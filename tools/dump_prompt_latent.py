# Capture exactly what the reference feeds to the VAE encoder for the clone
# case (A3) and the latent it produces, so the Rust side can be diffed at the
# waveform / latent level.
import sys
sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens/m5_params"
PROMPT_WAV = "data/voice_presets/voice_audio/lingguang.wav"

torch.manual_seed(1895)
tts = MingTTS("Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model

waveform, spk = tts._load_prompt([PROMPT_WAV])
print("padded waveform", tuple(waveform.shape))
np.save(f"{OUT}/prompt_wav_padded.npy", np.ascontiguousarray(waveform[0].float().cpu().numpy()))

latent = m.encode_prompt_latent(waveform)
print("latent", tuple(latent.shape))
np.save(f"{OUT}/prompt_latent.npy", np.ascontiguousarray(latent[0].float().cpu().numpy()))

# the speaker embedding the reference would splice in
se = spk[0].float().cpu().numpy()
np.save(f"{OUT}/spk_emb_192.npy", np.ascontiguousarray(se))
print("spk_emb", se.shape)

# what spk_head projects it to
with torch.inference_mode():
    proj = m.spk_head(torch.from_numpy(se).cuda().unsqueeze(0))
np.save(f"{OUT}/spk_proj_896.npy", np.ascontiguousarray(proj[0].float().cpu().numpy()))
print("spk_proj", tuple(proj.shape))
