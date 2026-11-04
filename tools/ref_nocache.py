# The reference's params_test batch reuses a cached prompt-latent encoding for
# A4 / I1-I4 (cache_prompts=True), so those WAVs were produced from a noise
# stream that skipped the posterior draw. A CLI invocation (ours, and infer.py)
# encodes the prompt every time. Re-render those cases with cache_prompts=False
# so the comparison is like-for-like.
import sys
HERE = Path(__file__).resolve().parents[1]
REF_REPO = os.environ.get("MING_REF", "D:/Ming-3.0-tts")
sys.path.insert(0, REF_REPO)
os.chdir(REF_REPO)
import numpy as np, torch, os
from ming_tts import MingTTS

OUT = REF_REPO + "/output/ref_nocache"
PROMPT_WAV = "data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"
CLONE = dict(prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT, use_spk_emb=True)
TEXT = "今天天气真不错，我们一起去公园散步吧。"

CASES = [
    ("A3_clone", "TTS", TEXT, dict(CLONE)),
    ("A1_bare_rerender", "TTS", TEXT, {}),
    ("A4_clone_latent_only", "TTS", TEXT, dict(prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT)),
    ("I1_temperature_06", "TTS", TEXT, dict(CLONE, temperature=0.6)),
    ("I2_temperature_10", "TTS", TEXT, dict(CLONE, temperature=1.0)),
    ("I3_cfg_10", "TTS", TEXT, dict(CLONE, cfg=1.0)),
    ("I4_cfg_35", "TTS", TEXT, dict(CLONE, cfg=3.5)),
]

os.makedirs(OUT, exist_ok=True)
model = MingTTS(REF_REPO + "/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False,
                cache_prompts=False)
for name, task, text, kwargs in CASES:
    torch.manual_seed(1895)
    wav = model.synthesize(task=task, text=text, max_length=24,
                           output_path=f"{OUT}/{name}.wav", **kwargs)
    print(f"[{name}] {wav.shape[-1]} samples", flush=True)
print("done")
