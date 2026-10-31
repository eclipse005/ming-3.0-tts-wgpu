# Capture the reference input sequence + embeddings for the clone case, to diff
# the assembly of speaker / prompt-latent rows.
import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
import numpy as np, torch
from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens/m5_params"
PROMPT_WAV = "data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"
TEXT = "今天天气真不错，我们一起去公园散步吧。"

torch.manual_seed(1895)
tts = MingTTS("Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model

orig = m.prepare_input_embed
cap = {}
def spy(**kw):
    ids, emb = orig(**kw)
    cap["ids"] = ids[0].detach().cpu().numpy().astype(np.int64)
    cap["emb"] = emb[0].detach().float().cpu().numpy()
    return ids, emb
m.prepare_input_embed = spy

req = __import__("tts_tasks").build_task_generation_kwargs(
    "TTS", text=TEXT, prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT,
    use_spk_emb=True, cfg=2.0, sigma=0.25, temperature=0.0)
print("request keys:", sorted(req))
prompt_latent, spk = tts.prepare_prompt(req["prompt_wav_path"], True)
with torch.inference_mode():
    _ = m.prepare_input_embed(prompt=req["prompt"], text=TEXT, spk_emb=spk,
                              instruction=None, prompt_latent=prompt_latent,
                              prompt_text=PROMPT_TEXT, use_zero_spk_emb=False)

ids = cap["ids"]; emb = cap["emb"]
print("ids", ids.shape, "emb", emb.shape)
np.save(f"{OUT}/a3_input_ids.npy", np.ascontiguousarray(ids))
np.save(f"{OUT}/a3_inputs_embeds.npy", np.ascontiguousarray(emb))
print("ids tail", ids[-8:].tolist())
print("n audioPatch", int((ids == 151705).sum()), "n vision_pad", int((ids == 151654).sum()))
