# Dump the real LLM decode sequence for M1 parity: prefill embeds, per-step
# decode inputs (linear_proj outputs), per-step last hidden state and stop
# probability, straight out of a seeded real synthesize run.
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS

OUT = "D:/Ming-3.0-tts/wgpu/goldens"
TEXT = "今天天气真不错，我们一起去公园散步吧。"
PROMPT_WAV = "D:/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

torch.manual_seed(1895)
tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model

rec = {"prefill_embeds": None, "step_inputs": [], "step_z": [], "step_prob": [],
       "step_pos": []}

orig_llm_fwd = m.model.forward
llm_calls = {"n": 0}

def llm_spy(*args, **kwargs):
    llm_calls["n"] += 1
    out = orig_llm_fwd(*args, **kwargs)
    if llm_calls["n"] == 1:
        rec["prefill_embeds"] = kwargs["inputs_embeds"][0].detach().float().cpu().numpy()
    z = out.hidden_states[-1][:, -1:, :]
    rec["step_z"].append(z[0, 0].detach().float().cpu().numpy())
    pos = kwargs["position_ids"]
    rec["step_pos"].append(int(pos[0, 0, -1].item()) if pos.ndim == 3
                           else int(pos[0, -1].item()))
    return out

m.model.forward = llm_spy

orig_pal = m._project_audio_latent
def pal_spy(x):
    out = orig_pal(x)
    rec["step_inputs"].append(out[0, 0].detach().float().cpu().numpy())
    return out

m._project_audio_latent = pal_spy

orig_stop = m.stop_head.forward
def stop_spy(*args, **kwargs):
    out = orig_stop(*args, **kwargs)
    rec["step_prob"].append(out[0][0].softmax(dim=-1)[1].item())
    return out

m.stop_head.forward = stop_spy

with torch.inference_mode():
    wav = tts.synthesize(task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV,
                         prompt_text=PROMPT_TEXT, use_spk_emb=True, output_path=None)

S = len(rec["step_inputs"])
print(f"llm calls={llm_calls['n']} steps={S} dur={wav.shape[-1]/tts.sample_rate:.2f}s")
stop_step = next(i for i, p in enumerate(rec["step_prob"]) if p > 0.5 and i > 3)
print(f"stop_step={stop_step} prob at stop={rec['step_prob'][stop_step]:.4f}")
print("positions:", rec["step_pos"][:5], "...", rec["step_pos"][-2:])

np.save(f"{OUT}/m1_prefill_embeds.npy", rec["prefill_embeds"])
np.save(f"{OUT}/m1_step_inputs.npy", np.stack(rec["step_inputs"]))
np.save(f"{OUT}/m1_step_z.npy", np.stack(rec["step_z"]))
np.save(f"{OUT}/m1_step_prob.npy", np.array(rec["step_prob"], dtype=np.float32))

from safetensors import safe_open
with safe_open("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/model.safetensors", framework="numpy") as f:
    for pat in ("stop_head", "model.model.norm.weight", "spk_head"):
        for k in f.keys():
            if pat in k:
                print("key:", k)
print("done")
