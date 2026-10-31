# Dump single-layer golden tensors from the Python pipeline for M0 parity.
#
#   python tools/dump_golden.py
#
# Writes wgpu/goldens/*.npy. Inputs are the real clone-demo request
# (lingguang prompt + A3 text), so activation ranges are realistic
# (including the 7.4e4 residual spike at layer 3).

import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

import numpy as np
import torch

from ming_tts import MingTTS
import tts_tasks

OUT = "D:/Ming-3.0-tts/wgpu/goldens"
TEXT = "今天天气真不错，我们一起去公园散步吧。"
PROMPT_WAV = "D:/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

torch.manual_seed(1895)

tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0",
              text_normalization=False)
m = tts.model

with torch.inference_mode():
    prompt_latent, spk_emb = tts.prepare_prompt(PROMPT_WAV, use_spk_emb=True)
    input_ids, inputs_embeds = m.prepare_input_embed(
        prompt=tts_tasks.DEFAULT_TTS_PROMPT,
        text=TEXT,
        spk_emb=[e.clone() for e in spk_emb],
        prompt_latent=prompt_latent,
        prompt_text=PROMPT_TEXT,
    )
    T = inputs_embeds.shape[1]
    print(f"sequence length T = {T}")

    outputs = m.model(
        attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
        position_ids=(torch.ones_like(input_ids).cumsum(-1) - 1),
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    h3 = outputs.hidden_states[3][0]          # [T, 896] — input to layer 4
    print(f"h3 absmax = {h3.abs().max().item():.3e}")  # expect ~7.4e4 spike

    layer4 = m.model.model.layers[4]
    normed = layer4.input_layernorm(h3)
    q_out = layer4.self_attn.q_proj(normed)
    emb = m.model.get_input_embeddings()(input_ids)[0]

    def save(name, tensor):
        a = tensor.detach().float().cpu().numpy()
        np.save(f"{OUT}/{name}", a)
        print(f"  {name}: shape={a.shape} absmax={np.abs(a).max():.3e}")

    save("h3_hidden.npy", h3)
    save("layer4_norm_w.npy", layer4.input_layernorm.weight)
    save("layer4_norm_out.npy", normed)
    save("layer4_q_out.npy", q_out)
    save("layer4_q_w.npy", layer4.self_attn.q_proj.weight)
    save("layer4_q_b.npy", layer4.self_attn.q_proj.bias)
    save("embed_out.npy", emb)
    np.save(f"{OUT}/input_ids.npy", input_ids[0].cpu().numpy().astype(np.int64))

# print the exact safetensors key for the cross-check test
from safetensors import safe_open
with safe_open("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/model.safetensors", framework="numpy") as f:
    keys = [k for k in f.keys() if "layers.4.self_attn.q_proj" in k]
    print("checkpoint keys:", keys)
    k0 = keys[0]
    t = f.get_tensor(k0)
    print(f"key dtype: {t.dtype}, shape: {t.shape}")
print("done ->", OUT)
