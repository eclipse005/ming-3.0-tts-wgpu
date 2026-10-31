# Dump VAE-encoder goldens for M3: the exact waveform fed to encode_latent,
# the gaussian parameters (mean/std), the sampled latent, and intermediates.
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

cap = {}
orig_enc = m.audio.encode_latent

def enc_spy(waveform, waveform_length):
    latent, frame_num = orig_enc(waveform, waveform_length)
    cap["waveform"] = waveform[0].detach().float().cpu().numpy()
    cap["len"] = int(waveform_length.item())
    cap["latent"] = latent[0].detach().float().cpu().numpy()
    cap["frame_num"] = int(frame_num.item())
    enc_mod = m.audio.encoder
    with torch.inference_mode():
        frames = enc_mod.get_frames(waveform)
        cap["frames"] = frames[0].detach().float().cpu().numpy()
        e = enc_mod.fc1(frames)
        e = enc_mod.fc2(e)
        cap["embeds"] = e[0].detach().float().cpu().numpy()
        h = None
        def l0_hook(mod, inp, out):
            nonlocal h
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        l0 = enc_mod.encoder.layers[0]
        orig_qkv = l0.self_attn.qkv_attention
        def qkv_spy(q, k, v, past_key_values=None):
            if "l0_q_rope" not in cap:
                cap["l0_q_rope"] = q[0].detach().float().cpu().numpy()
                cap["l0_k_rope"] = k[0].detach().float().cpu().numpy()
            return orig_qkv(q, k, v, past_key_values=past_key_values)
        l0.self_attn.qkv_attention = qkv_spy
        l0.self_attn.o_proj.register_forward_hook(
            lambda mod, i, o: cap.__setitem__("l0_attn", i[0][0].detach().float().cpu().numpy()))
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(l0.self_attn, nm).register_forward_hook(
                (lambda nm: lambda mod, i, o: cap.__setitem__(f"l0_{nm}", o[0].detach().float().cpu().numpy()))(nm))
        for nm in ("gate_proj", "up_proj", "down_proj"):
            getattr(l0.mlp, nm).register_forward_hook(
                (lambda nm: lambda mod, i, o: cap.__setitem__(f"l0_{nm}", o[0].detach().float().cpu().numpy()))(nm))
        l0.input_layernorm.register_forward_hook(
            lambda mod, i, o: cap.__setitem__("l0_attnln", o[0].detach().float().cpu().numpy()))
        hooks = [l0.register_forward_hook(
            lambda mod, i, o: cap.__setitem__("layer0", o[0].detach().float().cpu().numpy()))]
        h = enc_mod.encoder(inputs_embeds=e).last_hidden_state
        for hk in hooks:
            hk.remove()
        cap["enc_out"] = h[0].detach().float().cpu().numpy()
        a = enc_mod.pad_patch_insert_cls(h)
        a = enc_mod.aggregator(inputs_embeds=a).last_hidden_state
        cap["agg_out"] = a[0].detach().float().cpu().numpy()
        o = enc_mod.fc3(a)
        h2 = o.transpose(1, 2)
        mean, logvar = torch.chunk(h2, 2, dim=1)
        cap["mean"] = mean[0].detach().float().cpu().numpy().T
        cap["std"] = torch.exp(0.5 * logvar)[0].detach().float().cpu().numpy().T
    return latent, frame_num

m.audio.encode_latent = enc_spy

with torch.inference_mode():
    tts.synthesize(task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV,
                   prompt_text=PROMPT_TEXT, use_spk_emb=True, output_path=None)

wf = cap["waveform"]
print(f"waveform {wf.shape} len={cap['len']} frames={cap['frame_num']} "
      f"latent={cap['latent'].shape}")
for k in ("waveform", "frames", "embeds", "l0_q_rope", "l0_k_rope", "l0_attnln", "l0_attn", "l0_q_proj", "l0_k_proj", "l0_v_proj", "l0_o_proj", "l0_gate_proj", "l0_up_proj", "l0_down_proj", "layer0", "enc_out", "agg_out", "mean", "std", "latent"):
    np.save(f"{OUT}/m3_{k}.npy", cap[k])
np.save(f"{OUT}/m3_eps.npy",
        (cap["latent"] - cap["mean"]) / np.maximum(cap["std"], 1e-8))

from safetensors import safe_open
with safe_open("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/model.safetensors", framework="numpy") as f:
    for pat in ("audio.encoder.fc1", "audio.encoder.fc3", "audio.encoder.cls_embed",
                "audio.encoder.aggregator.layers.0.self_attn.q_proj.weight"):
        for k in f.keys():
            if k.startswith(pat) or k == pat:
                print("key:", k)
print("done")
