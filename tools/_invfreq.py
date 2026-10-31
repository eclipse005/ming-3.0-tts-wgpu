import torch, numpy as np
from safetensors.torch import load_file
sd = load_file("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b/model.safetensors")
k = "flowloss.cfm.model.rotary_embed.inv_freq"
ref = sd[k].float().numpy()
ours = np.array([10000.0 ** (-2.0 * i / 64.0) for i in range(32)], dtype=np.float64)
print("checkpoint inv_freq[:6] :", np.round(ref[:6], 8))
print("ours (theta=1e4, 2i/64) :", np.round(ours[:6], 8))
print("比值 ref/ours          :", np.round(ref / ours, 6))
print()
print("=> 参考用的 base 反推:")
i = 1
base_est = (1.0 / ref[i]) ** (64.0 / (2 * i))
print("   i=1  -> base = %.6f" % base_est)
i = 8
base_est = (1.0 / ref[i]) ** (64.0 / (2 * i))
print("   i=8  -> base = %.6f" % base_est)
print()
print("inv_freq 最大相对偏差: %.3e" % float(np.abs(ref - ours).max() / np.abs(ours).max()))
