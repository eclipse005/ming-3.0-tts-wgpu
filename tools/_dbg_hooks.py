import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
import numpy as np, torch, fm.modules as fmmod
from ming_tts import MingTTS

torch.manual_seed(1895)
tts = MingTTS("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model
dit = m.flowloss.cfm.model
print("dit type:", type(dit).__name__)
print("has blocks:", hasattr(dit, "blocks"), len(dit.blocks))
b0 = dit.blocks[0]
print("b0 modules:", [n for n, _ in b0.named_children()])
print("mlp:", b0.mlp)
state = {"done": False, "n": 0}
def cb(name):
    def f(mod, inp, out):
        state["n"] += 1
        print("HOOK", name, "done=", state["done"], "n=", state["n"])
    return f
b0.norm1.register_forward_hook(cb("norm1"))
b0.register_forward_hook(cb("block0"))
orig = dit.forward
def spy(x, t, c, lh, mask=None):
    print("FORWARD called, done=", state["done"])
    out = orig(x, t, c, lh, mask)
    state["done"] = True
    return out
dit.forward = spy
with torch.inference_mode():
    tts.synthesize(task="TTS", text="今天天气真不错，我们一起去公园散步吧。", controls=None,
                   instruction=None, output_path=None, max_decode_steps=2)
print("hooks fired:", state["n"])
