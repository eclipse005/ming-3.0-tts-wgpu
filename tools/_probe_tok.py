import sys, time
sys.path.insert(0, 'D:/Ming-3.0-tts')
import transformers
print("transformers", transformers.__version__)
from transformers import AutoTokenizer
t0=time.time()
tok = AutoTokenizer.from_pretrained("D:/Ming-3.0-tts/Ming-3.0-tts-0.5b", trust_remote_code=True)
print("loaded in", time.time()-t0, type(tok))
print(tok.encode("今天天气真不错，我们一起去公园散步吧。", add_special_tokens=False))
for s in ["<|endoftext|>","<|im_start|>","<|im_end|>","<|vision_start|>","<|vision_end|>","<|vision_pad|>","<audioPatch>","<audio>"]:
    i = tok.convert_tokens_to_ids(s)
    print(repr(s), i, tok.encode(s, add_special_tokens=False))
