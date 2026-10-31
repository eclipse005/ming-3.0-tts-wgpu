# Are the flagged cases even conditioned identically?  Dump the reference's
# token ids for C6 / G2 so they can be diffed against the CLI's.
import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
import numpy as np, torch
from ming_tts import MingTTS
import tts_tasks

OUT = "D:/Ming-3.0-tts/wgpu/goldens/m5_params"
CASES = [
    ("C6_surprised", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "惊讶"})),
    ("G2_empress", "STYLE",
     "本宫赏你的，是恩典；本宫不给的，你断不能抢。在这后宫里，规矩重于天。",
     dict(controls={"style": "这是一个宫斗剧中的威严皇后的声音，展现了充满张力的戏剧叙事风格。"
                            "她使用雍容华贵的普通话，中年女性声音沉稳。语速雍容和缓，"
                            "但每个字都掷地有声，语气表面波澜不惊，实则暗藏锋芒和久居上位的威压。"})),
]

torch.manual_seed(1895)
tts = MingTTS("Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
m = tts.model

for name, task, text, kwargs in CASES:
    req = tts_tasks.build_task_generation_kwargs(task, text=text, cfg=2.0,
                                                 sigma=0.25, temperature=0.0, **kwargs)
    ids, spk_pos, lat_pos = __import__("prompt_contract").build_input_ids(
        m.tokenizer, m.model_type, req["prompt"], text, num_speakers=0,
        instruction=req.get("instruction"), prompt_text=None, prompt_latent_steps=0)
    np.save(OUT + "/" + name + "_ids.npy", np.ascontiguousarray(np.array(ids, dtype=np.int64)))
    with open(OUT + "/" + name + "_prompt.txt", "w", encoding="utf-8") as f:
        f.write(req["prompt"])
    print(f"{name}: ids={len(ids)}  prompt_len={len(req['prompt'])}")
    print("   prompt 前 60 字:", repr(req["prompt"][:60]))
