import sys
sys.path.insert(0, "D:/Ming-3.0-tts")
from frontend.text_segment_cut import cut_text_by_semantic_length
from frontend.toolkit import add_period, preprosess_text

TEXTS = {
    "C6": "今天晚上吃火锅还是烧烤呢？",
    "G2": "本宫赏你的，是恩典；本宫不给的，你断不能抢。在这后宫里，规矩重于天。",
    "H1": "我跟你讲个秘密，你千万别告诉别人。",
    "E1": "这里曾是古蜀文明的中心，沉淀了三千年的时光。",
    "D1": "今日天气真好，一齐去公园行下啦。",
}
for k, t in TEXTS.items():
    frags = cut_text_by_semantic_length(add_period(preprosess_text(t)), 24)["fragments"]
    print(f"{k}: {len(frags)} 片 -> " + " | ".join(repr(f) for f in frags))

print()
print("参考实现在 STYLE 类任务有多片时会做什么（ming_tts.synthesize）:")
for line in open("D:/Ming-3.0-tts/ming_tts.py", encoding="utf-8"):
    if "reference_first_fragment" in line or "_prepare_generated_prompt" in line:
        print("   ", line.rstrip())
