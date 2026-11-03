"""Dump every stage of the reference TN so the Rust port can be checked stage by stage.

    C:/Users/ADMIN/miniconda3/envs/tn_ref/python.exe wgpu/tools/_tn_stages.py
"""
import json
import re
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

from frontend.talker_tn.talker_tn import TalkerTN  # noqa: E402
from frontend.talker_tn.talker_pre_processor import TalkerPreProcessor  # noqa: E402
from frontend.talker_tn.talker_re import TalkerRE  # noqa: E402

CASES = [
    "第 12 名",
    "用量 10GB",
    "共 3 人",
    "GPT-4 模型",
    "编号 A-1024",
    "3.2.1 版本",
    "5-8个",
    "3~5天",
    "3.5 万元",
    "百分比 99.9%",
    "时长 3.5小时",
    "华为 Mate 60",
    "价格350元",
    "2026年11月1日",
    "下午2点30分",
    "电话13800138000",
]


def main() -> None:
    out = []
    pre = TalkerPreProcessor()
    re_ = TalkerRE()
    re_.update("D:/Ming-3.0-tts/frontend/talker_tn/talker_re.json")
    tn = TalkerTN()
    for text in CASES:
        p = pre(text)
        if not p:
            r = p
        elif re.search(r"[\u4e00-\u9fff]", p):
            try:
                r = re_(p)
            except Exception as exc:  # noqa: BLE001
                r = f"<talker_re raised {exc!r}>"
            r = re.sub(r"(?<=\d)-(?=\d)", "，", r)
        else:
            r = p
        tagged = ""
        reordered = ""
        try:
            tagged = tn.tn_zh.tag(r)
            reordered = __import__(
                "frontend.talker_tn.token_parser", fromlist=["TokenParser"]
            ).TokenParser("tn").reorder(tagged)
        except Exception as exc:  # noqa: BLE001
            tagged = f"<tag raised {exc!r}>"
        out.append(
            {
                "input": text,
                "pre": p,
                "re": r,
                "tagged": tagged,
                "reordered": reordered,
                "out": tn(text),
            }
        )
    path = "D:/Ming-3.0-tts/wgpu/goldens/tn_stages.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {path} ({len(out)} cases)")


if __name__ == "__main__":
    main()
