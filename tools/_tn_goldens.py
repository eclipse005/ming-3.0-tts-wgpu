"""Generate reference text-normalisation goldens with the real pynini chain.

Run with the `tn_ref` env created from conda-forge (python 3.12 + pynini):

    C:/Users/ADMIN/miniconda3/envs/tn_ref/python.exe wgpu/tools/_tn_goldens.py

Writes goldens/tn_cases.json: [{"input": ..., "expected": ...}]. Only Chinese
text for now, so the `number_en` stage (needs `inflect`) is not exercised.
"""
import json
import re
import sys

sys.path.insert(0, "D:/Ming-3.0-tts")

from frontend.talker_tn.talker_tn import TalkerTN  # noqa: E402
from frontend.toolkit import normalize_text_kit  # noqa: E402


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


CASES = [
    "今天天气3度",
    "价格350元",
    "价格12.5万元",
    "3.5 万元",
    "增长12.5%",
    "GDP增长6.1%",
    "2026年11月1日",
    "11月1日",
    "2026-11-01",
    "下午2点30分",
    "8:30开会",
    "电话13800138000",
    "192.168.1.1",
    "版本v2.1.3",
    "3.2.1 版本",
    "1/2",
    "3/4",
    "第3名",
    "第 12 名",
    "5-8个",
    "3~5天",
    "温度-5度",
    "100,000元",
    "用量 10GB",
    "时长 3.5小时",
    "GPT-4 模型",
    "iPhone 15 Pro",
    "华为 Mate 60",
    "编号 A-1024",
    "百分比 99.9%",
    "$12.99",
    "￥88.5",
    "100美元",
    "共 3 人",
    "2026年",
    "12:00",
    "1.5倍",
    "第1季度",
]


def normalize_fragment(tn: TalkerTN, text: str) -> str:
    """ming_tts._normalize_fragment, without the number_en stage (zh only)."""
    assert is_chinese(text), text
    text = tn(text)
    text = normalize_text_kit(text)
    if text[:1] == "，":
        text = text[1:]
    return text


def main() -> None:
    tn = TalkerTN()
    rows = [{"input": c, "expected": normalize_fragment(tn, c)} for c in CASES if is_chinese(c)]
    out = "D:/Ming-3.0-tts/wgpu/goldens/tn_cases.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    for row in rows:
        print(f"{row['input']!r} -> {row['expected']!r}")
    print(f"wrote {out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
