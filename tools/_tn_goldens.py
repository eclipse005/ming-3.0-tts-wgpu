"""Generate reference text-normalisation goldens with the real pynini chain.

Run with the `tn_ref` env created from conda-forge (python 3.12 + pynini +
inflect):

    C:/Users/ADMIN/miniconda3/envs/tn_ref/python.exe wgpu/tools/_tn_goldens.py

Writes goldens/tn_cases.json: [{"input": ..., "expected": ...}], covering the
Chinese branch (TalkerRE + zh FSTs) and the English branch (normalize_numbers +
en FSTs).
"""
import json
import re
import sys

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, "D:/Ming-3.0-tts")

from frontend.number_en import normalize_numbers  # noqa: E402
from frontend.talker_tn.talker_tn import TalkerTN  # noqa: E402
from frontend.toolkit import normalize_text_kit  # noqa: E402


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


ZH_CASES = [
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
    "温度是-5.5度",
    "他跑了5.5公里",
    "重量约3.2kg",
    "文件大小是10.5MB",
    "房租每月3000元",
    "这件衣服199块钱",
    "工资是5000美元",
    "花了¥88",
    "价格上涨了3.5%",
    "下降了0.5个百分点",
    "电话是010-12345678",
    "客服400-800-1234",
    "服务器10.0.0.1",
    "版本号v1.2.3",
    "上午9:30开会",
    "下午3:45见",
    "晚上8点",
    "12:00整",
    "2026-11-01的会议",
    "11月1号",
    "3/4的苹果",
    "1+2=3",
    "3*4=12",
    "10-2=8",
    "序列号A1B2C3",
    "订单号AB-1234",
    "1. 第一项",
    "这是第3个",
    "共12人参加",
]

EN_CASES = [
    "I have 3 apples and 25 oranges.",
    "It costs $12.99 today.",
    "That is £-150 in total.",
    "The price is 25.5% higher.",
    "Add 3/4 of the flour.",
    "He finished 1st and she was 2nd.",
    "The file is 10ms long.",
    "It weighs 5.5 kg.",
    "The speed is 3.2 GHz.",
    "Use gpt-4 or v2.1 for this.",
    "Call me at 1,234,567.",
    "We need 100,000 dollars.",
    "Version 2.0 is out.",
    "In 2026 we ship.",
    "The answer is 42.",
    "Ratio 1/2 or 5/8.",
    "It takes 30s to load.",
    "Memory is 16GB.",
    "Score 99.9%.",
    "The 21st century.",
    "About 1,000,000 people.",
    "A 3.5-inch disk.",
    "Visit https://example.com/x?y=1 now.",
    "Use **bold** and `code` here.",
    "Email me at 5 pm, please.",
    "There are two dogs and 3 cats.",
    "Prices rose 12.5% and 3.2% respectively.",
    "Model gpt-3.5-turbo works.",
    "Temperature is -5.5°C outside.",
    "The 1st, 2nd and 3rd items.",
]


def normalize_fragment(tn: TalkerTN, text: str) -> str:
    """ming_tts._normalize_fragment, stage for stage."""
    if not is_chinese(text):
        text = normalize_numbers(text)
    text = tn(text)
    text = normalize_text_kit(text)
    if text[:1] == "，":
        text = text[1:]
    return text


def main() -> None:
    tn = TalkerTN()
    rows = [
        {"input": c, "expected": normalize_fragment(tn, c)}
        for c in ZH_CASES + EN_CASES
    ]
    out = "D:/Ming-3.0-tts/wgpu/goldens/tn_cases.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    for row in rows:
        print(f"{row['input']!r} -> {row['expected']!r}")
    print(f"wrote {out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
