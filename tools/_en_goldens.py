"""Dump the reference English number wording (inflect 7.6.0) and normalize_numbers goldens.

    C:/Users/ADMIN/miniconda3/envs/tn_ref/python.exe wgpu/tools/_en_goldens.py
"""
import json
import random
import sys

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, "D:/Ming-3.0-tts")

import inflect  # noqa: E402

from frontend.number_en import normalize_numbers  # noqa: E402

e = inflect.engine()

WORDS = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 15, 19, 20, 21, 22, 30, 40, 45, 55, 67, 78, 89, 99,
         100, 101, 105, 110, 111, 115, 120, 150, 199, 200, 201, 210, 300, 555, 900, 999,
         1000, 1001, 1010, 1100, 1111, 1500, 2000, 2026, 3000, 9999, 10000, 10001, 12345,
         20000, 99999, 100000, 100001, 123456, 999999, 1000000, 1000001, 2000000,
         1234567, 10000000, 12345678, 123456789, 1000000000, 1234567890,
         10**12, 10**12 + 1, 2 * 10**12 + 345, 10**15, 1234567890123]
random.seed(1895)
WORDS += [random.randrange(0, 10**6) for _ in range(120)]
WORDS += [random.randrange(0, 10**9) for _ in range(60)]
WORDS += [random.randrange(0, 10**12) for _ in range(30)]

ORDINALS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "twenty", "twenty-one", "thirty", "forty",
            "one hundred", "one hundred and one", "one hundred and twenty-one", "one thousand"]

CASES = [
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
    "Temperature -5 degrees.",
    "Ratio 1/2 or 5/8.",
    "It takes 30s to load.",
    "Memory is 16GB.",
    "Score 99.9%.",
    "Try 1/4 cup and 1/8 tsp.",
    "Model gpt-3.5-turbo works.",
    "The 21st century.",
    "About 1,000,000 people.",
    "A 3.5-inch disk.",
]


def main() -> None:
    words = [
        {"n": n, "and": e.number_to_words(n), "noand": e.number_to_words(n, andword="")}
        for n in WORDS
    ]
    ordinals = [{"word": w, "ordinal": e.ordinal(w)} for w in ORDINALS]
    cases = [{"input": t, "expected": normalize_numbers(t)} for t in CASES]
    base = "D:/Ming-3.0-tts/wgpu/goldens/"
    for name, data in [
        ("number_en_words.json", {"words": words, "ordinals": ordinals}),
        ("number_en_cases.json", cases),
    ]:
        with open(base + name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"wrote {name}")
    for row in cases[:6]:
        print(f"{row['input']!r} -> {row['expected']!r}")


if __name__ == "__main__":
    main()
