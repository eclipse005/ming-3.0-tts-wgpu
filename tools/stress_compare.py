# Coverage stress suite: the axes the 36 params_test cases never touch --
# multi-fragment (non-STYLE), long single utterances (incl. the 512-position
# KV ceiling), very short text, digits/symbols with the FST normaliser on,
# long English, mixed script, and --raw_text.
import gc, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REF_REPO = os.environ.get("MING_REF", "D:/Ming-3.0-tts")
sys.path.insert(0, REF_REPO)
sys.path.insert(0, str(HERE / "tools"))

import numpy as np
import soundfile as sf
import torch

from check_params_align import compare, logmel
from ming_tts import MingTTS

EXE = os.environ.get("MING_EXE", str(HERE / "target" / "release" / "ming-tts-wgpu.exe"))
MODEL = os.environ.get("MING_MODEL", REF_REPO + "/Ming-3.0-tts-0.5b")
OUT = REF_REPO + "/output/stress"
REF = OUT + "/ref"
RUST = OUT + "/rust"

ZH60 = ("清晨的山谷里飘着薄雾，远处传来溪水的声音，"
        "我沿着小路慢慢往前走，想把这安静的景色记在心里。")
ZH150 = ("傍晚的时候，城里的灯一盏一盏亮起来，街边的小店开始热闹了。"
         "有人在排队买刚出炉的面包，有人牵着狗在路口等红灯。"
         "我站在天桥上看了很久，忽然想起小时候住的那条巷子，"
         "夏天的晚上也是这样，风扇在转，电视里放着新闻，"
         "邻居家的孩子还在楼下喊着不肯回家。")
ZH200 = ("我们今天的会议主要讨论三件事：第一是模型蒸馏的进度，第二是推理速度的优化，"
         "第三是上线之前的验收标准。关于第一件事，训练组已经完成了教师模型的数据标注，"
         "学生的损失曲线在第三周之后开始收敛，目前验证集上的差距是百分之一点三，"
         "我们计划在下周再做一次消融实验，确认数据配比的影响。"
         "关于第二件事，工程组把注意力内核重新写了一遍，在同样的硬件上快了百分之四十，"
         "但显存占用也涨了一点，需要再压一压。第三件事等前两件定下来再细聊。")
EN260 = ("Good evening. Tonight we are going to talk about why small speech models "
         "suddenly became interesting again, what makes a fifty-megabyte checkpoint "
         "usable in a browser, and where the remaining quality gap actually comes from. "
         "Stay with us for the next twenty minutes.")

CASES = [
    # name, task, text, kwargs, tn
    ("S1_one_char",   "TTS", "嗯。", {}, False),
    ("S2_short",      "TTS", "好的，谢谢。", {}, False),
    ("S3_zh60_one",   "TTS", ZH60,  dict(max_length=500), False),
    ("S4_zh60_split", "TTS", ZH60,  {}, False),
    ("S5_zh150_split","TTS", ZH150, {}, False),
    ("S6_zh200_cap",  "TTS", ZH200, dict(max_length=500), False),
    ("S7_digits_tn",  "TTS", "价格是 350 元，2026 年 11 月 1 日，增长 12.5%。", {}, True),
    ("S8_en260",      "MULTILINGUAL", EN260, dict(language_id="en", max_length=500), False),
    ("S9_mixed",      "TTS", "我们用 GPT-4 和 Claude 3.5 做了 A/B 测试，效果提升 12%。", {}, False),
    ("S10_raw",       "TTS", ZH60, dict(raw_text=True, max_length=500), False),
]


def ref_kwargs(kw):
    out = {}
    for k in ("language_id", "instruction", "max_length", "raw_text",
              "prompt_wav_path", "prompt_text", "use_spk_emb", "use_zero_spk_emb",
              "cfg", "sigma", "temperature"):
        if k in kw:
            out[k] = kw[k]
    c = kw.get("controls")
    if c:
        out["controls"] = c
    return out


def rust_args(name, task, text, kw, tn):
    args = [EXE, "synth", "--model_path", MODEL, "--task", task, "--text", text,
            "--output", f"{RUST}/{name}.wav"]
    if not tn:
        args.append("--no_text_normalization")
    if kw.get("raw_text"):
        args.append("--raw_text")
    if "max_length" in kw:
        args += ["--max_length", str(kw["max_length"])]
    if kw.get("language_id"):
        args += ["--language_id", kw["language_id"]]
    if kw.get("prompt_wav_path"):
        args += ["--prompt_wav_path", kw["prompt_wav_path"]]
    if kw.get("prompt_text"):
        args += ["--prompt_text", kw["prompt_text"]]
    if kw.get("use_spk_emb"):
        args.append("--use_spk_emb")
    return args


def render_ref(model, name, task, text, kw):
    torch.manual_seed(1895)
    wav = model.synthesize(task=task, text=text,
                           output_path=f"{REF}/{name}.wav", **ref_kwargs(kw))
    return int(wav.shape[-1])


def main():
    os.makedirs(REF, exist_ok=True)
    os.makedirs(RUST, exist_ok=True)
    argv = sys.argv[1:]
    skip_ref = "--skip-ref" in argv
    only = [a for a in argv if not a.startswith("--")] or None
    cases = [c for c in CASES if not only or c[0] in only]

    # ---- reference (one model per TN mode) ------------------------------
    if not skip_ref:
        model = MingTTS(MODEL, device="cuda:0", text_normalization=False,
                        cache_prompts=False)
        for name, task, text, kw, tn in cases:
            if tn:
                continue
            n = render_ref(model, name, task, text, kw)
            print(f"[ref ] {name:16s} {n/44100:.2f}s", flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
        # The reference needs pynini for text_normalization=True. Report the
        # skip instead of dying: on a box without it, the TN flag is dead on
        # both sides and there is nothing comparable to render.
        tn_cases = [c for c in cases if c[4]]
        if tn_cases:
            try:
                model = MingTTS(MODEL, device="cuda:0", text_normalization=True,
                                cache_prompts=False)
            except Exception as exc:  # pynini / WeTextProcessing missing
                print(f"[skip] TN cases: {exc!r}", flush=True)
                model = None
            if model is not None:
                for name, task, text, kw, tn in tn_cases:
                    try:
                        n = render_ref(model, name, task, text, kw)
                        print(f"[ref ] {name:16s} {n/44100:.2f}s (TN on)", flush=True)
                    except Exception as exc:
                        print(f"[skip] {name}: {exc!r}", flush=True)
                del model
                gc.collect(); torch.cuda.empty_cache()

    # ---- port -----------------------------------------------------------
    for name, task, text, kw, tn in cases:
        p = subprocess.run(rust_args(name, task, text, kw, tn),
                           capture_output=True, text=True)
        tail = [l for l in (p.stdout or "").splitlines() if l.strip()]
        frag = [l for l in tail if l.startswith("fragments:")]
        if p.returncode != 0:
            print(f"[rust] {name:16s} FAILED {(p.stderr or p.stdout).strip().splitlines()[-1:]}",
                  flush=True)
        else:
            print(f"[rust] {name:16s} {frag[0] if frag else ''}", flush=True)

    # ---- compare --------------------------------------------------------
    rows = []
    for name, task, text, kw, tn in cases:
        a, b = f"{REF}/{name}.wav", f"{RUST}/{name}.wav"
        if not (os.path.exists(a) and os.path.exists(b)):
            rows.append(dict(name=name, ok=False))
            print(f"[MISS] {name}", flush=True)
            continue
        m = compare(a, b)
        m.update(name=name, ok=True, tn=tn)
        rows.append(m)
        print(f"[cmp ] {name:16s} dur {m['dur_ratio']:.4f} rel={m['rel_l2']:.2e} "
              f"mel={m['logmel_corr']:.5f} spk={m['spk_cos']:.4f}", flush=True)

    with open(f"{OUT}/report.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
