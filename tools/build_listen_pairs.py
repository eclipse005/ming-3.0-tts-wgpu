# Pair every params_test reference with the Rust port's rendering and build a
# click-to-play comparison page.
import os
import shutil
import sys

sys.path.insert(0, "D:/Ming-3.0-tts/wgpu/tools")

import numpy as np
import soundfile as sf

from check_params_align import SPEC, compare

REPO = "D:/Ming-3.0-tts"
REF = REPO + "/output/params_test"
NOCACHE = REPO + "/output/ref_nocache"
RUST = REPO + "/output/rust_params"
OUT = REPO + "/output/listen"

# the reference batch reused a cached prompt-latent encoding for these,
# so their fair counterpart is the freshly rendered one
CACHE = {"A4_clone_latent_only", "I1_temperature_06", "I2_temperature_10",
         "I3_cfg_10", "I4_cfg_35"}

TASK_CN = {"TTS": "裸/克隆", "BASIC": "语速音高音量", "EMOTION": "情感",
           "DIALECT": "方言", "IP": "内置音色", "MULTILINGUAL": "多语言",
           "STYLE": "声音设计", "FREEFORM": "自由指令"}

os.makedirs(OUT, exist_ok=True)


def to_pcm16(src, dst):
    x, sr = sf.read(src, dtype="float32")
    sf.write(dst, x, sr, subtype="PCM_16")


rows = []
for i, (name, task, text, kwargs) in enumerate(SPEC, 1):
    ref_src = NOCACHE + "/" + name + ".wav" if name in CACHE else REF + "/" + name + ".wav"
    rust_src = RUST + "/" + name + ".wav"
    if not (os.path.exists(ref_src) and os.path.exists(rust_src)):
        print("missing", name)
        continue
    ref_dst = os.path.join(OUT, "%02d_%s__ref.wav" % (i, name))
    rust_dst = os.path.join(OUT, "%02d_%s__rust.wav" % (i, name))
    # Browsers refuse WAVE_FORMAT_IEEE_FLOAT (which is what torchaudio/our
    # writer produce), so the page gets 16-bit PCM copies. Scoring still runs
    # on the original float files.
    to_pcm16(ref_src, ref_dst)
    to_pcm16(rust_src, rust_dst)
    m = compare(ref_src, rust_src)
    rows.append(dict(idx=i, name=name, task=task, cache=name in CACHE,
                     ref=os.path.basename(ref_dst), rust=os.path.basename(rust_dst), **m))
    print("%02d %-24s rel=%.2e mel=%.5f" % (i, name, m["rel_l2"], m["logmel_corr"]), flush=True)


def verdict(r):
    if abs(r["dur_ratio"] - 1.0) > 1e-9:
        return "时长不同", "warn"
    if r["rel_l2"] < 0.15:
        return "波形级重合", "ok"
    if r["logmel_corr"] > 0.98:
        return "听感级一致", "ok"
    return "需人工判断", "warn"


css = """
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:#14161a;color:#e6e6e6;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 6px}
.sub{color:#8b949e;margin-bottom:18px;max-width:900px}
table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{border-bottom:1px solid #2a2f37;padding:8px;text-align:left;vertical-align:top}
th{color:#8b949e;font-weight:600;font-size:12px;letter-spacing:.04em}
td.n{white-space:nowrap;font-family:ui-monospace,monospace}
.ok{color:#4ec9a0}.warn{color:#e3b341}.m{color:#8b949e;font-size:12px}
audio{height:30px;vertical-align:middle}
.tag{font-size:11px;padding:1px 6px;border-radius:9px;background:#232830;color:#9fd0ff}
.lab{width:34px;color:#8b949e;font-size:12px;display:inline-block}
"""
html = ['<!doctype html><meta charset="utf-8">',
        '<title>Ming-3.0-tts: Python reference vs Rust port</title>',
        '<style>' + css + '</style>',
        '<h1>Python 参考 &nbsp;vs&nbsp; Rust 移植</h1>',
        '<div class="sub">每行一对同名音频：<b>ref</b> = Python 原版，<b>rust</b> = 移植版'
        '（同 seed 1895、同文本、同参数）。建议先听 01 / 29 / 30 建立「听不出差别」的基线，'
        '再看判定为「需人工判断」的那几例。</div>',
        '<table><tr><th>#</th><th>用例</th><th>参考</th><th>移植</th><th>指标</th><th>判定</th></tr>']
for r in rows:
    v, cls = verdict(r)
    note = ' <span class="tag">ref=无缓存重渲染</span>' if r["cache"] else ""
    html.append(
        '<tr><td class="n">%02d</td><td class="n">%s<br><span class="m">%s</span>%s</td>'
        '<td><span class="lab">ref</span><audio controls preload="none" src="%s"></audio></td>'
        '<td><span class="lab">rust</span><audio controls preload="none" src="%s"></audio></td>'
        '<td class="m">时长 x%.4f<br>rel_l2 %.1e<br>log-mel %.5f<br>音色 cos %.4f</td>'
        '<td class="%s">%s</td></tr>'
        % (r["idx"], r["name"], TASK_CN.get(r["task"], r["task"]), note,
           r["ref"], r["rust"], r["dur_ratio"], r["rel_l2"], r["logmel_corr"],
           r["spk_cos"], cls, v))
html.append("</table>")
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write("\n".join(html))

md = ["# 对照听音清单", "",
      "每对文件：NN_<用例>__ref.wav（Python 原版）与 NN_<用例>__rust.wav（Rust 移植）。",
      "", "打开 index.html 可在浏览器里直接点播。", "",
      "| # | 用例 | 时长比 | rel_l2 | log-mel | 音色cos | 判定 |", "|---|---|---|---|---|---|---|"]
for r in rows:
    v, _ = verdict(r)
    md.append("| %02d | %s | x%.4f | %.1e | %.5f | %.4f | %s |"
              % (r["idx"], r["name"], r["dur_ratio"], r["rel_l2"], r["logmel_corr"],
                 r["spk_cos"], v))
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\nwrote", OUT, len(rows), "pairs")
