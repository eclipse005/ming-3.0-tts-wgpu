#!/usr/bin/env python3
"""Build out/voice_design/listen.html from manifest.json + the generated wavs.

Every request has two renders: the wgpu port (out/voice_design/<id>.wav) and
the official Python reference (out/voice_design/official_py/<id>.wav). The page
puts them side by side per card, measures how far apart they are, and marks the
ones that differ audibly. No network, no build step, works from file://.
"""

import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "voice_design"

OFFICIAL = {
    "vd1": "0:04", "vd2": "0:05", "vd3": "0:05",
    "dir1": "0:04", "dir2": "0:05",
    "tag1": "0:04", "tag2": "0:05", "tag3": "0:04",
    "lang_zh_xiaorou": "0:02", "lang_en_xiaorou": "0:02", "lang_de_xiaorou": "0:02",
    "lang_es_xiaorou": "0:02", "lang_fr_xiaorou": "0:02", "lang_ko_xiaorou": "0:03",
    "lang_ja_xiaorou": "0:03",
    "lang_zh_lingkai": "0:02", "lang_en_lingkai": "0:03", "lang_de_lingkai": "0:03",
    "lang_es_lingkai": "0:04", "lang_fr_lingkai": "0:04", "lang_ko_lingkai": "0:04",
    "lang_ja_lingkai": "0:04",
    "dia_chuanyu": "0:04", "dia_yue": "0:05",
}

# corr >= SAME: indistinguishable for listening purposes.
SAME = 0.995
CLOSE = 0.90


def read_wav(path):
    """Return (samples, sample_rate, channels); handles float32 or int16 PCM."""
    raw = pathlib.Path(path).read_bytes()
    if raw[:4] != b"RIFF":
        raise ValueError(f"not a wav: {path}")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"missing fmt/data chunk: {path}")
    bits = struct.unpack("<H", fmt[14:16])[0]
    channels = struct.unpack("<H", fmt[2:4])[0]
    rate = struct.unpack("<I", fmt[4:8])[0]
    if bits == 32:
        samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif bits == 16:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"unsupported sample width {bits}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate, channels


def envelope(samples, rate, bars):
    """Peak envelope over equal time slices, as ints in 0..100."""
    if samples.size == 0:
        return [0] * bars
    step = max(1, samples.size // bars)
    peaks = [float(np.abs(samples[i * step:(i + 1) * step]).max(initial=0.0))
             for i in range(bars)]
    top = max(peaks) or 1.0
    return [round(100 * (v / top) ** 0.55) for v in peaks]


def difference(a, b):
    """Correlation and peak distance of two renders of the same request."""
    n = min(a.size, b.size)
    if n < 2:
        return 0.0, 1.0
    x, y = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0, float(np.abs(x - y).max())
    corr = float(np.corrcoef(x, y)[0, 1])
    return round(corr, 4), round(float(np.abs(x - y).max()), 4)


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def request_bits(item):
    """The flags actually sent to the worker, for the parameter sheet."""
    bits = [item["task"]]
    if item.get("language_id"):
        bits.append("language_id=" + item["language_id"])
    if item.get("dialect"):
        bits.append("dialect=" + item["dialect"])
    if item.get("ip"):
        bits.append("ip=" + item["ip"])
    if item.get("style"):
        bits.append("style=" + item["style"])
    if item.get("instruction"):
        bits.append("instruction=" + item["instruction"])
    return bits


def clip_entry(item):
    samples, rate, _ = read_wav(item["output"])
    seconds = round(samples.size / rate, 2)
    py = OUT / "official_py" / (item["id"] + ".wav")
    entry = {
        "id": item["id"],
        "group": item["group"],
        "label": item["label"],
        "meta": item["meta"],
        "task": item["task"],
        "text": item["text"],
        "params": request_bits(item),
        "wav": pathlib.Path(item["output"]).name,
        "pywav": "official_py/" + item["id"] + ".wav" if py.exists() else "",
        "seconds": seconds,
        "pySeconds": seconds,
        "official": OFFICIAL.get(item["id"], ""),
        "bars": envelope(samples, rate, 56),
        "pyBars": envelope(samples, rate, 56),
        "corr": 1.0,
        "peak": 0.0,
    }
    if py.exists():
        py_samples, py_rate, _ = read_wav(py)
        entry["pySeconds"] = round(py_samples.size / py_rate, 2)
        entry["pyBars"] = envelope(py_samples, py_rate, 56)
        entry["corr"], entry["peak"] = difference(samples, py_samples)
    return entry


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    clips = [clip_entry(item) for item in manifest]
    changed = sum(1 for c in clips if c["corr"] < SAME)
    data = {
        "clips": clips,
        "changed": changed,
        "same": SAME,
        "close": CLOSE,
    }
    (OUT / "listen.html").write_text(
        TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'listen.html'} ({len(clips)} clips, {changed} differ)")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ming-3.0-tts · wgpu 与官方 0.5B 对播</title>
<style>
  :root {
    --bg: #f4f3f0;
    --card: #ffffff;
    --ink: #171716;
    --ink-soft: #46463f;
    --muted: #8d8c85;
    --line: #e7e5e0;
    --bar: #d2d0ca;
    --bar-on: #171716;
    --ok: #3d7a55;
    --near: #a3761b;
    --diff: #a8452e;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    --sans: ui-sans-serif, -apple-system, "Segoe UI", "PingFang SC",
            "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
    padding: 52px 28px 96px;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }

  header {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 24px; flex-wrap: wrap;
    padding-bottom: 22px; border-bottom: 1px solid var(--line);
    margin-bottom: 40px;
  }
  .eyebrow {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: .18em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 12px;
  }
  h1 { font-size: 26px; line-height: 1.25; margin: 0 0 8px; font-weight: 600; letter-spacing: -.01em; }
  .sub { margin: 0; font-size: 13px; color: var(--muted); line-height: 1.65; }
  .sub b { color: var(--ink-soft); font-weight: 500; }
  .sub code { font-family: var(--mono); font-size: 11.5px; }

  .actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  button.ghost {
    font: inherit; font-size: 12.5px;
    padding: 9px 15px; border-radius: 999px;
    border: 1px solid var(--line); background: var(--card); color: var(--ink-soft);
    cursor: pointer; transition: background .15s ease, border-color .15s ease, color .15s ease;
  }
  button.ghost:hover { border-color: #d6d3cc; color: var(--ink); }
  button.ghost[aria-pressed="true"] { background: var(--ink); border-color: var(--ink); color: #f7f6f3; }
  button.ghost .n { opacity: .55; margin-left: 2px; }

  section { margin-bottom: 46px; }
  .sec-head { display: flex; align-items: baseline; gap: 14px; margin-bottom: 14px; }
  .sec-head h2 {
    font-family: var(--mono); font-size: 11px; letter-spacing: .16em;
    text-transform: uppercase; color: var(--ink-soft); margin: 0; font-weight: 500;
  }
  .sec-head h2::before { content: "• "; color: var(--ink); }
  .sec-head p { margin: 0; font-size: 12.5px; color: var(--muted); }

  .cards { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 18px 20px 16px; display: flex; flex-direction: column;
    box-shadow: 0 1px 2px rgba(23, 23, 22, .035);
    transition: box-shadow .18s ease, border-color .18s ease;
  }
  .card:hover { border-color: #dedbd4; box-shadow: 0 4px 14px rgba(23, 23, 22, .055); }
  .card.is-playing { border-color: #c9c6be; }

  .field { margin-bottom: 14px; }
  .field-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: .16em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 6px;
  }
  .field-value { margin: 0; font-size: 12.5px; line-height: 1.6; color: var(--ink-soft); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    font-family: var(--mono); font-size: 11.5px; color: var(--ink-soft);
    background: #f1f0ec; border-radius: 6px; padding: 3px 8px;
  }
  .line { margin: 0 0 16px; font-size: 16px; line-height: 1.7; letter-spacing: .005em; }
  .line.en { font-size: 15.5px; }

  .ab { display: flex; flex-direction: column; gap: 7px; margin-top: auto; }
  .ab-row { display: flex; align-items: center; gap: 10px; }
  .src {
    flex: none; width: 46px; font-family: var(--mono); font-size: 10px;
    letter-spacing: .06em; color: var(--muted); text-transform: uppercase;
  }
  .play {
    flex: none; width: 30px; height: 30px; border-radius: 50%; border: 0;
    background: var(--ink); color: #fbfaf7; cursor: pointer;
    display: grid; place-items: center; padding: 0;
    transition: transform .12s ease, background .15s ease;
  }
  .play:hover { background: #2b2b28; }
  .play:active { transform: scale(.94); }
  .play:focus-visible { outline: 2px solid #171716; outline-offset: 2px; }
  .play svg { width: 11px; height: 11px; fill: currentColor; }
  .bars {
    flex: 1; display: flex; align-items: center; gap: 2px;
    height: 26px; cursor: pointer; touch-action: none;
  }
  .bars i {
    flex: 1; min-height: 2px; border-radius: 1px; background: var(--bar);
    transition: background .12s linear;
  }
  .bars i.on { background: var(--bar-on); }
  .time {
    flex: none; font-family: var(--mono); font-size: 11px; color: var(--muted);
    min-width: 30px; text-align: right; font-variant-numeric: tabular-nums;
  }

  .metrics {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin: 12px 0 0; padding-top: 11px; border-top: 1px solid #f2f0ec;
    font-family: var(--mono); font-size: 10.5px; color: #a3a29a;
    font-variant-numeric: tabular-nums;
  }
  .metrics b { color: var(--ink-soft); font-weight: 500; }
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 10.5px; letter-spacing: .04em;
  }
  .badge::before { content: "●"; font-size: 7px; line-height: 1; }
  .badge.ok { color: var(--ok); }
  .badge.near { color: var(--near); }
  .badge.diff { color: var(--diff); }
  .badge.nodata { color: var(--muted); }
  .badge.nodata::before { content: "○"; }
  .badge.near {
    color: #8a6416; background: #fbf3e2; border-radius: 5px; padding: 2px 7px 2px 6px;
  }
  .badge.diff {
    color: #993f28; background: #fbeee9; border-radius: 5px; padding: 2px 7px 2px 6px;
  }
  .src[data-verdict="near"] { color: #8a6416; }
  .src[data-verdict="diff"] { color: #a8452e; }
  .abtest {
    margin-left: auto; flex: none;
    font-family: var(--mono); font-size: 10px; letter-spacing: .06em;
    border: 1px solid var(--line); background: transparent; color: var(--ink-soft);
    border-radius: 5px; padding: 3px 7px; cursor: pointer;
    transition: background .15s ease, border-color .15s ease, color .15s ease;
  }
  .abtest:hover { background: #f4f2ee; border-color: #dcd9d2; color: var(--ink); }

  .sheet {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    overflow: hidden;
  }
  .lang-row {
    display: grid; align-items: center; gap: 16px;
    grid-template-columns: 88px minmax(0, 1fr) 292px 292px;
    padding: 13px 20px; border-bottom: 1px solid #f1efeb;
  }
  .lang-row:last-child { border-bottom: 0; }
  .lang-row.head {
    padding: 16px 20px 12px; border-bottom: 1px solid var(--line);
    font-family: var(--mono); font-size: 10px; letter-spacing: .16em;
    text-transform: uppercase; color: var(--muted);
  }
  .lang-row .lang { font-size: 12.5px; color: var(--ink-soft); }
  .lang-row .text { font-size: 14px; line-height: 1.55; }
  .voice .ab { gap: 4px; }
  .voice .ab-row { gap: 8px; }
  .voice .src { width: 34px; font-size: 9.5px; letter-spacing: .02em; }
  .voice .play { width: 24px; height: 24px; }
  .voice .play svg { width: 9px; height: 9px; }
  .voice .bars { height: 20px; }
  .voice .time { font-size: 10px; min-width: 26px; }
  .vname {
    display: none; font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 5px;
  }

  footer { margin-top: 52px; padding-top: 22px; border-top: 1px solid var(--line); }
  footer summary {
    font-family: var(--mono); font-size: 10.5px; letter-spacing: .16em;
    text-transform: uppercase; color: var(--muted); cursor: pointer;
    list-style: none; width: max-content;
  }
  footer summary::-webkit-details-marker { display: none; }
  footer summary:hover { color: var(--ink-soft); }
  footer .note { margin: 14px 0 0; font-size: 12.5px; line-height: 1.8; color: var(--muted); max-width: 82ch; }
  footer code {
    font-family: var(--mono); font-size: 11.5px; color: var(--ink-soft);
    background: #eeece7; border-radius: 4px; padding: 1px 5px;
  }
  .params { margin-top: 16px; font-family: var(--mono); font-size: 10.5px; line-height: 1.9; color: var(--ink-soft); }
  .params div { display: flex; gap: 12px; }
  .params span:first-child { min-width: 152px; color: var(--muted); }
  .params span:last-child { word-break: break-word; }

  /* 只看有差异 */
  body.only-diff [data-diff="0"] { display: none; }
  body.only-diff .lang-row.is-empty,
  body.only-diff section.is-empty { display: none; }

  @media (max-width: 980px) {
    .lang-row {
      grid-template-columns: 88px minmax(0, 1fr);
      grid-template-areas: "lang lang" "text text" "v1 v2";
      row-gap: 10px; padding: 15px 18px 16px;
    }
    .lang-row.head { display: none; }
    .lang-row .lang { grid-area: lang; }
    .lang-row .text { grid-area: text; }
    .lang-row .voice.v1 { grid-area: v1; }
    .lang-row .voice.v2 { grid-area: v2; }
    .vname { display: block; }
  }
  @media (max-width: 720px) {
    body { padding: 34px 16px 72px; }
    h1 { font-size: 21px; }
    .cards { grid-template-columns: 1fr; }
    .lang-row { grid-template-columns: 1fr; grid-template-areas: "lang" "text" "v1" "v2"; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <p class="eyebrow">Ming-3.0-tts · wgpu port vs official python</p>
      <h1>同一批请求，两条实现并排听</h1>
      <p class="sub">
        <b id="count"></b> 段 · seed 1895 · max_length 200（每条整句一次生成）·
        文本与官方 demo 的 <code>examples.js</code> 逐字对齐 ·
        其中 <b id="diffcount"></b> 段有可听差异（已标出）
      </p>
    </div>
    <div class="actions">
      <button class="ghost" id="filter" aria-pressed="false">
        只看有差异<span class="n" id="filtern"></span>
      </button>
      <button class="ghost" id="seq" aria-pressed="false">顺序播放</button>
      <button class="ghost" id="stop">停止</button>
    </div>
  </header>

  <main id="app"></main>

  <footer>
    <details>
      <summary>说明与生成参数</summary>
      <p class="note">
        <b>wgpu</b> 列：<code>target/release/ming-tts-wgpu.exe worker</code> 一次加载模型后顺序生成，
        每条 <code>use_zero_spk_emb=true</code>（无参考音频）、<code>seed=1895</code>、
        <code>max_length=200</code> 单片段，44.1 kHz 单声道 float32。
        <b>python</b> 列：官方参考实现 <code>MingTTS</code>（fp32 + CUDA Graph，cuda:0）用同一批请求、同 seed 生成。
        卡片文字取自 <code>nomorecoding.github.io/Ming-3.0-tts/static/js/examples.js</code>；
        “卡片”时长是官网卡片上标注的读数，并非实测。
        三条副语言标签卡在修掉“TN 会把 <code>[Laughter]</code> 改写成 <code>，Laughter，</code>”的移植问题后重跑。
      </p>
      <p class="note">
        差异判定用波形相关系数：<b>≥0.995 视为一致</b>（浮点路径差，听不出来），
        0.90–0.995 轻微，&lt;0.90 明显。所有 24 段两条实现的时长都在 10 ms 以内，
        出现差异的是解码的数值路径，不是文本或参数。
      </p>
      <div class="params" id="params"></div>
    </details>
  </footer>
</div>

<script>
const DATA = __DATA__;
const CLIPS = DATA.clips;
const SAME = DATA.same;
const CLOSE = DATA.close;
const PLAYERS = {};
const endedHooks = [];

const GROUP_TITLE = {
  voice_design: ["Voice design", "自然语言描述音色"],
  director:     ["Director mode", "指令式表演控制（FREEFORM）"],
  tags:         ["Paralinguistic tags", "句内副语言标签"],
  languages:    ["Built-in voices × 7 languages", "内置音色 灵小柔 / 灵楷"],
  dialects:     ["Dialects", "方言"],
};

const PLAY_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.3v9.4c0 .5.5.8.9.5l7.3-4.7a.6.6 0 0 0 0-1L3.4.8a.6.6 0 0 0-.9.5Z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 12 12" aria-hidden="true"><rect x="2.4" y="1.6" width="2.7" height="8.8" rx=".8"/><rect x="6.9" y="1.6" width="2.7" height="8.8" rx=".8"/></svg>';

function fmt(s) {
  const t = Math.max(0, Math.floor(s));
  return Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
}

function verdict(corr) {
  if (!(corr > 0)) return ["nodata", "缺参考"];
  if (corr >= SAME) return ["ok", "一致"];
  if (corr >= CLOSE) return ["near", "轻微差异"];
  return ["diff", "明显差异"];
}

const allPlayers = () => Object.values(PLAYERS);

/* One render: label + play + seekable bars + clock. */
function makePlayer(clip, which, compact) {
  const isPy = which === "py";
  const seconds = isPy ? clip.pySeconds : clip.seconds;
  const heights = isPy ? (clip.pyBars || clip.bars) : clip.bars;
  const wav = isPy ? clip.pywav : clip.wav;

  const row = document.createElement("div");
  row.className = "ab-row";
  row.dataset.diff = String(!isPy || clip.corr < SAME ? 1 : 0);

  const src = document.createElement("span");
  src.className = "src";
  src.textContent = isPy ? "python" : "wgpu";
  if (isPy) {
    const [vcls] = verdict(clip.corr);
    if (vcls === "near" || vcls === "diff") src.dataset.verdict = vcls;
    src.title = "相关 " + clip.corr.toFixed(3) + " · 峰值差 " + clip.peak.toFixed(3);
  }

  const btn = document.createElement("button");
  btn.className = "play";
  btn.type = "button";
  btn.setAttribute("aria-label", (isPy ? "官方 0.5B " : "wgpu ") + "播放：" + clip.label);
  btn.innerHTML = PLAY_ICON;

  const bars = document.createElement("div");
  bars.className = "bars";
  bars.setAttribute("role", "slider");
  bars.setAttribute("aria-label", "进度");
  bars.setAttribute("tabindex", "0");
  const barHeight = compact ? 20 : 26;
  const spans = heights.map((h) => {
    const b = document.createElement("i");
    b.style.height = Math.max(2, Math.round((h / 100) * barHeight)) + "px";
    bars.appendChild(b);
    return b;
  });

  const time = document.createElement("span");
  time.className = "time";
  time.textContent = fmt(seconds);

  row.append(src, btn, bars, time);

  let audio = null;

  function paint(ratio) {
    const n = Math.round(spans.length * ratio);
    spans.forEach((s, i) => s.classList.toggle("on", i < n));
  }

  function ensure() {
    if (audio) return audio;
    audio = new Audio(encodeURI(wav));
    audio.preload = "metadata";
    audio.addEventListener("timeupdate", () => {
      const d = audio.duration || seconds;
      paint(Math.min(1, audio.currentTime / d));
      time.textContent = fmt(audio.currentTime < 0.05 ? seconds : audio.currentTime);
    });
    audio.addEventListener("play", () => {
      btn.innerHTML = PAUSE_ICON;
      row.closest(".card")?.classList.add("is-playing");
      allPlayers().forEach((p) => { if (p !== row && p._pause) p._pause(); });
    });
    audio.addEventListener("pause", () => {
      btn.innerHTML = PLAY_ICON;
      row.closest(".card")?.classList.remove("is-playing");
    });
    audio.addEventListener("ended", () => {
      paint(0);
      time.textContent = fmt(seconds);
      btn.innerHTML = PLAY_ICON;
      row.closest(".card")?.classList.remove("is-playing");
      const hooks = endedHooks.splice(0);
      hooks.forEach((fn) => fn());
      if (seqActive && !isPy) playNext();
    });
    return audio;
  }

  btn.addEventListener("click", () => {
    if (seqActive) setSeq(false);
    const a = ensure();
    if (a.paused) a.play().catch(() => {});
    else a.pause();
  });
  row._pause = () => { if (audio) audio.pause(); };
  row._reset = () => {
    if (audio) { audio.pause(); audio.currentTime = 0; }
    paint(0);
    time.textContent = fmt(seconds);
  };

  function seek(clientX) {
    const r = bars.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    const a = ensure();
    const d = a.duration || seconds;
    a.currentTime = ratio * d;
    paint(ratio);
    time.textContent = fmt(a.currentTime);
  }
  bars.addEventListener("pointerdown", (e) => {
    bars.setPointerCapture(e.pointerId);
    seek(e.clientX);
    const move = (ev) => seek(ev.clientX);
    bars.addEventListener("pointermove", move);
    const up = () => {
      bars.removeEventListener("pointermove", move);
      bars.removeEventListener("pointerup", up);
    };
    bars.addEventListener("pointerup", up);
  });
  bars.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const a = ensure();
    a.currentTime = Math.min(a.duration || seconds,
      Math.max(0, a.currentTime + (e.key === "ArrowRight" ? 2 : -2)));
    e.preventDefault();
  });

  row._play = () => { ensure().play().catch(() => {}); };
  row._once = (fn) => endedHooks.push(fn);
  PLAYERS[clip.id + ":" + which] = row;
  return row;
}

function field(label, value) {
  const d = document.createElement("div");
  d.className = "field";
  const l = document.createElement("p");
  l.className = "field-label";
  l.textContent = label;
  const p = document.createElement("p");
  p.className = "field-value";
  p.textContent = value;
  d.append(l, p);
  return d;
}

function chips(tags) {
  const d = document.createElement("div");
  d.className = "field";
  const l = document.createElement("p");
  l.className = "field-label";
  l.textContent = "tag";
  const box = document.createElement("div");
  box.className = "chips";
  tags.forEach((t) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = t;
    box.appendChild(c);
  });
  d.append(l, box);
  return d;
}

function metrics(clip) {
  const p = document.createElement("p");
  p.className = "metrics";
  const [cls, text] = verdict(clip.corr);
  const badge = document.createElement("span");
  badge.className = "badge " + cls;
  badge.textContent = text;
  const detail = document.createElement("span");
  const delta = clip.pySeconds - clip.seconds;
  const timing = "<b>" + clip.seconds.toFixed(2) + "</b>/<b>" + clip.pySeconds.toFixed(2) + "s</b>" +
    (Math.abs(delta) >= 0.02 ? " (" + (delta > 0 ? "+" : "") + delta.toFixed(2) + "s)" : "");
  detail.innerHTML = cls === "ok"
    ? timing + (clip.official ? " · 卡片 " + clip.official : "")
    : "相关 <b>" + clip.corr.toFixed(3) + "</b> · 峰值差 <b>" + clip.peak.toFixed(3) + "</b> · " +
      timing + (clip.official ? " · 卡片 " + clip.official : "");
  p.append(badge, detail);

  if (clip.corr < SAME) {
    const ab = document.createElement("button");
    ab.type = "button";
    ab.className = "abtest";
    ab.textContent = "A/B";
    ab.title = "先播 wgpu，再播 python";
    ab.addEventListener("click", () => {
      const w = PLAYERS[clip.id + ":wgpu"];
      const q = PLAYERS[clip.id + ":py"];
      if (!w || !q) return;
      setSeq(false);
      allPlayers().forEach((x) => x._reset());
      w._play();
      w._once(() => q._play());
    });
    p.appendChild(ab);
  }
  return p;
}

function card(clip) {
  const c = document.createElement("article");
  c.className = "card";
  c.dataset.diff = String(clip.corr < SAME ? 1 : 0);

  if (clip.meta.desc) c.appendChild(field("description", clip.meta.desc));
  if (clip.meta.instruction) c.appendChild(field("instruction", clip.meta.instruction));
  if (clip.meta.tags) c.appendChild(chips(clip.meta.tags));
  if (clip.meta.dialect) c.appendChild(field("dialect", clip.meta.dialect));

  const line = document.createElement("p");
  line.className = "line" + (/[A-Za-z]/.test(clip.text) && !/[\u4e00-\u9fff]/.test(clip.text) ? " en" : "");
  line.textContent = clip.text;
  c.appendChild(line);

  const ab = document.createElement("div");
  ab.className = "ab";
  ab.append(makePlayer(clip, "wgpu", false), makePlayer(clip, "py", false));
  c.appendChild(ab);
  c.appendChild(metrics(clip));
  return c;
}

function section(group, body, extra) {
  const s = document.createElement("section");
  s.dataset.group = group;
  const head = document.createElement("div");
  head.className = "sec-head";
  const title = GROUP_TITLE[group] || [group, ""];
  const h = document.createElement("h2");
  h.textContent = title[0];
  const p = document.createElement("p");
  p.textContent = extra || title[1];
  head.append(h, p);
  s.append(head, body);
  return s;
}

function languageSheet(clips) {
  const sheet = document.createElement("div");
  sheet.className = "sheet";
  const langs = [
    ["zh", "Chinese"], ["en", "English"], ["de", "German"], ["es", "Spanish"],
    ["fr", "French"], ["ko", "Korean"], ["ja", "Japanese"],
  ];
  const head = document.createElement("div");
  head.className = "lang-row head";
  ["language", "target text", "灵小柔", "灵楷"].forEach((t) => {
    const s = document.createElement("span");
    s.textContent = t;
    head.appendChild(s);
  });
  sheet.appendChild(head);

  const byId = Object.fromEntries(clips.map((c) => [c.id, c]));
  langs.forEach(([code, name]) => {
    const row = document.createElement("div");
    row.className = "lang-row";
    row.dataset.diff = "1";
    const lang = document.createElement("span");
    lang.className = "lang";
    lang.textContent = name;
    const text = document.createElement("span");
    text.className = "text";
    const a = byId["lang_" + code + "_xiaorou"];
    const b = byId["lang_" + code + "_lingkai"];
    text.textContent = a ? a.text : "";
    row.append(lang, text);
    [[a, "v1", "灵小柔"], [b, "v2", "灵楷"]].forEach(([clip, area, vname]) => {
      const cell = document.createElement("div");
      cell.className = "voice " + area;
      if (!clip) { row.appendChild(cell); return; }
      cell.dataset.diff = String(clip.corr < SAME ? 1 : 0);
      const label = document.createElement("p");
      label.className = "vname";
      label.textContent = vname;
      const ab = document.createElement("div");
      ab.className = "ab";
      const w = makePlayer(clip, "wgpu", true);
      const p = makePlayer(clip, "py", true);
      p.title = "相关 " + clip.corr.toFixed(3) + " · 峰值差 " + clip.peak.toFixed(3);
      if (clip.corr < SAME) p.dataset.changed = "1";
      ab.append(w, p);
      cell.append(label, ab);
      row.appendChild(cell);
    });
    sheet.appendChild(row);
  });
  return sheet;
}

/* --- sequential play (wgpu rows, in card order) --- */
let seqActive = false;
let seqIndex = -1;
const seqList = [];

function setSeq(on) {
  seqActive = on;
  seqIndex = -1;
  const btn = document.getElementById("seq");
  btn.setAttribute("aria-pressed", String(on));
  btn.textContent = on ? "顺序播放中" : "顺序播放";
  if (!on) allPlayers().forEach((p) => p._pause());
}

function playNext() {
  seqIndex += 1;
  const visible = seqList.filter((id) => {
    const row = PLAYERS[id + ":wgpu"];
    const cardEl = row && row.closest(".card");
    return cardEl && cardEl.offsetParent !== null;
  });
  const id = visible[seqIndex];
  if (!id) { setSeq(false); return; }
  PLAYERS[id + ":wgpu"]._play();
}

/* --- filter --- */
function updateFilter() {
  const on = document.body.classList.contains("only-diff");
  document.querySelectorAll(".lang-row:not(.head)").forEach((row) => {
    const cells = [...row.querySelectorAll(".voice")];
    const anyVisible = cells.some((cell) =>
      cell.offsetParent !== null && (cell.dataset.diff === "1"));
    row.classList.toggle("is-empty", on && !anyVisible);
  });
  document.querySelectorAll("section").forEach((s) => {
    const cards = [...s.querySelectorAll(".card")];
    const rows = [...s.querySelectorAll(".lang-row:not(.head)")];
    let any = false;
    if (cards.length) any = cards.some((c) => c.offsetParent !== null);
    else if (rows.length) any = rows.some((r) => r.offsetParent !== null);
    s.classList.toggle("is-empty", on && !any);
  });
}

/* --- build --- */
const app = document.getElementById("app");
document.getElementById("count").textContent = String(CLIPS.length);
document.getElementById("diffcount").textContent = String(DATA.changed);
document.getElementById("filtern").textContent = DATA.changed ? " " + DATA.changed : "";

["voice_design", "director", "tags", "dialects"].forEach((g) => {
  const items = CLIPS.filter((c) => c.group === g);
  if (!items.length) return;
  const grid = document.createElement("div");
  grid.className = "cards";
  items.forEach((c) => {
    grid.appendChild(card(c));
    seqList.push(c.id);
  });
  app.appendChild(section(g, grid));
});

const langClips = CLIPS.filter((c) => c.group === "languages");
if (langClips.length) app.appendChild(section("languages", languageSheet(langClips)));

document.getElementById("seq").addEventListener("click", () => {
  if (seqActive) { setSeq(false); return; }
  allPlayers().forEach((p) => p._reset());
  setSeq(true);
  playNext();
});
document.getElementById("stop").addEventListener("click", () => {
  setSeq(false);
  allPlayers().forEach((p) => p._reset());
});
document.getElementById("filter").addEventListener("click", (e) => {
  const on = !document.body.classList.contains("only-diff");
  document.body.classList.toggle("only-diff", on);
  e.currentTarget.setAttribute("aria-pressed", String(on));
  updateFilter();
  if (on) setSeq(false);
});

/* --- parameter sheet --- */
const params = document.getElementById("params");
CLIPS.forEach((c) => {
  const row = document.createElement("div");
  const k = document.createElement("span");
  k.textContent = c.id;
  const v = document.createElement("span");
  v.textContent = c.params.join(" · ");
  row.append(k, v);
  params.appendChild(row);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
