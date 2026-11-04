# Run every params_test case through the Rust CLI and compare with the Python
# reference rendering, per case.
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]                  # this port's repo root
REF_REPO = os.environ.get("MING_REF", "D:/Ming-3.0-tts")    # the Python reference
EXE = os.environ.get("MING_EXE", str(HERE / "target" / "release" / "ming-tts-wgpu.exe"))
MODEL = os.environ.get("MING_MODEL", REF_REPO + "/Ming-3.0-tts-0.5b")
REF = REF_REPO + "/output/params_test"
OUT = REF_REPO + "/output/rust_params"
PROMPT_WAV = REF_REPO + "/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"
CLONE = dict(prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT, use_spk_emb=True)

SPEC = [
    ("A1_bare", "TTS", "今天天气真不错，我们一起去公园散步吧。", {}),
    ("A2_zero_spk", "TTS", "今天天气真不错，我们一起去公园散步吧。", dict(use_zero_spk_emb=True)),
    ("A3_clone", "TTS", "今天天气真不错，我们一起去公园散步吧。", dict(CLONE)),
    ("A4_clone_latent_only", "TTS", "今天天气真不错，我们一起去公园散步吧。",
     dict(prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT)),
    ("B1_speed_slow", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"speed": "慢速"})),
    ("B2_speed_mid", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"speed": "中速"})),
    ("B3_speed_fast", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"speed": "快速"})),
    ("B4_pitch_low", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"pitch": "低"})),
    ("B5_pitch_high", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"pitch": "高"})),
    ("B6_volume_low", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"volume": "低"})),
    ("B7_volume_high", "BASIC", "雨后的山谷很安静，风吹过松林。", dict(controls={"volume": "高"})),
    ("C1_happy", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "高兴"})),
    ("C2_sad", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "悲伤"})),
    ("C3_angry", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "愤怒"})),
    ("C4_fearful", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "恐惧"})),
    ("C5_disgusted", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "厌恶"})),
    ("C6_surprised", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "惊讶"})),
    ("C7_neutral", "EMOTION", "今天晚上吃火锅还是烧烤呢？", dict(controls={"emotion": "中性"})),
    ("D1_cantonese", "DIALECT", "今日天气真好，一齐去公园行下啦。", dict(controls={"dialect": "cantonese"})),
    ("D2_sichuanese", "DIALECT", "今天天气巴适得很，走，去公园头转一哈。",
     dict(controls={"dialect": "chuanyu dialect"})),
    ("E1_ip_lingyuechuan", "IP", "这里曾是古蜀文明的中心，沉淀了三千年的时光。",
     dict(controls={"ip": "灵阅川"})),
    ("E2_ip_lingxiyue_happy", "IP", "哇，这次真的成功啦！", dict(controls={"ip": "灵汐月_高兴"})),
    ("E3_ip_us_emma", "IP", "The weather is lovely today, shall we take a walk in the park?",
     dict(controls={"ip": "US_Emma"})),
    ("F1_japanese", "MULTILINGUAL", "今日はとても良い天気ですね、一緒に散歩しましょう。",
     dict(language_id="ja")),
    ("F2_korean", "MULTILINGUAL", "오늘 날씨가 참 좋네요, 같이 공원에 가요.", dict(language_id="ko")),
    ("F3_german", "MULTILINGUAL", "Guten Tag, das Wetter ist heute wirklich wunderbar.",
     dict(language_id="de")),
    ("F4_french", "MULTILINGUAL", "Quel beau temps aujourd'hui, allons nous promener au parc.",
     dict(language_id="fr")),
    ("F5_spanish", "MULTILINGUAL", "Hoy hace muy buen tiempo, vamos a caminar por el parque.",
     dict(language_id="es")),
    ("G1_asmr_whisper", "STYLE", "听……这个声音……是不是……很舒服……",
     dict(controls={"style": "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。"
                            "这个女性使用轻柔的普通话进行耳语，声音气音成分重。音量极低，"
                            "紧贴麦克风，语速极慢，旨在制造触发听者颅内快感的声学刺激。"})),
    ("G2_empress", "STYLE", "本宫赏你的，是恩典；本宫不给的，你断不能抢。在这后宫里，规矩重于天。",
     dict(controls={"style": "这是一个宫斗剧中的威严皇后的声音，展现了充满张力的戏剧叙事风格。"
                            "她使用雍容华贵的普通话，中年女性声音沉稳。语速雍容和缓，"
                            "但每个字都掷地有声，语气表面波澜不惊，实则暗藏锋芒和久居上位的威压。"})),
    ("H1_whisper_secret", "FREEFORM", "我跟你讲个秘密，你千万别告诉别人。",
     dict(instruction="用非常轻的气声说这句话，像在图书馆里悄悄说话。")),
    ("H2_news_anchor", "FREEFORM", "欢迎收看今天晚间新闻。",
     dict(instruction="像新闻联播主播一样字正腔圆地播报这句话。")),
    ("I1_temperature_06", "TTS", "今天天气真不错，我们一起去公园散步吧。", dict(CLONE, temperature=0.6)),
    ("I2_temperature_10", "TTS", "今天天气真不错，我们一起去公园散步吧。", dict(CLONE, temperature=1.0)),
    ("I3_cfg_10", "TTS", "今天天气真不错，我们一起去公园散步吧。", dict(CLONE, cfg=1.0)),
    ("I4_cfg_35", "TTS", "今天天气真不错，我们一起去公园散步吧。", dict(CLONE, cfg=3.5)),
]


def cli_args(name, task, text, kwargs):
    args = [EXE, "synth", "--model_path", MODEL, "--task", task, "--text", text,
            "--output", f"{OUT}/{name}.wav", "--no_text_normalization",
            "--device", os.environ.get("MING_ALIGN_DEVICE", "cuda:0")]
    c = kwargs.get("controls") or {}
    for k in ("speed", "pitch", "volume", "emotion", "dialect", "style", "ip"):
        if c.get(k):
            args += [f"--{k}", c[k]]
    if kwargs.get("language_id"):
        args += ["--language_id", kwargs["language_id"]]
    if kwargs.get("instruction"):
        args += ["--instruction", kwargs["instruction"]]
    if kwargs.get("prompt_wav_path"):
        args += ["--prompt_wav_path", kwargs["prompt_wav_path"]]
    if kwargs.get("prompt_text"):
        args += ["--prompt_text", kwargs["prompt_text"]]
    if kwargs.get("use_spk_emb"):
        args += ["--use_spk_emb"]
    if kwargs.get("use_zero_spk_emb"):
        args += ["--use_zero_spk_emb"]
    if "temperature" in kwargs:
        args += ["--temperature", str(kwargs["temperature"])]
    if "cfg" in kwargs:
        args += ["--cfg", str(kwargs["cfg"])]
    return args


def logmel(x, sr, n_mels=80, n_fft=1024, hop=256):
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    if n <= 0:
        return np.zeros((n_mels, 0))
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, n_fft), strides=(x.strides[0] * hop, x.strides[0]))
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    mel = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mel) / sr).astype(int)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return np.log(np.maximum(spec @ fb.T, 1e-8)).T


_SESS = None


def speaker_emb(path):
    """CAMPlus embedding, for the voice-conditioned cases."""
    global _SESS
    import torch
    import torchaudio
    import onnxruntime as ort
    if _SESS is None:
        _SESS = ort.InferenceSession(MODEL + "/campplus.onnx",
                                     providers=["CPUExecutionProvider"])
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    t = torch.from_numpy(x).float().unsqueeze(0)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    f = torchaudio.compliance.kaldi.fbank(t, num_mel_bins=80, dither=0.0,
                                        sample_frequency=16000)
    f = f - f.mean(dim=0, keepdim=True)
    return _SESS.run(None, {"input": f.unsqueeze(0).numpy()})[0][0]


def compare(a, b):
    x, sr = sf.read(a, dtype="float32")
    y, _ = sf.read(b, dtype="float32")
    n = min(len(x), len(y))
    rel = float(np.linalg.norm(x[:n] - y[:n]) / max(np.linalg.norm(x[:n]), 1e-12))
    mx, my = logmel(x, sr), logmel(y, sr)
    k = min(mx.shape[1], my.shape[1])
    ca = mx[:, :k] - mx[:, :k].mean()
    cb = my[:, :k] - my[:, :k].mean()
    corr = float((ca * cb).sum() / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    out = dict(ref_samples=len(x), got_samples=len(y), rel_l2=rel, logmel_corr=corr,
               dur_ratio=len(y) / max(len(x), 1))
    try:
        ea, eb = speaker_emb(a), speaker_emb(b)
        out["spk_cos"] = float(np.dot(ea, eb) /
                               (np.linalg.norm(ea) * np.linalg.norm(eb)))
    except Exception:
        out["spk_cos"] = None
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or None
    rows = []
    for name, task, text, kwargs in SPEC:
        if only and name not in only:
            continue
        p = subprocess.run(cli_args(name, task, text, kwargs),
                           capture_output=True, text=True)
        if p.returncode != 0:
            rows.append(dict(name=name, task=task, ok=False,
                             err=(p.stderr or p.stdout).strip().splitlines()[-1:]))
            print(f"[FAIL] {name}: {rows[-1]['err']}", flush=True)
            continue
        m = compare(f"{REF}/{name}.wav", f"{OUT}/{name}.wav")
        m.update(name=name, task=task, ok=True)
        rows.append(m)
        spk = m["spk_cos"]
        print(f"[ok] {name:24s} dur {m['dur_ratio']:.4f} rel={m['rel_l2']:.2e} "
              f"mel={m['logmel_corr']:.5f} spk={spk:.4f}" if spk is not None else
              f"[ok] {name:24s} dur {m['dur_ratio']:.4f} rel={m['rel_l2']:.2e} "
              f"mel={m['logmel_corr']:.5f}", flush=True)

    with open(f"{OUT}/report.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    ok = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok)}/{len(rows)} ran")
    if ok:
        exact = [r for r in ok if r["ref_samples"] == r["got_samples"]]
        print(f"same sample count: {len(exact)}/{len(ok)}")
        print(f"logmel corr  min={min(r['logmel_corr'] for r in ok):.6f} "
              f"median={float(np.median([r['logmel_corr'] for r in ok])):.6f}")
        print(f"rel_l2       max={max(r['rel_l2'] for r in ok):.3e}")


if __name__ == "__main__":
    main()