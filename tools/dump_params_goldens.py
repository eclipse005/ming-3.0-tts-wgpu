# Record, for every params_test case, the exact torch.randn draws the reference
# pipeline consumes. The Rust port replays them (MING_NOISE_GOLDEN) so the two
# sides can be compared sample-for-sample.
#
# Mirrors test_params.py exactly: one MingTTS instance (so the prompt-latent
# cache behaves the same), reseed 1895 before each case, same case order.
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REF_REPO = os.environ.get("MING_REF", "D:/Ming-3.0-tts")
sys.path.insert(0, REF_REPO)
os.chdir(REF_REPO)          # prompt wav paths below are relative to it

import numpy as np
import torch

from ming_tts import MingTTS

OUT = str(HERE / "goldens" / "m5_params")
PROMPT_WAV = "data/voice_presets/voice_audio/lingguang.wav"
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


def seed_all(seed=1895):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


os.makedirs(OUT, exist_ok=True)
model = MingTTS(REF_REPO + "/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
print("model loaded", flush=True)

orig_randn = torch.randn
index = {}

for name, task, text, kwargs in SPEC:
    draws = []

    def spy(*a, _d=draws, **kw):
        t = orig_randn(*a, **kw)
        _d.append(t.detach().float().cpu().numpy())
        return t

    torch.randn = spy
    seed_all()
    try:
        wav = model.synthesize(task=task, text=text, max_length=24, output_path=None, **kwargs)
    finally:
        torch.randn = orig_randn

    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    for i, arr in enumerate(draws):
        np.save(os.path.join(d, f"noise_{i:03d}.npy"), np.ascontiguousarray(arr))
    x = wav[0].float().cpu().numpy() if wav.dim() > 1 else wav.float().cpu().numpy()
    np.save(os.path.join(d, "ref_waveform.npy"), np.ascontiguousarray(x))
    index[name] = {
        "task": task,
        "n_draws": len(draws),
        "draw_shapes": [list(a.shape) for a in draws],
        "n_samples": int(x.shape[-1]),
        "seconds": round(float(x.shape[-1]) / model.sample_rate, 4),
    }
    print(f"[{name}] draws={len(draws)} len={x.shape[-1]}", flush=True)

with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=1)
print("done", len(index))
