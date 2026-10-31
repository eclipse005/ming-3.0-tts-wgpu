#!/usr/bin/env python
"""Dump the M5 goldens (audio I/O, resampling, tokenizer, speaker embedding).

Every golden in goldens/m5_audio/ is produced here from the Python reference so
the Rust port can assert equality instead of plausibility:

  *.f32                  raw little-endian float32 payloads (C order), sizes in
                         meta.json -- avoids the unreadable F-contiguous .npy trap.
  wav_ref_*.wav          exact bytes torchaudio.save writes, for the Rust writer
                         to reproduce (all bytes except the PEAK timestamp,
                         which is wall-clock).
  ids.json               token ids for the params_test corpus and the exact
                         prompt_contract.build_input_ids sequences.
  spk_lingguang.json     CAMPlus speaker embedding (192 floats) plus the kaldi
                         fbank features it consumes.

Run (see the module docstring in RUN_LOCAL.md for the env):
  set PYTHONPATH=D:/Ming-3.0-tts/.runtime/pylibs446
  C:/Users/ADMIN/miniconda3/envs/videolingo/python.exe -I tools/dump_m5_audio.py
"""

import json
import os
import sys

sys.path.insert(0, 'D:/Ming-3.0-tts')

import numpy as np
import torch
import torchaudio

ROOT = 'D:/Ming-3.0-tts'
OUT = os.path.join(ROOT, 'wgpu', 'goldens', 'm5_audio')
PROMPT_WAV = os.path.join(ROOT, 'data/voice_presets/voice_audio/lingguang.wav')
SECOND_WAV = os.path.join(ROOT, 'data/voice_presets/voice_audio/db30.wav')
MODEL_DIR = os.path.join(ROOT, 'Ming-3.0-tts-0.5b')

NL = chr(10)
TAB = chr(9)

meta = {'f32': {}, 'json': [], 'wav': []}


def save_f32(name, arr, **info):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    path = os.path.join(OUT, name + '.f32')
    with open(path, 'wb') as fh:
        fh.write(a.astype('<f4').tobytes())
    entry = {'shape': list(a.shape), 'count': int(a.size)}
    entry.update(info)
    meta['f32'][name + '.f32'] = entry
    return a


def save_json(name, obj):
    path = os.path.join(OUT, name + '.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write(NL)
    meta['json'].append(name + '.json')


# ---------------------------------------------------------------- audio input
def load_mono(path):
    """Exactly what ming_tts._load_prompt does before any resampling."""
    waveform, sample_rate = torchaudio.load(path, backend='soundfile')
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def dump_audio_inputs():
    ling, ling_sr = load_mono(PROMPT_WAV)
    assert ling_sr == 48000, ling_sr
    save_f32('lingguang_mono', ling[0].numpy(), sample_rate=ling_sr,
             source='torchaudio.load(soundfile) then mean(dim=0)')
    db30, db30_sr = load_mono(SECOND_WAV)
    assert db30_sr == 16000, db30_sr
    save_f32('db30_mono', db30[0].numpy(), sample_rate=db30_sr,
             source='torchaudio.load(soundfile) then mean(dim=0)')

    # A stereo file proves the averaging path and that the reader averages the
    # channels instead of taking channel 0.
    left = ling[0].numpy()
    right = np.roll(left, 1) * 0.5
    stereo = np.stack([left, right], axis=0).astype(np.float32)
    stereo_path = os.path.join(OUT, 'stereo_input.wav')
    torchaudio.save(stereo_path, torch.from_numpy(stereo), sample_rate=48000)
    reloaded, _ = torchaudio.load(stereo_path, backend='soundfile')
    assert reloaded.shape[0] == 2
    save_f32('stereo_mean', reloaded.mean(dim=0).numpy(), sample_rate=48000,
             source='mean(dim=0) of the 2-channel wav')
    meta['wav'].append({'name': 'stereo_input.wav', 'writer': 'torchaudio.save',
                        'channels': 2, 'sample_rate': 48000})
    return ling, ling_sr, db30, db30_sr


def dump_resample(ling, ling_sr, db30, db30_sr):
    def resample(x, orig, new):
        return torchaudio.transforms.Resample(orig, new)(x)

    # Real prompt: native 48 kHz -> the 16 kHz speaker path and the 44.1 kHz
    # model path.
    save_f32('lingguang_48000_to_16000',
             resample(ling, ling_sr, 16000)[0].numpy(), orig=48000, new=16000)
    save_f32('lingguang_48000_to_44100',
             resample(ling, ling_sr, 44100)[0].numpy(), orig=48000, new=44100)
    # 16 kHz preset -> model rate.
    save_f32('db30_16000_to_44100',
             resample(db30, db30_sr, 44100)[0].numpy(), orig=16000, new=44100)

    # Synthetic material: a linear chirp plus a few strong tones, so the
    # anti-aliasing roll-off is exercised, then resampled both ways.
    sr = 44100
    n = int(0.5 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    freq = 120.0 + (7000.0 - 120.0) * t / t[-1]
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    chirp = (0.55 * np.sin(phase)
             + 0.3 * np.sin(2.0 * np.pi * 1500.0 * t)
             + 0.1 * np.sin(2.0 * np.pi * 9000.0 * t)
             + 0.05 * np.sin(2.0 * np.pi * 19000.0 * t))
    chirp = chirp.astype(np.float32)
    save_f32('chirp_44100', chirp, sample_rate=sr,
             source='chirp 120->7000 Hz plus 1.5/9/19 kHz tones, 0.5 s')
    x = torch.from_numpy(chirp)[None]
    save_f32('chirp_44100_to_16000', resample(x, 44100, 16000)[0].numpy(),
             orig=44100, new=16000)

    sr2 = 22050
    n2 = int(0.25 * sr2)
    t2 = np.arange(n2, dtype=np.float64) / sr2
    sine = (0.9 * np.sin(2.0 * np.pi * 440.0 * t2)).astype(np.float32)
    save_f32('sine_22050', sine, sample_rate=sr2, source='440 Hz, 0.25 s')
    save_f32('sine_22050_to_44100',
             resample(torch.from_numpy(sine)[None], 22050, 44100)[0].numpy(),
             orig=22050, new=44100)
    # Same-rate case must be the identity, as torchaudio returns the input.
    save_f32('sine_22050_to_22050',
             resample(torch.from_numpy(sine)[None], 22050, 22050)[0].numpy(),
             orig=22050, new=22050)


# --------------------------------------------------------------- wav writing
def dump_wav_refs(ling, ling_sr):
    rng = np.random.default_rng(20240531)

    cases = {}
    # 1 channel, 44.1 kHz, ordinary speech-like floats.
    cases['wav_ref_1ch_44100'] = (
        (0.6 * rng.standard_normal(1000)).astype(np.float32), 44100)
    # 2 channels: one PEAK entry per channel, different peak positions.
    stereo = np.stack([
        (0.4 * rng.standard_normal(500)).astype(np.float32),
        (0.4 * rng.standard_normal(500)).astype(np.float32),
    ])
    stereo[1, 17] = -0.875
    cases['wav_ref_2ch_16000'] = (stereo, 16000)
    # All zeros: PEAK still present with a zero peak and position 0.
    cases['wav_ref_zeros_44100'] = (np.zeros(64, dtype=np.float32), 44100)
    # Out-of-range values: libsndfile writes float payloads verbatim.
    cases['wav_ref_clip_44100'] = (
        np.array([0.0, 1.5, -2.25, 1.5, 3.0, -0.5], dtype=np.float32), 44100)
    # Ties on abs(max): the first index must win in the PEAK position.
    cases['wav_ref_tie_44100'] = (
        np.array([0.1, -0.9, 0.5, 0.9, -0.2], dtype=np.float32), 44100)
    # A single sample, and a short odd-length payload.
    cases['wav_ref_one_44100'] = (np.array([0.25], dtype=np.float32), 44100)
    cases['wav_ref_odd_16000'] = (
        np.array([0.125, -0.25, 0.5], dtype=np.float32), 16000)

    for name, (arr, sr) in cases.items():
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if tensor.ndim == 1:
            tensor = tensor[None]
        save_f32(name + '_in', np.ascontiguousarray(arr), sample_rate=sr,
                 layout='(channels, samples)' if arr.ndim == 2 else '(samples,)')
        path = os.path.join(OUT, name + '.wav')
        torchaudio.save(path, tensor, sample_rate=sr)
        meta['wav'].append({'name': name + '.wav', 'writer': 'torchaudio.save',
                            'channels': int(tensor.shape[0]), 'sample_rate': sr,
                            'samples': int(tensor.shape[1])})

    # The real prompt re-saved as float32 at its native rate: the end-to-end
    # read -> write -> byte-compare case.
    save_f32('lingguang_mono_48000_for_wav', ling[0].numpy(), sample_rate=ling_sr)
    path = os.path.join(OUT, 'wav_ref_lingguang_48000.wav')
    torchaudio.save(path, ling, sample_rate=ling_sr)
    meta['wav'].append({'name': 'wav_ref_lingguang_48000.wav',
                        'writer': 'torchaudio.save', 'channels': 1,
                        'sample_rate': ling_sr, 'samples': int(ling.shape[-1])})


# ----------------------------------------------------------------- tokenizer
def dump_tokenizer():
    from prompt_contract import (DENSE_TOKEN_IDS, DENSE_SYSTEM_PREFIX,
                                 DENSE_USER_PREFIX, DENSE_USER_SUFFIX,
                                 DENSE_ASSISTANT_PREFIX, DENSE_SPEAKER_START,
                                 DENSE_SPEAKER_PAD, DENSE_SPEAKER_END,
                                 DENSE_INSTRUCTION_EOS, TEXT_INPUT_PREFIX,
                                 AUDIO_START, AUDIO_PATCH, build_input_ids)
    from transformers import AutoTokenizer
    import test_params
    import tts_tasks

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    def enc(text):
        return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]

    corpus = []

    def add(name, text):
        corpus.append({'name': name, 'text': text, 'ids': enc(text)})

    for name, task, text, kwargs in test_params.SPEC:
        add('params_test/%s/text' % name, text)
    for name, task, text, kwargs in test_params.SPEC:
        for key, value in sorted((kwargs.get('controls') or {}).items()):
            add('params_test/%s/control_%s' % (name, key), value)
        for key in ('instruction', 'prompt_text', 'language_id'):
            if kwargs.get(key):
                add('params_test/%s/%s' % (name, key), kwargs[key])
    add('prompt/default_tts', tts_tasks.DEFAULT_TTS_PROMPT)
    add('prompt/target_language_en', tts_tasks.build_tts_prompt('en'))
    add('contract/system_prefix', DENSE_SYSTEM_PREFIX)
    add('contract/user_prefix', DENSE_USER_PREFIX)
    add('contract/user_suffix', DENSE_USER_SUFFIX)
    add('contract/assistant_prefix', DENSE_ASSISTANT_PREFIX)
    add('contract/speaker_start', DENSE_SPEAKER_START)
    add('contract/speaker_pad', DENSE_SPEAKER_PAD)
    add('contract/speaker_end', DENSE_SPEAKER_END)
    add('contract/instruction_eos', DENSE_INSTRUCTION_EOS)
    add('contract/text_input_prefix', TEXT_INPUT_PREFIX)
    add('contract/audio_start', AUDIO_START)
    add('contract/audio_patch', AUDIO_PATCH)
    add('contract/speaker_1_label', '  speaker_1:')
    add('contract/ascii',
        'The weather is lovely today, shall we take a walk in the park?')
    add('contract/ja', '今日はとても良い天気ですね、一緒に散歩しましょう。')
    add('contract/ko', '오늘 날씨가 참 좋네요, 같이 공원에 가요.')
    add('contract/de', 'Guten Tag, das Wetter ist heute wirklich wunderbar.')
    add('contract/emoji_ws',
        '  ' + chr(233) + chr(232) + chr(234) + '  ' + NL + TAB
        + ' ' + chr(20320) + chr(22909) + ' ')
    add('contract/empty', '')

    tokens = []
    for token, want in sorted(DENSE_TOKEN_IDS.items()):
        tokens.append({'token': token,
                       'id': int(tokenizer.convert_tokens_to_ids(token)),
                       'encode': enc(token), 'expected_id': want})

    # Exact build_input_ids sequences the pipeline feeds the model.
    text_a = '今天天气真不错，我们一起去公园散步吧。'
    prompt = tts_tasks.DEFAULT_TTS_PROMPT
    prompt_text = '诶，你今天有空吗？我们能不能聊聊这件事啊？'
    cases = [
        ('tts_bare', dict(prompt=prompt, text=text_a)),
        ('tts_one_speaker', dict(prompt=prompt, text=text_a, num_speakers=1)),
        ('tts_two_speakers', dict(prompt=prompt, text=text_a, num_speakers=2)),
        ('tts_freeform_instruction', dict(
            prompt=prompt, text='欢迎收看今天晚间新闻。',
            instruction='像新闻联播主播一样字正腔圆地播报这句话。')),
        ('tts_structured_instruction_basic', dict(
            prompt=prompt, text='雨后的山谷很安静，风吹过松林。',
            instruction=tts_tasks.build_task_instruction('BASIC', speed='慢速'))),
        ('tts_structured_instruction_ip', dict(
            prompt=prompt, text='哇，这次真的成功啦！',
            instruction=tts_tasks.build_task_instruction('IP', ip='灵汐月_高兴'))),
        ('tts_prompt_text_1_step', dict(
            prompt=prompt, text=text_a, prompt_text=prompt_text,
            prompt_latent_steps=1)),
        ('tts_prompt_text_13_steps', dict(
            prompt=prompt, text=text_a, prompt_text=prompt_text,
            prompt_latent_steps=13)),
        ('tts_prompt_text_40_steps', dict(
            prompt=prompt, text=text_a, prompt_text=prompt_text,
            prompt_latent_steps=40)),
        ('tts_everything', dict(
            prompt=tts_tasks.build_tts_prompt('ja'),
            text='今日はとても良い天気ですね、一緒に散歩しましょう。',
            num_speakers=1,
            instruction=tts_tasks.build_task_instruction('DIALECT',
                                                         dialect='cantonese'),
            prompt_text=prompt_text,
            prompt_latent_steps=7)),
        ('tts_bgm_no_text_prefix', dict(
            prompt=prompt,
            text='Genre: Ambient' + NL + 'Mood: calm' + NL + 'Instrument: piano'
                 + NL + 'Theme: rain' + NL + 'Duration: 30' + NL)),
    ]
    built = []
    for name, kwargs in cases:
        ids, spk_pos, latent_pos = build_input_ids(tokenizer, 'dense', **kwargs)
        built.append({
            'name': name,
            'args': {k: v for k, v in kwargs.items()},
            'input_ids': [int(i) for i in ids],
            'speaker_positions': [int(i) for i in spk_pos],
            'latent_position': int(latent_pos),
        })

    save_json('ids', {
        'model_dir': MODEL_DIR,
        'tokenizer_class': type(tokenizer).__name__,
        'corpus': corpus,
        'tokens': tokens,
        'build_input_ids': built,
    })
    print('tokenizer: %d corpus entries, %d build_input_ids cases'
          % (len(corpus), len(built)))


# ------------------------------------------------------------ speaker embed
def dump_speaker(ling, ling_sr):
    import onnxruntime
    import torchaudio.compliance.kaldi as kaldi

    wav16 = torchaudio.transforms.Resample(ling_sr, 16000)(ling)
    feat = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat_norm = feat - feat.mean(dim=0, keepdim=True)

    # Every intermediate stage, so the Rust port can be checked pointwise
    # instead of only through the 192-dim embedding.
    window = kaldi._feature_window_function(
        'povey', 400, 0.42, wav16.device, wav16.dtype)
    save_f32('spk_window_povey', window.numpy(), size=400)
    fft_input, _ = kaldi._get_window(
        wav16[0], 512, 400, 160, 'povey', 0.42, True, True, 1.0, 0.0, True,
        0.97)
    save_f32('spk_fft_input', fft_input.numpy(),
             source='strided frames after DC removal, preemphasis, povey '
                    'window and zero pad to 512')
    spectrum = torch.fft.rfft(fft_input).abs().pow(2.0)
    save_f32('spk_power_spectrum', spectrum.numpy(),
             source='rfft(frame).abs().pow(2)')
    mel, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0,
                                 1.0)
    mel_padded = torch.nn.functional.pad(mel, (0, 1), mode='constant',
                                        value=0)
    save_f32('spk_mel_padded', mel_padded.numpy(),
             source='get_mel_banks(80, 512, 16000, low=20) padded with a zero '
                    'column')
    fbank_raw = torch.mm(spectrum, mel_padded.T)
    fbank_raw = torch.max(fbank_raw, kaldi._get_epsilon(
        wav16.device, wav16.dtype)).log()
    save_f32('spk_fbank_raw', fbank_raw.numpy(),
             source='log mel energies before the per-bin mean subtraction')

    sess = onnxruntime.InferenceSession(
        os.path.join(MODEL_DIR, 'campplus.onnx'),
        providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    assert len(inputs) == 1, inputs
    input_meta = {
        'name': inputs[0].name,
        'shape': [str(d) for d in inputs[0].shape],
        'type': inputs[0].type,
        'outputs': [{'name': o.name, 'shape': [str(d) for d in o.shape],
                     'type': o.type} for o in sess.get_outputs()],
    }
    out = sess.run(None, {inputs[0].name: feat_norm.unsqueeze(0).numpy()})[0]
    emb = np.asarray(out).reshape(-1)

    save_f32('spk_lingguang_fbank', feat_norm.numpy(),
             source='kaldi.fbank(80 bins, dither=0, sr=16000) minus the '
                    'per-bin mean',
             frames=int(feat.shape[0]), bins=int(feat.shape[1]))
    save_json('spk_lingguang', {
        'wav': PROMPT_WAV,
        'resampled_to': 16000,
        'num_samples_16k': int(wav16.shape[-1]),
        'fbank_frames': int(feat.shape[0]),
        'fbank_bins': int(feat.shape[1]),
        'onnxruntime_version': onnxruntime.__version__,
        'input': input_meta,
        'embedding_shape': list(np.asarray(out).shape),
        'embedding': [float(v) for v in emb],
    })
    print('speaker: fbank %s, embedding %s, first 3 = %s'
          % (tuple(feat.shape), list(np.asarray(out).shape), emb[:3]))


def main():
    os.makedirs(OUT, exist_ok=True)
    print('torch %s torchaudio %s numpy %s'
          % (torch.__version__, torchaudio.__version__, np.__version__))
    ling, ling_sr, db30, db30_sr = dump_audio_inputs()
    dump_resample(ling, ling_sr, db30, db30_sr)
    dump_wav_refs(ling, ling_sr)
    dump_tokenizer()
    dump_speaker(ling, ling_sr)
    with open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
        fh.write(NL)
    print('wrote %d f32 arrays, %d json, %d wavs to %s'
          % (len(meta['f32']), len(meta['json']), len(meta['wav']), OUT))


if __name__ == '__main__':
    main()
