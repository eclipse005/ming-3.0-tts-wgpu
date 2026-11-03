"""Ad-hoc stage timing for the Python reference (flow / agg / VAE / LLM).

Not part of the port. Wraps the three hot calls with CUDA events so the numbers
are GPU time, not host wall time. Run from D:/Ming-3.0-tts with the videolingo
python (needs torch + the model directory).

    python wgpu/tools/_py_stage_timing.py [--runs 2]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, r"D:/Ming-3.0-tts")
from ming_tts import MingTTS  # noqa: E402

TEXT = "雨后的山谷很安静，风吹过松林，带着一点湿润的草木气息。"
PROMPT_WAV = r"D:/Ming-3.0-tts/data/voice_presets/voice_audio/lingguang.wav"
PROMPT_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--model_path", default=r"D:/Ming-3.0-tts/Ming-3.0-tts-0.5b")
    args = ap.parse_args()

    t0 = time.perf_counter()
    m = MingTTS(args.model_path, device="cuda:0", text_normalization=False)
    print(f"model_load={time.perf_counter() - t0:.2f}s")

    stats = {"flow": 0.0, "agg": 0.0, "vae": 0.0, "flow_n": 0, "agg_n": 0, "vae_n": 0}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def timed(name, fn):
        def wrapper(*a, **kw):
            ev[0].record()
            out = fn(*a, **kw)
            ev[1].record()
            ev[1].synchronize()
            if name in stats:
                stats[name] += ev[0].elapsed_time(ev[1])
                stats[name + "_n"] += 1
            return out
        return wrapper

    # Warmup: caches the prompt latents and builds the CUDA graphs.
    m.synthesize(
        task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT,
        use_spk_emb=True, max_length=24, output_path=None,
    )
    m.model.flowloss.sample = timed("flow", m.model.flowloss.sample)
    m.model._project_audio_latent = timed("agg", m.model._project_audio_latent)
    m.model.audio.decode = timed("vae", m.model.audio.decode)

    for run in range(args.runs):
        for k in stats:
            stats[k] = 0 if not k.endswith("_n") else stats[k]
        torch.cuda.synchronize()
        t = time.perf_counter()
        wav = m.synthesize(
            task="TTS", text=TEXT, prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT,
            use_spk_emb=True, max_length=24, output_path=None,
        )
        total = time.perf_counter() - t
        dur = wav.shape[-1] / m.sample_rate
        flow, agg, vae = stats["flow"] / 1e3, stats["agg"] / 1e3, stats["vae"] / 1e3
        print(
            f"run{run}: total={total:.2f}s audio={dur:.2f}s rtf={total / dur:.3f} | "
            f"flow={flow:.2f}s ({stats['flow_n']} calls) agg={agg:.2f}s "
            f"({stats['agg_n']}) vae={vae:.2f}s ({stats['vae_n']}) "
            f"llm+rest={total - flow - agg - vae:.2f}s"
        )


if __name__ == "__main__":
    main()
