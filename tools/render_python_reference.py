r"""Batch-run the official Python reference (Ming-3.0-tts-0.5b) on the same 24
requests the Rust worker got, so the two can be compared A/B.

    set PYTHONPATH=D:\Ming-3.0-tts\.runtime\pylibs446
    %USERPROFILE%\miniconda3\envs\videolingo\python.exe tools/render_python_reference.py

Then rebuild the page with `python tools/build_listen_page.py`.

Mirrors out/voice_design/manifest.json field-for-field. Text normalisation is
off because pynini has no Windows wheel here; every text in the set is free of
digits and abbreviations, so TN would be a no-op anyway.
"""

import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(r"D:/Ming-3.0-tts")
sys.path.insert(0, str(ROOT))

from ming_tts import MingTTS  # noqa: E402

MANIFEST = pathlib.Path(r"D:/ming-3.0-tts-wgpu/out/voice_design/manifest.json")
OUT_DIR = pathlib.Path(r"D:/ming-3.0-tts-wgpu/out/voice_design/official_py")
SEED = 1895


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {ROOT / 'Ming-3.0-tts-0.5b'}", flush=True)
    model = MingTTS(str(ROOT / "Ming-3.0-tts-0.5b"), device="cuda:0",
                    text_normalization=False)
    print("loaded", flush=True)

    for item in items:
        controls = {}
        if item.get("style"):
            controls["style"] = item["style"]
        if item.get("dialect"):
            controls["dialect"] = item["dialect"]
        if item.get("ip"):
            controls["ip"] = item["ip"]

        seed_everything()
        out = OUT_DIR / (item["id"] + ".wav")
        start = time.time()
        waveform = model.synthesize(
            task=item["task"],
            text=item["text"],
            language_id=item.get("language_id"),
            instruction=item.get("instruction"),
            controls=controls or None,
            use_zero_spk_emb=item.get("use_zero_spk_emb", False),
            cfg=item.get("cfg", 2.0),
            sigma=item.get("sigma", 0.25),
            temperature=item.get("temperature", 0.0),
            max_length=item.get("max_length", 24),
            trim_silence=True,
            raw_text=False,
            output_path=str(out),
        )
        seconds = waveform.shape[-1] / model.sample_rate
        print(f"{item['id']:<18} {seconds:5.2f}s  {time.time() - start:5.1f}s wall",
              flush=True)


if __name__ == "__main__":
    main()
