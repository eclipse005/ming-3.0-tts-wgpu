"""The 36 params_test cases against the STORED reference goldens.

`check_params_align.py` re-renders the reference with torch; this one replays the
recorded `torch.randn` draws (MING_NOISE_GOLDEN) and compares to the
`ref_waveform.npy` that was dumped alongside them, so the audio gate works on a
box with no working Python/CUDA environment.

    python tools/_goldens_align.py            # all 36, cuda:0
    python tools/_goldens_align.py A1_bare A3_clone
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REF_REPO = os.environ.get("MING_REF", "D:/Ming-3.0-tts")
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, REF_REPO)
import numpy as np
import soundfile as sf
import subprocess

from check_params_align import SPEC, cli_args, compare

GOLD = str(HERE / "goldens" / "m5_params")
OUT = REF_REPO + "/output/rust_params"


def main():
    only = set(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ming_ref_")
    rows = []
    for name, task, text, kwargs in SPEC:
        if only and name not in only:
            continue
        gdir = os.path.join(GOLD, name)
        ref = np.load(os.path.join(gdir, "ref_waveform.npy")).astype(np.float32)
        ref_wav = os.path.join(tmp, f"{name}.wav")
        sf.write(ref_wav, ref, 44100)

        env = dict(os.environ)
        env["MING_NOISE_GOLDEN"] = gdir
        p = subprocess.run(cli_args(name, task, text, kwargs),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, cwd=REF_REPO)
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
            print(f"{name:22s} FAILED {tail}", flush=True)
            rows.append((name, None))
            continue
        got = os.path.join(OUT, f"{name}.wav")
        r = compare(ref_wav, got)
        rows.append((name, r))
        print(f"{name:22s} ref {len(ref)/44100:6.2f}s  dur {r['dur_ratio']:.4f}  "
              f"rel {r['rel_l2']:.4f}  mel {r['logmel_corr']:.5f}  "
              f"spk {r['spk_cos']:.4f}", flush=True)

    ok = [r for _, r in rows if r]
    if not ok:
        return 1
    durs = [r["dur_ratio"] for r in ok]
    print(f"\n{len(ok)}/{len(rows)} rendered | dur ratio min {min(durs):.4f} "
          f"max {max(durs):.4f} | exact {sum(d == 1.0 for d in durs)}/{len(durs)} | "
          f"mel>=0.995 {sum(r['logmel_corr'] >= 0.995 for r in ok)}/{len(ok)} | "
          f"rel<=5e-2 {sum(r['rel_l2'] <= 5e-2 for r in ok)}/{len(ok)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
