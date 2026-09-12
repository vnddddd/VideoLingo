#!/usr/bin/env python3
"""Run the full text/TTS/audio pipeline for a video already sitting in output/.

Why this exists
---------------
There are three ways into this pipeline, and they do not agree on where loudness
normalization happens:

  * Streamlit (st.py)            -- runs merge_full_audio, then normalize_dub_audio
  * Browser bridge (plugin)      -- runs the pipeline, then calls
                                    _normalize_plugin_audio itself
  * tools/split_pipeline.py      -- stops at merge_full_audio

split_pipeline.py deliberately stops early because it is the shared base for the
other two, which each own their own normalization step. Running it from a bare
shell therefore produces output/dub.mp3 with NO output/dub_loudnorm.mp3.

This wrapper is the CLI equivalent of what the other two do: run the pipeline,
then normalize. It calls the exact same shared normalize_audio_file used by both
(-13 LUFS, -1.5 dBFS true peak, 96k), so the result is identical to a Web UI or
plugin run.

It does NOT touch split_pipeline.py or any UI code path.

Usage:
    python tools/run_full_audio.py [--no-audio-prep] [--skip-normalize]

Expects exactly one video file in output/. Run prep first if output/audio/raw.mp3
does not exist yet:

    python tools/split_pipeline.py prep-audio --no-demucs
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DUB_FILE = Path("output/dub.mp3")
DUB_NORMALIZED_FILE = Path("output/dub_loudnorm.mp3")
PIPELINE = PROJECT_ROOT / "tools" / "split_pipeline.py"


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    started = time.time()
    proc = subprocess.run(
        [sys.executable, *cmd],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    elapsed = time.time() - started
    ok = proc.returncode == 0
    print(f"<<< {label}: {'OK' if ok else 'FAILED'} in {elapsed / 60:.1f} min", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio-prep", action="store_true",
                    help="skip prep-audio (assumes output/audio/raw.mp3 exists)")
    ap.add_argument("--skip-normalize", action="store_true",
                    help="stop after the pipeline, without loudness normalization")
    args = ap.parse_args()

    if not args.no_audio_prep and not Path("output/audio/raw.mp3").exists():
        if not _run([str(PIPELINE), "prep-audio", "--no-demucs"], "Extract source audio"):
            return 1

    # The pipeline itself: ASR -> split -> translate -> audio tasks -> TTS -> merge.
    # Stops before _12_dub_to_vid, so nothing is burned into the video.
    if not _run([str(PIPELINE), "local-until-audio"], "Text + TTS + audio pipeline"):
        return 1

    if args.skip_normalize:
        print("\nSkipping loudness normalization (--skip-normalize).")
        return 0

    # This is the step split_pipeline.py does not do for CLI callers. Same shared
    # function the Web UI and plugin use, so the result matches a UI run exactly.
    if not DUB_FILE.exists():
        print(f"\nERROR: {DUB_FILE} not found; cannot normalize.", flush=True)
        return 1

    print(f"\n{'=' * 70}\n>>> Loudness normalization\n{'=' * 70}", flush=True)
    from core._11_merge_audio import normalize_dub_audio

    normalize_dub_audio()

    print("\n" + "=" * 70)
    for f in (DUB_FILE, DUB_NORMALIZED_FILE, Path("output/dub.srt")):
        state = f"{f.stat().st_size / 1e6:.1f} MB" if f.exists() else "MISSING"
        print(f"  {str(f):<34} {state}")
    print("=" * 70)
    return 0 if DUB_NORMALIZED_FILE.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
