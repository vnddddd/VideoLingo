"""Loudness normalization shared by Web UI, final video, and browser bridge.

ffmpeg's one-pass ``loudnorm`` normalizes in *dynamic* mode: it recomputes the
gain every 100 ms from a sliding window, so a pause in the dub lets the gain
climb far above its steady-state value (measured on a real job: ~+5 dB while
someone speaks, ~+30 dB after a 3 s gap). When the next sentence starts, the
first syllable is still riding that stale gain and gets slammed into the
limiter -- a pop right before the speech, which is exactly what silent gaps
between dubbed lines produce.

So we normalize in two passes instead: measure the programme loudness, then
apply one constant gain followed by a look-ahead limiter. The gain never
moves, so a silent gap cannot pump it up, and the limiter only ever pulls
peaks down -- never boosts an onset.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

# Perceived loudness we aim for, and the peak ceiling we refuse to cross.
TARGET_LOUDNESS_LUFS = -13.0
TARGET_TRUE_PEAK_DBFS = -1.5
# alimiter caps *sample* peaks, but the target is a *true* (inter-sample) peak.
# The gap between the two grows as the sample rate falls -- measured on real
# 16 kHz dubs it ranged from 1.0 dB to 2.7 dB, so no fixed headroom is safe.
# Running the limiter on an oversampled signal lets it see the inter-sample
# peaks directly, which controls the true peak whatever the rate.
_LIMITER_OVERSAMPLE = 4
# Never boost a quiet track so hard that its noise floor becomes audible.
_MAX_GAIN_DB = 20.0
# ebur128 reports about -70 LUFS for silence; anything at or below that has no
# programme material to normalize.
_SILENCE_FLOOR_LUFS = -70.0

# The ebur128 end-of-run summary prints the integrated loudness on its own
# line; the per-frame progress lines carry other fields before the "I:" so they
# do not match.
_INTEGRATED_RE = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS\s*$", re.MULTILINE)


def probe_sample_rate(path: str, default: int = 48000) -> int:
    """Return the sample rate of the first audio stream in ``path``.

    Used to keep a normalized copy at its source rate instead of resampling it
    up, and to size the limiter's oversampling. Falls back to ``default`` when
    ffprobe cannot tell.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    rate = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not rate or not rate[0].isdigit():
        return default
    return int(rate[0])


def measure_integrated_loudness(
    inputs: list[str],
    filter_complex: str | None = None,
    audio_label: str | None = None,
) -> float:
    """Return the integrated loudness (LUFS) of an ffmpeg audio input.

    ``inputs`` are the ffmpeg input arguments, e.g. ``["-i", "output/dub.mp3"]``.
    To measure a mix rather than a single file, pass the ``filter_complex`` that
    builds it plus the ``audio_label`` naming its output pad. Only audio is
    decoded, so keep video inputs out of ``inputs``.
    """
    command = ["ffmpeg", "-hide_banner", "-nostats", *inputs]
    if filter_complex is not None:
        if not audio_label:
            raise ValueError("audio_label is required when filter_complex is given")
        command += [
            "-filter_complex",
            f"{filter_complex};{audio_label}ebur128=peak=true[vl_loudness]",
            "-map", "[vl_loudness]",
        ]
    else:
        command += ["-filter:a", "ebur128=peak=true"]
    command += ["-f", "null", "-"]

    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    output = result.stderr or result.stdout or ""
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to measure loudness (return code {result.returncode}): "
            f"{output.strip()[-2000:]}"
        )
    matches = _INTEGRATED_RE.findall(output)
    if not matches:
        raise RuntimeError(
            "ffmpeg produced no ebur128 loudness summary; "
            f"output was: {output.strip()[-2000:]}"
        )
    return float(matches[-1])


def build_normalize_filter(measured_lufs: float, sample_rate: int) -> str:
    """Build the ffmpeg audio filter that moves ``measured_lufs`` onto target.

    The result is a constant gain plus a look-ahead limiter run at
    ``_LIMITER_OVERSAMPLE`` times ``sample_rate``, then resampled back down, so
    it can be dropped into either ``-filter:a`` or a ``filter_complex`` chain.
    ``sample_rate`` is the rate the normalized audio should come out at.
    """
    if measured_lufs <= _SILENCE_FLOOR_LUFS:
        gain_db = 0.0
    else:
        gain_db = min(TARGET_LOUDNESS_LUFS - measured_lufs, _MAX_GAIN_DB)
    return (
        f"volume={gain_db:.2f}dB,"
        f"aresample={sample_rate * _LIMITER_OVERSAMPLE},"
        f"alimiter=limit={TARGET_TRUE_PEAK_DBFS:.2f}dB:level=disabled,"
        f"aresample={sample_rate}"
    )


def normalize_audio_file(
    src: str | Path,
    dst: str | Path,
    *,
    bitrate: str = "96k",
) -> tuple[float, int, str]:
    """Normalize one audio file into ``dst`` with the shared two-pass method.

    The destination is written through a temporary file and atomically replaced
    only after ffmpeg succeeds, so a failed normalization cannot corrupt an
    existing final output. Returns ``(measured_lufs, sample_rate, filter)`` for
    logging and tests.
    """
    source = Path(src).resolve()
    destination = Path(dst).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    if source == destination:
        raise ValueError("Source and destination must be different audio files")

    destination.parent.mkdir(parents=True, exist_ok=True)
    measured_lufs = measure_integrated_loudness(["-i", str(source)])
    sample_rate = probe_sample_rate(source)
    audio_filter = build_normalize_filter(measured_lufs, sample_rate)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-filter:a",
            audio_filter,
            "-b:a",
            bitrate,
            str(temp_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"ffmpeg failed to normalize audio (return code {result.returncode}): "
                f"{details[-2000:]}"
            )
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)

    return measured_lufs, sample_rate, audio_filter
