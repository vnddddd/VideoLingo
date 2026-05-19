"""Build ONE stable global reference clip for MiMo voiceclone (single-speaker mode).

Why this exists
---------------
The original single-speaker clone path in ``mimo_tts.mimo_tts_for_videolingo``
feeds a per-sentence reference (``output/audio/refers/{number}.wav``) into the
voiceclone model for every line. Those per-sentence clips are often <5s and
their timbre/pitch varies wildly across sentences, which makes the cloned
voice drift between lines.

This module produces a single, longer (target ~22s), better-balanced
reference by concatenating the longest phrases from the whole transcript.
The same wav is then reused for every sentence, so cloned timbre stays
consistent line-to-line.

Strategy (reuses :func:`core._3_speaker_preview._collect_clips`):

1.  Read word-level ``cleaned_chunks.xlsx`` (no speaker filtering — this is
    the single-speaker path; if true multi-speaker is needed, set
    ``multi_speaker_enabled: true`` and use the diarization flow instead).
2.  Merge adjacent words within ``PHRASE_MERGE_GAP_SECONDS`` into phrases.
3.  Sort phrases by duration desc, greedy-pick until accumulated duration
    >= target_seconds OR count >= max_clips.
4.  Restore chronological order, concat with 200ms silence between clips
    (matches the multi-speaker preview format).
5.  Export to ``output/audio/refers/_long_ref.wav`` (leading underscore so
    it never collides with per-sentence ``{number}.wav`` files).

MiMo voiceclone accepts ≤10 MB base64-encoded mp3/wav — at 16 kHz mono 16-bit
that's roughly 4 minutes, so a 22s clip is well within limits.

Idempotent: if the long-ref file already exists and looks healthy, the call
is a no-op. Pass ``force=True`` to rebuild.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from pydub import AudioSegment

from core.utils import load_key
from core.utils.models import _2_CLEANED_CHUNKS

_REFERS_DIR = Path("output/audio/refers")
LONG_REF_PATH = _REFERS_DIR / "_long_ref.wav"

# Hard floor: if we somehow get a clip much shorter than the requested
# target, treat the cache as stale and rebuild. 60% of target is generous
# enough to tolerate short videos while still catching corruption.
_HEALTH_DURATION_RATIO = 0.6
_HEALTH_MIN_SECONDS = 8.0


def _safe_load_key(key: str, default):
    try:
        v = load_key(key)
        return default if v is None else v
    except Exception:
        return default


def _is_healthy(path: Path, target_seconds: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        seg = AudioSegment.from_file(path)
    except Exception:
        return False
    floor = max(target_seconds * _HEALTH_DURATION_RATIO, _HEALTH_MIN_SECONDS)
    return seg.duration_seconds >= floor


def ensure_long_ref(force: bool = False) -> Path:
    """Build (or reuse) the global long reference wav.

    Returns the path on success. **Raises** on any failure (missing source
    audio, missing cleaned_chunks, malformed columns, empty transcript,
    phrase selector yielding nothing, …) — callers should NOT silently fall
    back; if the merged reference cannot be built the user wants to see the
    error and decide.
    """
    target_seconds = float(_safe_load_key(
        "mimo_tts.single_speaker_merged_ref_target_seconds", 22.0))
    max_clips = int(_safe_load_key(
        "mimo_tts.single_speaker_merged_ref_max_clips", 8))

    if not force and _is_healthy(LONG_REF_PATH, target_seconds):
        return LONG_REF_PATH

    # Lazy import: _3_speaker_preview pulls in a lot (rich, pyannote helpers).
    from core._3_speaker_preview import _collect_clips, _source_audio_path

    src = _source_audio_path()
    if not src or not os.path.exists(src):
        # Trigger demucs/raw extraction the same way _9_refer_audio does.
        # Any exception from extract_refer_audio_main propagates intentionally.
        from core._9_refer_audio import extract_refer_audio_main
        extract_refer_audio_main()
        src = _source_audio_path()
        if not src or not os.path.exists(src):
            raise FileNotFoundError(
                "long-ref: source audio not available even after "
                "extract_refer_audio_main(); expected vocal.mp3 or raw.mp3 "
                "under output/audio/"
            )

    if not os.path.exists(_2_CLEANED_CHUNKS):
        raise FileNotFoundError(
            f"long-ref: required transcript missing: {_2_CLEANED_CHUNKS}"
        )

    df = pd.read_excel(_2_CLEANED_CHUNKS)
    # _collect_clips only reads start/end/text; speaker_id intentionally ignored
    # (this is the single-speaker stable-timbre path).
    needed = {"start", "end", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"long-ref: cleaned_chunks missing required columns {missing}"
        )
    df = df.dropna(subset=["start", "end", "text"])
    if df.empty:
        raise ValueError("long-ref: cleaned_chunks empty after dropna")

    clips = _collect_clips(df, total_seconds=target_seconds, max_clips=max_clips)
    if not clips:
        raise RuntimeError(
            "long-ref: phrase selector returned no clips "
            f"(target={target_seconds}s, max_clips={max_clips})"
        )

    audio = AudioSegment.from_file(src)
    combined = AudioSegment.silent(duration=0)
    picked_text_parts = []
    for (start_s, end_s, text) in clips:
        seg = audio[int(start_s * 1000): int(end_s * 1000)]
        combined += seg + AudioSegment.silent(duration=200)
        picked_text_parts.append(text.strip())

    _REFERS_DIR.mkdir(parents=True, exist_ok=True)
    combined.export(str(LONG_REF_PATH), format="wav")
    # Sidecar text file for debugging — what phrases got picked.
    try:
        (_REFERS_DIR / "_long_ref.txt").write_text(
            " / ".join(p for p in picked_text_parts if p),
            encoding="utf-8",
        )
    except Exception:
        pass

    print(
        f"🎙️ long-ref built: {LONG_REF_PATH} "
        f"({combined.duration_seconds:.1f}s, {len(clips)} phrases)"
    )
    return LONG_REF_PATH
