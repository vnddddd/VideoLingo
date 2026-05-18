# ------------------------------------------
# Multi-speaker preview generator
# ------------------------------------------
# Called AFTER _2_asr.transcribe() has written cleaned_chunks.xlsx.
# When config.multi_speaker_enabled == true AND ASR returned >= 2 distinct
# speaker_id values, this module assembles a short audible preview per speaker
# under output/preview/ and drops a .pending flag so the UI knows to ask the
# user "which voice for which speaker?" before the rest of the pipeline runs.
#
# Idempotent + cheap: re-running just rewrites the wav/txt/manifest. No LLM,
# no GPU, no network. Pure file I/O.
#
# Side products:
#   output/preview/spk_<idx>.wav   - concatenated long-phrase clips (<=15s)
#   output/preview/spk_<idx>.txt   - corresponding sample transcript
#   output/preview/manifest.json   - [{idx, speaker_id, wav, txt, text,
#                                     duration, num_words}, ...]
#   output/preview/.pending        - presence == UI must show picker
# ------------------------------------------
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydub import AudioSegment
from rich import print as rprint

from core.utils.config_utils import load_key, update_key
from core.utils.models import (
    _2_CLEANED_CHUNKS,
    _RAW_AUDIO_FILE,
    _VOCAL_AUDIO_FILE,
)

# ------------------------------------------
# Constants
# ------------------------------------------
PREVIEW_DIR = Path("output/preview")
PENDING_FLAG = PREVIEW_DIR / ".pending"
MANIFEST_FILE = PREVIEW_DIR / "manifest.json"

PREVIEW_TARGET_SECONDS = 15.0   # accumulate up to this much speech per speaker (user spec: 15s)
MAX_CLIPS_PER_SPEAKER = 6       # cap clip count to keep the wav simple
PHRASE_MERGE_GAP_SECONDS = 0.5  # consecutive words within this gap stick together
MIN_WORDS_PER_SPEAKER = 5       # drop speakers that never speak meaningfully
MIN_SPEAKERS_TO_TRIGGER = 2     # 1 speaker => single-voice, no preview needed


# ------------------------------------------
# Public API
# ------------------------------------------
def generate_previews() -> List[Dict]:
    """Generate per-speaker preview wav/txt + manifest after ASR.

    Returns the manifest list (possibly empty). Sets PENDING_FLAG when the
    UI should prompt the user. Safe to call when multi-speaker is disabled
    or when ASR produced fewer than 2 speakers - both early-return cleanly.
    """
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if not bool(load_key("multi_speaker_enabled")):
        rprint("[yellow]🎤 Speaker preview skipped: multi_speaker_enabled = false[/yellow]")
        _reset_state()
        return []

    if not os.path.exists(_2_CLEANED_CHUNKS):
        rprint(f"[red]🎤 Speaker preview skipped: missing {_2_CLEANED_CHUNKS}[/red]")
        _reset_state()
        return []

    df = pd.read_excel(_2_CLEANED_CHUNKS)
    if "speaker_id" not in df.columns:
        rprint("[yellow]🎤 Speaker preview skipped: cleaned_chunks has no speaker_id column[/yellow]")
        _reset_state()
        return []

    valid = df[
        df["speaker_id"].notna()
        & (df["speaker_id"].astype(str).str.strip() != "")
        & (df["speaker_id"].astype(str).str.lower() != "nan")
    ].copy()
    if valid.empty:
        rprint("[yellow]🎤 Speaker preview skipped: no words tagged with speaker_id[/yellow]")
        _reset_state()
        return []

    valid["speaker_id"] = valid["speaker_id"].astype(str).str.strip()
    speaker_order = _stable_speaker_order(valid)
    if len(speaker_order) < MIN_SPEAKERS_TO_TRIGGER:
        rprint(
            f"[yellow]🎤 Speaker preview skipped: only {len(speaker_order)} distinct speaker(s) detected[/yellow]"
        )
        _reset_state()
        return []

    src_audio_path = _source_audio_path()
    if src_audio_path is None:
        raise FileNotFoundError(
            "Speaker preview: neither vocal.mp3 nor raw.mp3 exists under output/audio/"
        )
    rprint(f"[blue]🎤 Speaker preview: loading source audio {src_audio_path}[/blue]")
    audio = AudioSegment.from_file(src_audio_path)

    _purge_previous_outputs()

    manifest: List[Dict] = []
    for idx, speaker_id in enumerate(speaker_order):
        spk_rows = valid[valid["speaker_id"] == speaker_id].sort_values("start")
        if len(spk_rows) < MIN_WORDS_PER_SPEAKER:
            rprint(
                f"[yellow]  - speaker {speaker_id!r}: only {len(spk_rows)} words, skip[/yellow]"
            )
            continue
        clips = _collect_clips(spk_rows, PREVIEW_TARGET_SECONDS, MAX_CLIPS_PER_SPEAKER)
        if not clips:
            rprint(f"[yellow]  - speaker {speaker_id!r}: no usable phrase, skip[/yellow]")
            continue

        combined = AudioSegment.silent(duration=0)
        text_parts: List[str] = []
        for (start_s, end_s, text) in clips:
            seg = audio[int(start_s * 1000): int(end_s * 1000)]
            combined += seg + AudioSegment.silent(duration=200)
            text_parts.append(text)
        sample_text = " / ".join(part.strip() for part in text_parts if part.strip())

        wav_path = PREVIEW_DIR / f"spk_{idx}.wav"
        txt_path = PREVIEW_DIR / f"spk_{idx}.txt"
        combined.export(str(wav_path), format="wav")
        txt_path.write_text(sample_text, encoding="utf-8")

        manifest.append({
            "idx": idx,
            "speaker_id": speaker_id,
            "wav": str(wav_path).replace("\\", "/"),
            "txt": str(txt_path).replace("\\", "/"),
            "text": sample_text,
            "duration": round(combined.duration_seconds, 2),
            "num_words": int(len(spk_rows)),
        })
        rprint(
            f"[green]  ✓ speaker {speaker_id!r} -> spk_{idx}.wav "
            f"({combined.duration_seconds:.1f}s, {len(spk_rows)} words)[/green]"
        )

    if len(manifest) < MIN_SPEAKERS_TO_TRIGGER:
        rprint(
            f"[yellow]🎤 Speaker preview produced only {len(manifest)} usable speaker(s); "
            f"clearing pending flag[/yellow]"
        )
        _write_manifest(manifest)
        _clear_pending()
        return manifest

    _write_manifest(manifest)
    _set_pending({"speaker_ids": [m["speaker_id"] for m in manifest]})
    rprint(
        f"[bold blue]🎤 Speaker preview ready: {len(manifest)} speakers. "
        f"Awaiting user voice picks (UI flag: {PENDING_FLAG})[/bold blue]"
    )
    return manifest


def is_pending() -> bool:
    """True iff the previous run left a preview waiting for user voice picks."""
    return PENDING_FLAG.exists()


def is_required() -> bool:
    """True iff multi-speaker mode is enabled in config (regardless of state)."""
    return bool(load_key("multi_speaker_enabled"))


def read_manifest() -> List[Dict]:
    """Return manifest list, or [] if missing/corrupt."""
    if not MANIFEST_FILE.exists():
        return []
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        rprint(f"[red]🎤 manifest.json unreadable: {exc}[/red]")
        return []


def confirm_picks(speaker_voice_map: Dict[str, Dict]) -> None:
    """Persist user's choice into config.yaml's speaker_voice_map and clear flag.

    speaker_voice_map shape:
        {
            "<speaker_id>": {"mode": "fixed", "voice": "<tts_voice_name>"},
            "<speaker_id>": {"mode": "clone", "ref_wav": "<path>"},
            "<speaker_id>": {"mode": "default"},
            ...
        }
    """
    if not isinstance(speaker_voice_map, dict):
        raise TypeError(f"speaker_voice_map must be dict, got {type(speaker_voice_map).__name__}")
    update_key("speaker_voice_map", speaker_voice_map)
    _clear_pending()
    rprint(f"[green]🎤 Speaker voice map saved ({len(speaker_voice_map)} entries); pending flag cleared[/green]")


def reset() -> None:
    """Wipe preview directory so the next run starts clean. Used when the user
    re-uploads a fresh video."""
    _reset_state()
    _purge_previous_outputs()
    if MANIFEST_FILE.exists():
        MANIFEST_FILE.unlink()


# ------------------------------------------
# Internal helpers
# ------------------------------------------
def _source_audio_path() -> Optional[str]:
    if os.path.exists(_VOCAL_AUDIO_FILE):
        return _VOCAL_AUDIO_FILE
    if os.path.exists(_RAW_AUDIO_FILE):
        return _RAW_AUDIO_FILE
    return None


def _stable_speaker_order(valid: pd.DataFrame) -> List[str]:
    """Order speakers by first-appearance time (deterministic UI labelling)."""
    firsts = valid.groupby("speaker_id")["start"].min().sort_values()
    return list(firsts.index)


def _collect_clips(
    spk_rows: pd.DataFrame,
    total_seconds: float,
    max_clips: int,
) -> List[Tuple[float, float, str]]:
    """Greedy phrase selector.

    1. Merge adjacent words whose inter-gap <= PHRASE_MERGE_GAP_SECONDS into phrases.
    2. Sort phrases by duration desc, pick until accumulated >= total_seconds
       or len(picked) >= max_clips.
    3. Return picks in chronological order.
    """
    if spk_rows.empty:
        return []

    phrases: List[Tuple[float, float, str]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_words: List[str] = []

    for _, row in spk_rows.iterrows():
        try:
            s = float(row["start"])
            e = float(row["end"])
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        word = str(row.get("text", "")).strip().strip('"').strip()
        if not word:
            continue
        if cur_start is None:
            cur_start, cur_end, cur_words = s, e, [word]
        elif s - cur_end <= PHRASE_MERGE_GAP_SECONDS:
            cur_end = e
            cur_words.append(word)
        else:
            phrases.append((cur_start, cur_end, " ".join(cur_words)))
            cur_start, cur_end, cur_words = s, e, [word]
    if cur_start is not None and cur_end is not None:
        phrases.append((cur_start, cur_end, " ".join(cur_words)))

    if not phrases:
        return []

    phrases_sorted = sorted(phrases, key=lambda p: (p[1] - p[0]), reverse=True)
    picked: List[Tuple[float, float, str]] = []
    acc = 0.0
    for ph in phrases_sorted:
        picked.append(ph)
        acc += ph[1] - ph[0]
        if acc >= total_seconds or len(picked) >= max_clips:
            break
    picked.sort(key=lambda p: p[0])
    return picked


def _purge_previous_outputs() -> None:
    if not PREVIEW_DIR.exists():
        return
    for pattern in ("spk_*.wav", "spk_*.txt"):
        for old in PREVIEW_DIR.glob(pattern):
            try:
                old.unlink()
            except OSError as exc:
                rprint(f"[yellow]preview: cannot delete {old}: {exc}[/yellow]")


def _write_manifest(manifest: List[Dict]) -> None:
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_pending(payload: Dict) -> None:
    PENDING_FLAG.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _clear_pending() -> None:
    if PENDING_FLAG.exists():
        try:
            PENDING_FLAG.unlink()
        except OSError as exc:
            rprint(f"[yellow]preview: cannot delete {PENDING_FLAG}: {exc}[/yellow]")


def _reset_state() -> None:
    _clear_pending()


# ------------------------------------------
# CLI entry: `python -m core._3_speaker_preview` for manual probing
# ------------------------------------------
if __name__ == "__main__":
    out = generate_previews()
    rprint(f"[bold]Generated manifest:[/bold] {json.dumps(out, ensure_ascii=False, indent=2)}")
