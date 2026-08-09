"""Wall-clock timings for each dubbing pipeline stage.

The two entry points behave differently: the browser bridge runs all nine
stages inside one `local-stop-before-video` process, while the Streamlit UI
runs each stage as its own `local-step` subprocess. Timings therefore
accumulate in a file rather than in memory, so both paths end up with one
complete record.

The record lives beside the other pipeline logs, so browser_bridge carries it
along when it copies work_output/ into a job directory -- which is what makes
the numbers available for later statistics.
"""
from __future__ import annotations

import json
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

TIMINGS_FILE = Path("output/log/stage_timings.json")
_SOURCE_AUDIO = Path("output/audio/raw.mp3")

# Stage names in both languages, keyed by the command-line alias rather than by
# the caller's label: the Streamlit UI passes an already-translated label while
# the bridge passes the English one, so the alias is the only stable key.
STAGE_NAMES: dict[str, tuple[str, str]] = {
    "asr": ("语音识别转字幕", "Transcription (ASR)"),
    "speaker-preview": ("说话人预览", "Speaker preview"),
    "split": ("语义切分", "Sentence segmentation"),
    "translate": ("摘要与翻译", "Summarize + translate"),
    "subtitles": ("字幕切分对齐", "Subtitle splitting"),
    "timeline": ("时间轴与字幕", "Timeline + subtitles"),
    "audio-tasks": ("配音任务与分块", "Audio tasks + chunks"),
    "reference-audio": ("参考音频提取", "Reference audio"),
    "tts-merge": ("TTS 生成与合并", "TTS + merge"),
}

# The figures worth watching. Splitting and translating are one logical step
# from a user's point of view, so they are reported together while still being
# stored separately.
HEADLINE_GROUPS: list[tuple[tuple[str, str], tuple[str, ...]]] = [
    (("语音识别", "Transcription (ASR)"), ("asr",)),
    (("切分 + 翻译", "Split + translation"), ("split", "translate")),
    (("TTS 生成 + 合并", "TTS + merge"), ("tts-merge",)),
]


def _width(text: str) -> int:
    """Terminal columns the text occupies; CJK glyphs take two, not one."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text: str, width: int) -> str:
    """Left-align to a column count, which str.ljust cannot do for CJK."""
    return text + " " * max(0, width - _width(text))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load() -> dict:
    if not TIMINGS_FILE.exists():
        return {"media": {}, "stages": {}}
    try:
        with TIMINGS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # A truncated file from an interrupted run must not break the pipeline.
        return {"media": {}, "stages": {}}
    data.setdefault("media", {})
    data.setdefault("stages", {})
    return data


def _save(data: dict) -> None:
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now_iso()
    with TIMINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _source_duration() -> float | None:
    """Length of the source media, the baseline for the realtime ratio."""
    if not _SOURCE_AUDIO.exists():
        return None
    try:
        from core.asr_backend.audio_preprocess import get_audio_duration

        return float(get_audio_duration(str(_SOURCE_AUDIO)))
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def record_stage(alias: str, label: str, seconds: float) -> None:
    """Add one stage run to the record.

    Re-running a stage accumulates its time and bumps `runs` rather than
    overwriting, so a resumed or retried pipeline still reports the real cost.
    """
    data = _load()
    entry = data["stages"].get(alias) or {}
    entry["label"] = label
    entry["seconds"] = round(float(entry.get("seconds", 0.0)) + seconds, 3)
    entry["runs"] = int(entry.get("runs", 0)) + 1
    entry["last_finished"] = _now_iso()
    data["stages"][alias] = entry

    # Resolve the media length once it exists; raw.mp3 is absent on the very
    # first stages of a fresh run.
    if not data["media"].get("duration_seconds"):
        duration = _source_duration()
        if duration:
            data["media"] = {
                "source": _SOURCE_AUDIO.as_posix(),
                "duration_seconds": round(duration, 3),
            }
    _save(data)


@contextmanager
def timed_stage(alias: str, label: str):
    """Time one stage, recording it even when the stage raises."""
    start = time.monotonic()
    try:
        yield
    finally:
        record_stage(alias, label, time.monotonic() - start)


def summary_lines() -> list[str]:
    """Render the bilingual timing report; empty when nothing was recorded."""
    data = _load()
    stages: dict = data.get("stages") or {}
    if not stages:
        return []

    total = sum(float(e.get("seconds", 0.0)) for e in stages.values())
    media_seconds = float((data.get("media") or {}).get("duration_seconds") or 0.0)

    lines = ["", "[bold cyan]━━━ 配音耗时统计 / Pipeline timings ━━━[/bold cyan]"]
    if media_seconds > 0:
        lines.append(
            f"  {_pad('原视频时长', 14)}{_pad('Source media', 22)}"
            f"{format_duration(media_seconds):>9}   ({media_seconds:.1f}s)"
        )
    lines.append(
        f"  {_pad('配音总耗时', 14)}{_pad('Dubbing total', 22)}"
        f"{format_duration(total):>9}   ({total:.1f}s)"
    )
    if media_seconds > 0:
        lines.append(
            f"  {_pad('实时倍率', 14)}{_pad('Realtime factor', 22)}"
            f"{total / media_seconds:>8.2f}x   (耗时 / 视频时长)"
        )

    def _pct(seconds: float) -> str:
        return f"{seconds / total * 100:5.1f}%" if total > 0 else "    -"

    lines.append("")
    lines.append("  [bold]分阶段明细 / Per stage[/bold]")
    for alias, entry in stages.items():
        seconds = float(entry.get("seconds", 0.0))
        runs = int(entry.get("runs", 1))
        zh, en = STAGE_NAMES.get(alias, ("", str(entry.get("label") or alias)))
        suffix = f"   重跑 x{runs}" if runs > 1 else ""
        lines.append(
            f"    {_pad(zh, 18)}{_pad(en, 30)}"
            f"{format_duration(seconds):>8}  {_pct(seconds)}{suffix}"
        )

    lines.append("")
    lines.append("  [bold]关键指标 / Headline[/bold]")
    for (zh, en), aliases in HEADLINE_GROUPS:
        seconds = sum(float((stages.get(a) or {}).get("seconds", 0.0)) for a in aliases)
        if seconds <= 0:
            continue
        extra = f"   {seconds / media_seconds:.2f}x" if media_seconds > 0 else ""
        lines.append(
            f"    {_pad(zh, 18)}{_pad(en, 30)}"
            f"{format_duration(seconds):>8}  {_pct(seconds)}{extra}"
        )

    lines.append(f"\n  明细已保存 / Saved to {TIMINGS_FILE.as_posix()}")
    return lines
