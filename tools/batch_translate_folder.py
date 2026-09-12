#!/usr/bin/env python3
"""Batch-translate + dub every loose video in a folder with Fish Audio.

Companion to run_full_audio.py: that one handles a single video already staged
in output/, this one drives the whole folder.

For each video (shortest first by default):
  1. copy it into output/ (the pipeline finds its source video there)
  2. run tools/run_full_audio.py  (prep -> pipeline -> loudness normalization)
  3. archive dub*.mp3/dub.srt/refers/log/gpt_log into a project subfolder
  4. clear output/ for the next video

It never touches the source videos: they are copied, not moved.

Stops automatically when:
  * --until HH:MM passes (so an unattended run cannot outlive the night), or
  * the translation provider runs out of quota (insufficient_user_quota).

Usage:
    python tools/batch_translate_folder.py --root "D:\\path\\to\\folder" [--until 01:00]
                                          [--dry-run] [--min-seconds N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

OUTPUT = PROJECT_ROOT / "output"
RUNNER = PROJECT_ROOT / "tools" / "run_full_audio.py"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

# Phrases the providers use for an exhausted balance. Matched loosely because
# each reseller words it differently (and some wrap it in Chinese).
QUOTA_MARKERS = (
    "insufficient_user_quota",
    "insufficient_quota",
    "quota exceeded",
    "余额不足",
    "额度不足",
    "用户额度",
)


def log(msg: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        import json
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def project_dir_name(video: Path) -> str:
    """Turn '1 Q and A.mp4' into a filesystem-friendly project folder name."""
    stem = video.stem.strip()
    stem = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    return stem or "video"


def has_quota_error(text: str) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in QUOTA_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--until", default=None,
                    help="stop starting new videos after this local time, HH:MM")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-seconds", type=float, default=0.0)
    ap.add_argument("--longest-first", action="store_true",
                    help="default is shortest first")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        log(f"ERROR: not a folder: {root}")
        return 1

    deadline = None
    if args.until:
        hh, mm = (int(x) for x in args.until.split(":"))
        now = dt.datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            deadline += dt.timedelta(days=1)

    videos = []
    for f in sorted(root.iterdir()):
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTS:
            continue
        # Skip a video whose project folder already exists (already processed).
        if (root / project_dir_name(f)).is_dir():
            log(f"SKIP (already has project folder): {f.name}")
            continue
        d = duration(f)
        if d < args.min_seconds:
            continue
        videos.append((d, f))
    videos.sort(key=lambda x: x[0], reverse=args.longest_first)

    total = sum(v[0] for v in videos)
    log(f"{len(videos)} video(s) to process | {total / 3600:.2f} h of footage")
    if deadline:
        log(f"Deadline: {deadline.strftime('%H:%M')} (no new video started after it)")
    if not videos:
        return 0

    done, failed, skipped_for_time = [], [], []
    quota_hit = False

    for idx, (dur, video) in enumerate(videos, 1):
        if deadline and dt.datetime.now() >= deadline:
            log(f"Deadline reached; not starting: {video.name}")
            # videos holds (duration, path) tuples, not paths.
            skipped_for_time.extend(video_path.name for _d, video_path in videos[idx - 1:])
            break

        log("=" * 72)
        log(f"[{idx}/{len(videos)}] {video.name}  ({dur / 60:.1f} min)")
        log("=" * 72)

        if args.dry_run:
            log("  [dry] would copy -> output/, run pipeline, archive")
            continue

        # 1. stage the source video (the pipeline expects it in output/)
        OUTPUT.mkdir(exist_ok=True)
        for stale in OUTPUT.iterdir():
            if stale.name in ("audio", "log", "gpt_log"):
                shutil.rmtree(stale, ignore_errors=True)
            else:
                try:
                    stale.unlink()
                except Exception:
                    pass
        shutil.copy2(video, OUTPUT / video.name)

        # 2. run the single-video pipeline (prep -> pipeline -> normalize)
        started = time.time()
        proc = subprocess.run(
            [sys.executable, str(RUNNER)],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        elapsed = time.time() - started

        if has_quota_error(combined):
            log("TRANSLATION QUOTA EXHAUSTED -- stopping here.")
            log("  " + combined.strip().splitlines()[-1][:200])
            quota_hit = True
            break

        ok = proc.returncode == 0 and (OUTPUT / "dub_loudnorm.mp3").exists()
        if not ok:
            log(f"  FAILED in {elapsed / 60:.1f} min; keeping output/ for inspection")
            tail = "\n".join(combined.strip().splitlines()[-12:])
            log("  " + tail.replace("\n", "\n  "))
            failed.append(video.name)
            continue

        # 3. archive the results beside the source video
        dest = root / project_dir_name(video)
        dest.mkdir(exist_ok=True)
        for name in ("dub.mp3", "dub_loudnorm.mp3", "dub.srt"):
            src = OUTPUT / name
            if src.exists():
                shutil.copy2(src, dest / name)
        for sub in ("refers", "log", "gpt_log"):
            s = OUTPUT / "audio" / sub if sub == "refers" else OUTPUT / sub
            if s.is_dir():
                shutil.copytree(s, dest / sub, dirs_exist_ok=True)

        log(f"  OK in {elapsed / 60:.1f} min -> archived to {dest.name}/")
        done.append((video.name, elapsed, dur))

    log("")
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    for name, el, dur in done:
        log(f"  OK    {el / 60:6.1f} min  {dur / 60:7.1f} min video  {name}")
    for name in failed:
        log(f"  FAIL  {'':>6}       {name}")
    for name in skipped_for_time:
        log(f"  TODO  {'':>6}       {name}")
    log(f"\ncompleted {len(done)}, failed {len(failed)}, not started {len(skipped_for_time)}")
    if quota_hit:
        log("STOPPED: translation quota exhausted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
