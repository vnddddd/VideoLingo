#!/usr/bin/env python3
"""D - run, watch, repeat.

Processes a flat folder of videos one after another in NATURAL FILENAME order
(1.1, 1.2, ... 10.6) so a finished episode can be watched while the next one
renders.

What it deliberately does NOT do: reimplement the pipeline. Every video is
executed by tools/run_full_audio.py, i.e. the exact same chain the Web UI runs:

    prep-audio -> ASR (Soniox, diarized) -> NLP sentence split -> summarise +
    terminology -> 3-step translate/reflect/adapt -> subtitle cut & align ->
    audio tasks & chunks -> per-speaker reference clips -> TTS (Fish Audio) ->
    merge -> loudness normalisation

It stops before core/_12_dub_to_vid.py, so nothing is burned into the video and
the source file is never touched (it is copied into output/ for the run).

What it adds on top:
  * only the two deliverables survive, exported beside the source video:
        <name>.zh-CN.mp3    dubbed audio (loudness-normalised)
        <name>.zh-CN.srt    translated subtitles on the dubbing timeline
  * every intermediate artefact is deleted after each video
  * a video whose deliverables already exist is skipped, so an interrupted run
    resumes exactly where it stopped
  * a failure is quarantined under .failed/<name>/ and the run continues

Usage:
    python tools/batch_sequential_dub.py --root "D:\path\to\folder" [--dry-run]
    python tools/batch_sequential_dub.py --root ... --limit 1
    python tools/batch_sequential_dub.py --root ... --only 3.2
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

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

OUTPUT = PROJECT_ROOT / "output"
RUNNER = PROJECT_ROOT / "tools" / "run_full_audio.py"
FAILED_DIR = PROJECT_ROOT / ".failed"
LOGS = PROJECT_ROOT / "logs"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")

# Provider wording for an exhausted balance, matched case-insensitively.
QUOTA_MARKERS = (
    "insufficient_user_quota", "insufficient_quota", "quota exceeded",
    "余额不足", "额度不足", "用户额度", "402",
)


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# Separators that carry no meaning for ordering. Filenames in the same course
# use them interchangeably ("AlgoHub - 2.1" vs "AlgoHub _ 2.2"), and letting
# them take part in the comparison makes "-" sort before "_", which pushes 2.1
# behind 2.2. Normalising them to a single space removes that false difference.
_SEPARATOR_RE = re.compile(r"[\s_\-.,@&+]+")


def natural_key(path: Path):
    """Sort 1.2 before 1.10, and both before 2.1.

    Uses the stem for files so the extension cannot skew the comparison, and
    relies on list comparison to place an unnumbered name before its numbered
    siblings: "The Oracle" < "The Oracle (2)" < "The Oracle (7)".
    """
    text = path.stem if path.suffix else path.name
    text = _SEPARATOR_RE.sub(" ", text.lower())
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p for p in parts]


# A lesson/chapter number that ends the name, optionally in brackets or after a
# dash: "AlgoHub - 1.7", "Algo Trading Course _ AlgoHub - 2.4". This is what
# actually orders a course; the words in front of it are just the course name.
_TRAILING_LESSON_RE = re.compile(r"(\d+)\s*[.\-]\s*(\d+)\s*$")


def lesson_key(path: Path):
    """Order by the trailing "chapter.lesson" number, falling back to the name.

    Courses in one folder are often named inconsistently ("AlgoHub - 1.7" next
    to "Algo Trading Course - AlgoHub - 2.4"), and plain natural sorting then
    puts the whole numeric block in front of the other. Extracting the trailing
    pair first restores chapter 1 before chapter 2 regardless of the prefix.
    Files with no such number (documentaries, single lessons) sort after the
    numbered ones, keeping their own natural order.
    """
    text = path.stem if path.suffix else path.name
    match = _TRAILING_LESSON_RE.search(text.strip())
    if match:
        return (0, int(match.group(1)), int(match.group(2)), natural_key(path))
    return (1, 0, 0, natural_key(path))


def _name_key(name: str):
    """Natural key for a bare folder/file name (never treats dots as suffixes)."""
    text = _SEPARATOR_RE.sub(" ", name.lower())
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text))


def tree_key(root: Path, video: Path):
    """Order by the FULL sub-folder chain, then by filename inside it.

    Every path component is compared in turn, so "1.INTRO/2.Strong..." sorts
    inside chapter 1 instead of being compared against a top-level "2.THEORY".
    Comparing only the last directory name — the previous behaviour — mixed the
    two chapters together. Without --recursive every file sits in one folder
    and this collapses to sorting on the filename.
    """
    try:
        relative_parent = video.parent.relative_to(root)
    except ValueError:
        relative_parent = Path(".")
    folders = tuple(_name_key(part) for part in relative_parent.parts if part not in (".", ""))
    return (folders, natural_key(video))


def safe_stem(video: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z]+", "_", video.stem).strip("_")
    return stem or "video"


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


def deliverable_paths(video: Path) -> tuple[Path, Path]:
    """The two files this runner promises, resolved beside the source video."""
    from core.utils.output_names import output_filename

    audio = video.parent / output_filename(Path("output/dub_loudnorm.mp3"), video)
    dub_sub = video.parent / output_filename(Path("output/dub.srt"), video)
    return audio, dub_sub


def clear_dir(path: Path) -> None:
    """Delete the contents of a directory, keeping the directory itself."""
    if not path.is_dir():
        return
    for child in list(path.iterdir()):
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except Exception as exc:  # noqa: BLE001
            log(f"    could not remove {child.name}: {exc}")


def export_deliverables(video: Path) -> list[Path]:
    """Copy the finished audio + subtitles out, using the project's own exporter.

    dub_loudnorm.mp3 -> <name>.zh-CN.mp3    (dubbed, loudness-normalised)
    dub.srt          -> <name>.zh-CN.srt    (translated script, on the dub timeline)

    trans.srt (the same translation on the original timeline) is deliberately
    NOT exported: it duplicates the content of dub.srt and only differs in
    timing, which is not wanted here.
    """
    from core.utils.output_names import export_output_file, output_filename

    written: list[Path] = []
    for name in ("dub_loudnorm.mp3", "dub.srt"):
        source = OUTPUT / name
        if not source.is_file():
            continue
        destination = video.parent / output_filename(Path("output") / name, video)
        if source.resolve() == destination.resolve():
            written.append(destination)
            continue
        exported = export_output_file(source, video_file=video, destination_dir=video.parent)
        if exported is not None:
            written.append(exported)
    return written


def has_quota_error(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in QUOTA_MARKERS)


def run_one(video: Path, log_file: Path) -> tuple[bool, float, str]:
    """Run the full project pipeline for one video. Returns (ok, seconds, tail)."""
    started = time.time()

    OUTPUT.mkdir(exist_ok=True)
    clear_dir(OUTPUT)
    shutil.copy2(video, OUTPUT / video.name)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    tail: list[str] = []
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(
            "\n" + "=" * 78 + "\n"
            + f"{dt.datetime.now().isoformat(timespec='seconds')}  {video.name}\n"
            + "=" * 78 + "\n"
        )
        handle.flush()
        proc = subprocess.Popen(
            [sys.executable, "-u", str(RUNNER)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            stripped = line.rstrip()
            if stripped.startswith(">>>") or stripped.startswith("<<<"):
                log(f"    {stripped[3:].strip()}")
            tail.append(stripped)
            if len(tail) > 40:
                tail.pop(0)
        code = proc.wait()

    elapsed = time.time() - started
    ok = code == 0 and (OUTPUT / "dub_loudnorm.mp3").is_file() and (OUTPUT / "dub.srt").is_file()
    return ok, elapsed, "\n".join(tail[-15:])


def quarantine(video: Path) -> None:
    """Move a failed run's output aside so the next video starts clean."""
    destination = FAILED_DIR / safe_stem(video)
    try:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        FAILED_DIR.mkdir(exist_ok=True)
        shutil.move(str(OUTPUT), str(destination))
        log(f"    failed artefacts kept in {destination.relative_to(PROJECT_ROOT)}/")
    except Exception as exc:  # noqa: BLE001
        log(f"    could not quarantine output/: {exc}")
    OUTPUT.mkdir(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the videos")
    ap.add_argument("--only", default=None, help="substring filter on the filename")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (0 = all)")
    ap.add_argument("--start", type=int, default=1, help="1-based position in the sorted list")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if deliverables exist")
    ap.add_argument("--recursive", action="store_true",
                    help="also process videos in sub-folders, ordered folder by folder")
    ap.add_argument("--folder-order", default=None,
                    help="comma-separated sub-folder names to run first, in that order; "
                         "any folder not listed follows in natural order")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        log(f"ERROR: not a folder: {root}")
        return 1

    if args.recursive:
        videos = [
            f for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
        rank = []
        if args.folder_order:
            rank = [name.strip().lower() for name in args.folder_order.split(",") if name.strip()]

        def folder_rank(video: Path) -> int:
            """Folders named on the command line come first, in the given order."""
            try:
                top = video.relative_to(root).parts[0].lower()
            except ValueError:
                return len(rank)
            return rank.index(top) if top in rank else len(rank)

        videos.sort(key=lambda v: (folder_rank(v), tree_key(root, v)))
    else:
        videos = [
            f for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
        videos.sort(key=lesson_key)
    if args.only:
        videos = [v for v in videos if args.only.lower() in v.name.lower()]
    videos = videos[max(0, args.start - 1):]
    if args.limit:
        videos = videos[: args.limit]

    LOGS.mkdir(exist_ok=True)
    log_file = LOGS / f"batch_seq_{dt.datetime.now():%Y%m%d_%H%M%S}.log"

    todo = []
    for video in videos:
        audio, dub_sub = deliverable_paths(video)
        if not args.force and audio.is_file() and dub_sub.is_file():
            log(f"SKIP (already dubbed): {video.name}")
            continue
        todo.append((video, duration(video)))

    total = sum(d for _v, d in todo)
    log(f"{len(todo)} video(s) queued | {total / 3600:.2f} h of footage")
    log(f"Run log: {log_file}")
    if not todo:
        return 0

    done: list[tuple[str, float, float]] = []
    failed: list[str] = []
    stopped = ""

    for index, (video, seconds) in enumerate(todo, 1):
        log("=" * 72)
        shown = video.relative_to(root) if video.parent != root else video.name
        log(f"[{index}/{len(todo)}] {shown}  ({seconds / 60:.1f} min)")
        log("=" * 72)
        if args.dry_run:
            log("  [dry] would run the full pipeline and export audio + subtitles")
            continue

        ok, elapsed, tail_text = run_one(video, log_file)
        if not ok:
            log(f"  FAILED after {elapsed / 60:.1f} min")
            if has_quota_error(tail_text):
                log("  translation/TTS quota exhausted -- stopping the whole batch.")
                quarantine(video)
                failed.append(f"{video.name} (quota)")
                stopped = "quota"
                break
            for line in tail_text.splitlines()[-12:]:
                log(f"    {line}")
            quarantine(video)
            failed.append(video.name)
            continue

        written = export_deliverables(video)
        log("  exported: " + ", ".join(p.name for p in written))
        clear_dir(OUTPUT)
        done.append((video.name, elapsed, seconds))
        log(f"  OK in {elapsed / 60:.1f} min")

    log("")
    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    for name, elapsed, seconds in done:
        log(f"  OK    {elapsed / 60:6.1f} min render  {seconds / 60:7.1f} min video  {name}")
    for name in failed:
        log(f"  FAIL  {'':>6}         {name}")
    log(f"\ncompleted {len(done)}, failed {len(failed)}, remaining "
        f"{max(0, len(todo) - len(done) - len(failed))}")
    if stopped:
        log(f"STOPPED: {stopped}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
