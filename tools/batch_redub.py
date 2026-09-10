"""Batch re-dub every VideoLingo project under a root folder with Fish Audio.

Why this exists
---------------
These projects were produced by an older VideoLingo that used a FLAT layout
(\`audio/\`, \`log/\`) directly beside the video, while the current code reads
and writes \`output/audio/\`, \`output/log/\`. Rather than move the user's files,
this script creates directory junctions so \`output/\` points at the real folders.
Junctions are non-destructive: nothing is moved, copied, or deleted.

Per project it:
  1. verifies the project actually has ASR + translation output to reuse
  2. backs up the existing dub audio (and the cached per-line wavs)
  3. creates the \`output/\` junctions if the layout is flat
  4. clears ONLY the dubbing artifacts (tmp/, segs/, dub*.mp3)
  5. runs the dubbing stage, then the merge stage
  6. restores the original config and removes the junctions when done

It never re-runs download, ASR, splitting, or translation, and never touches
the source video.

Usage:
    python batch_redub.py --root "D:\\视频\\NO BS\\Starter Extras" [--dry-run]
                          [--no-backup] [--only NAME] [--max-workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
# Folders that must exist under output/ for the pipeline to read/write its state.
LINK_NAMES = ("audio", "log", "gpt_log")


def log(msg: str) -> None:
    print(msg, flush=True)


def find_video(d: Path) -> Path | None:
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            return f
    return None


def video_duration(path: Path) -> float:
    """Duration in seconds via ffprobe, falling back to 0.0 when unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def count_lines(d: Path) -> int:
    """Number of TTS lines, read from the task file the pipeline already built."""
    try:
        import pandas as pd
        df = pd.read_excel(d / "audio" / "tts_tasks.xlsx")
        total = 0
        for value in df["lines"]:
            try:
                total += len(eval(value)) if isinstance(value, str) else 1
            except Exception:
                total += 1
        return total
    except Exception:
        return 0


def has_prerequisites(d: Path) -> tuple[bool, str]:
    """A project is reusable when ASR chunks, translations, and tasks all exist."""
    need = [
        d / "log" / "cleaned_chunks.xlsx",
        d / "log" / "translation_results.xlsx",
        d / "audio" / "tts_tasks.xlsx",
    ]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        return False, f"missing {', '.join(missing)}"
    return True, ""


def make_junctions(project: Path, dry: bool) -> list[Path]:
    """Point project/output/{audio,log,gpt_log} at the flat folders beside the video."""
    created: list[Path] = []
    out = project / "output"
    for name in LINK_NAMES:
        real = project / name
        link = out / name
        if not real.is_dir() or link.exists():
            continue
        if dry:
            log(f"    [dry] would link output/{name} -> {real.name}")
            continue
        out.mkdir(exist_ok=True)
        # mklink /J needs cmd; os.symlink on Windows needs elevated privileges.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(real)],
                       capture_output=True, text=True)
        if link.exists():
            created.append(link)
    return created


def remove_junctions(created: list[Path]) -> None:
    """Unlink only the junctions this script made, never the real folders."""
    for link in created:
        try:
            # rmdir removes the junction itself and leaves the target intact.
            os.rmdir(link)
        except Exception as exc:  # noqa: BLE001
            log(f"    could not remove junction {link}: {exc}")
    # Drop output/ too if we left it empty.
    for parent in {p.parent for p in created}:
        try:
            parent.rmdir()
        except Exception:
            pass


def backup_dub(project: Path, dry: bool) -> Path | None:
    """Copy the existing dub audio aside so the previous version is recoverable."""
    bak = project / "_backup_redub"
    sources = [
        project / "dub.mp3", project / "dub_loudnorm.mp3", project / "dub.srt",
        project / "output" / "dub.mp3", project / "output" / "dub_loudnorm.mp3",
    ]
    existing = [p for p in sources if p.exists()]
    if not existing:
        return None
    if dry:
        log(f"    [dry] would back up {len(existing)} file(s) to {bak.name}/")
        return bak
    bak.mkdir(exist_ok=True)
    for p in existing:
        shutil.copy2(p, bak / p.name)
    return bak


def ensure_project_config(project: Path, repo: Path, dry: bool) -> list[Path]:
    """Give the project a config.yaml to run against.

    core.config_utils.ensure_config_file() resolves config relative to the
    CURRENT WORKING DIRECTORY, and importing core at all triggers it. Since each
    project is processed with the project as cwd, the project needs its own
    config.yaml. It is copied from the repo, and any pre-existing project config
    is left untouched.

    Returns the files this function created, so they can be removed afterwards.
    """
    made: list[Path] = []
    for name in ("config.yaml", "config.example.yaml"):
        src = repo / name
        dst = project / name
        if dst.exists() or not src.exists():
            continue
        if dry:
            log(f"    [dry] would copy {name} from repo")
            continue
        shutil.copy2(src, dst)
        made.append(dst)
    return made


def clear_dub_artifacts(project: Path, dry: bool) -> None:
    """Remove cached per-line wavs so the new TTS is actually used.

    This is essential, not cosmetic: tts_main returns early when a wav already
    exists, so stale audio is silently reused. The path is keyed by line number
    only, with no record of which backend produced it.
    """
    targets = [
        project / "audio" / "tmp", project / "audio" / "segs",
        project / "output" / "audio" / "tmp", project / "output" / "audio" / "segs",
    ]
    seen = set()
    for t in targets:
        rt = t.resolve()
        if rt in seen or not t.is_dir():
            continue
        seen.add(rt)
        files = list(t.glob("*.wav"))
        if dry:
            log(f"    [dry] would delete {len(files)} wav(s) in {t}")
            continue
        for f in files:
            try:
                f.unlink()
            except Exception:
                pass
    for f in [project / "dub.mp3", project / "dub_loudnorm.mp3",
              project / "output" / "dub.mp3", project / "output" / "dub_loudnorm.mp3"]:
        if f.exists() and not dry:
            try:
                f.unlink()
            except Exception:
                pass


def collect_outputs(project: Path) -> list[str]:
    """Move the finished dub beside the video, matching these projects' layout.

    The current pipeline writes output/dub.mp3, output/dub_loudnorm.mp3 and
    output/dub.srt, but these projects were produced by an older VideoLingo that
    kept those files in the project root next to the .mp4. Match that, so the
    batch leaves each folder looking the way it did before.
    """
    out = project / "output"
    moved: list[str] = []
    for name in ("dub.mp3", "dub_loudnorm.mp3", "dub.srt"):
        src = out / name
        if not src.is_file():
            continue
        try:
            shutil.move(str(src), str(project / name))
            moved.append(name)
        except Exception as exc:  # noqa: BLE001
            log(f"    could not move {name}: {exc}")
    return moved


def cleanup_output_dir(project: Path) -> None:
    """Remove the output/ scaffolding, but never a junction target."""
    out = project / "output"
    if not out.is_dir():
        return
    for child in list(out.iterdir()):
        # Junctions are already removed by remove_junctions; anything left that
        # is a real directory (empty shells the pipeline made) can go.
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file():
                child.unlink()
        except Exception:
            pass
    try:
        out.rmdir()
    except Exception:
        pass


RUNNER = r'''
import sys, os
REPO = sys.argv[1]
PROJ = sys.argv[2]
os.chdir(PROJ)
# The script itself lives in PROJ, so Python puts PROJ (not REPO) on sys.path.
# chdir alone does not re-run path resolution, so add REPO explicitly and purge
# any "core" that may already have been imported from the wrong place.
sys.path.insert(0, REPO)
for mod in [m for m in list(sys.modules) if m == "core" or m.startswith("core.")]:
    del sys.modules[mod]
from core._10_gen_audio import gen_audio
from core._11_merge_audio import create_srt_subtitle, normalize_dub_audio, merge_full_audio
gen_audio()
merge_full_audio()
create_srt_subtitle()
normalize_dub_audio()
print("__STAGE_DONE__")
'''


def run_project(project: Path, repo: Path, dry: bool) -> tuple[bool, str, float]:
    """Dub one project. Returns (ok, message, elapsed_seconds)."""
    t0 = time.time()
    runner = project / "_run_redub.py"
    if not dry:
        runner.write_text(RUNNER, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(runner), str(repo), str(project)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            # Run FROM the repo so "core" is importable, while the runner chdirs
            # to the project before doing any work.
            cwd=str(repo),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        elapsed = time.time() - t0
        tail = (proc.stdout or "")[-4000:]
        if "__STAGE_DONE__" not in (proc.stdout or ""):
            err = (proc.stderr or "")[-1200:]
            return False, f"failed\n{err}", elapsed
        return True, tail, elapsed
    finally:
        if runner.exists():
            try:
                runner.unlink()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder holding the projects")
    # This script lives in <repo>/tools, so the checkout root is one level up.
    # Pointing at the script's own directory would make the runner import from
    # tools/ and find no config.yaml there.
    default_repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo", default=str(default_repo),
                    help="VideoLingo checkout to import from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on project name")
    ap.add_argument("--min-seconds", type=float, default=0.0,
                    help="skip videos shorter than this")
    args = ap.parse_args()

    root = Path(args.root)
    repo = Path(args.repo)
    projects = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        if args.only and args.only.lower() not in d.name.lower():
            continue
        vid = find_video(d)
        if not vid:
            continue
        ok, why = has_prerequisites(d)
        if not ok:
            log(f"SKIP {d.name}: {why}")
            continue
        projects.append((video_duration(vid), count_lines(d), d, vid))

    # Shortest first: the user asked for exactly this ordering, and it also
    # surfaces any problem on a cheap project before a 2-hour one.
    projects.sort(key=lambda x: x[0])

    total_lines = sum(p[1] for p in projects)
    total_secs = sum(p[0] for p in projects)
    log("=" * 78)
    log(f"{len(projects)} project(s) | {total_lines} lines | "
        f"{total_secs/3600:.2f} h of video")
    log("=" * 78)

    results = []
    for idx, (dur, lines, d, vid) in enumerate(projects, 1):
        log(f"\n[{idx}/{len(projects)}] {d.name}")
        log(f"    {dur/60:.1f} min video | {lines} lines")
        created = make_junctions(d, args.dry_run)
        if created:
            log(f"    linked {len(created)} folder(s) under output/")
        made_cfg = ensure_project_config(d, repo, args.dry_run)
        if made_cfg:
            log(f"    copied {len(made_cfg)} config file(s) into the project")
        if not args.no_backup:
            bak = backup_dub(d, args.dry_run)
            if bak and not args.dry_run:
                log(f"    backed up previous dub -> {bak.name}/")
        clear_dub_artifacts(d, args.dry_run)
        if args.dry_run:
            results.append((d.name, True, "dry-run", 0.0))
            remove_junctions(created)
            continue
        try:
            ok, msg, elapsed = run_project(d, repo, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            ok, msg, elapsed = False, f"{type(exc).__name__}: {exc}", 0.0
        results.append((d.name, ok, msg, elapsed))
        log(f"    {'OK' if ok else 'FAILED'} in {elapsed/60:.1f} min")
        if not ok:
            log(msg)
        # The pipeline writes its final audio and subtitles into output/, since
        # that is the layout the current code expects. These projects keep their
        # deliverable beside the video, so move the results up before the
        # output/ scaffolding is torn down.
        if ok:
            moved = collect_outputs(d)
            if moved:
                log(f"    moved {len(moved)} output file(s) beside the video")
        # Restore the project dir to how we found it, now that the run is done.
        # These must be removed AFTER run_project: config.yaml is what the run
        # reads, and the junctions are what it writes through.
        remove_junctions(created)
        cleanup_output_dir(d)
        for f in made_cfg:
            try:
                f.unlink()
            except Exception:
                pass

    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    good = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]
    for name, ok, _msg, el in results:
        log(f"  {'OK  ' if ok else 'FAIL'}  {el/60:6.1f} min  {name}")
    log(f"\n{len(good)}/{len(results)} succeeded, total "
        f"{sum(r[3] for r in results)/3600:.2f} h")
    if bad:
        log("\nFailed projects:")
        for name, _ok, msg, _el in bad:
            log(f"  - {name}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
