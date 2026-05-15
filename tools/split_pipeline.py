#!/usr/bin/env python3
"""Split VideoLingo into local/remote pipeline stages.

This is a small non-UI command entry point for the workflow:

1. remote-demucs / prep-audio
   Convert the source video in output/ to output/audio/raw.mp3 and generate
   output/audio/vocal.mp3 + output/audio/background.mp3 with Demucs.

2. local-stop-before-video / local-until-audio
   Run VideoLingo text + dubbing pipeline through core/_11_merge_audio.py,
   producing output/dub.mp3 and output/dub.srt, but intentionally skip
   core/_12_dub_to_vid.py.

3. pack-render-inputs
   Validate/list the minimal files that must be copied to the render machine.
   Optionally create a zip package.

4. remote-render / render
   Validate render inputs and run only core/_12_dub_to_vid.py to produce
   output/output_dub.mp4.

Run from the VideoLingo project root, for example:
    python tools/split_pipeline.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep all relative paths compatible with VideoLingo's existing modules, which
# assume the current working directory is the project root.
os.chdir(PROJECT_ROOT)

# Do not import VideoLingo's heavy modules at file import time.  Some optional
# packages (for example demucs on the local/text machine) may be intentionally
# absent, but `python tools/split_pipeline.py --help` and status checks should
# still work.  Command handlers import the modules they actually need lazily.

DUB_AUDIO_FILE = Path("output/dub.mp3")
DUB_SUB_FILE = Path("output/dub.srt")
DUB_VIDEO_FILE = Path("output/output_dub.mp4")
OUTPUT_DIR = Path("output")
AUDIO_DIR = Path("output/audio")
RAW_AUDIO_FILE = Path("output/audio/raw.mp3")
VOCAL_AUDIO_FILE = Path("output/audio/vocal.mp3")
BACKGROUND_AUDIO_FILE = Path("output/audio/background.mp3")


def _rprint(message: str) -> None:
    try:
        from rich import print as rich_print
    except Exception:
        print(message)
    else:
        rich_print(message)

REMOTE_TO_LOCAL_FILES = [
    RAW_AUDIO_FILE,
    VOCAL_AUDIO_FILE,
    BACKGROUND_AUDIO_FILE,
]

LOCAL_TO_REMOTE_FILES = [
    DUB_AUDIO_FILE,
    DUB_SUB_FILE,
    BACKGROUND_AUDIO_FILE,
]

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def _display_path(path: Path | str) -> str:
    return str(Path(path).as_posix())


def _exists(path: Path | str) -> bool:
    return Path(path).exists()


def _require_files(paths: Iterable[Path | str], context: str) -> None:
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        missing_lines = "\n".join(f"  - {_display_path(path)}" for path in missing)
        raise FileNotFoundError(f"Missing required file(s) for {context}:\n{missing_lines}")


def _find_unique_video() -> Path:
    """Return the unique source video in output/ without importing VideoLingo's heavy modules."""
    candidates: list[Path] = []
    if OUTPUT_DIR.exists():
        candidates = [
            path for path in OUTPUT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and not path.name.startswith("output")
        ]
    if len(candidates) != 1:
        candidate_text = "\n".join(f"  - {_display_path(path)}" for path in candidates) or "  (none)"
        raise RuntimeError(
            f"Could not resolve a unique source video under output/; found {len(candidates)}. "
            "Keep exactly one source video there before running split pipeline commands.\n"
            f"Candidates:\n{candidate_text}"
        )
    return candidates[0]


def _print_file_status(paths: Sequence[Path | str], title: str) -> None:
    print(f"\n[{title}]")
    for path in paths:
        p = Path(path)
        mark = "OK" if p.exists() else "MISSING"
        print(f"  {mark:7} {_display_path(p)}")


def _run_steps(steps: Sequence[tuple[str, Callable[[], object]]]) -> None:
    for index, (label, func) in enumerate(steps, 1):
        _rprint(f"\n[bold cyan]▶ Step {index}/{len(steps)}: {label}[/bold cyan]")
        func()


def cmd_status(args: argparse.Namespace) -> None:
    """Print the split pipeline file contract and current local status."""
    video = None
    try:
        video = _find_unique_video()
    except Exception as exc:
        print(f"[WARN] source video: {exc}")

    remote_to_local = list(REMOTE_TO_LOCAL_FILES)
    local_to_remote = list(LOCAL_TO_REMOTE_FILES)
    if video is not None:
        remote_to_local.append(video)
        local_to_remote.append(video)

    _print_file_status(remote_to_local, "remote -> local required after prep-audio")
    _print_file_status(local_to_remote, "local -> remote required before render")
    _print_file_status([DUB_VIDEO_FILE], "render output")


def cmd_prep_audio(args: argparse.Namespace) -> None:
    """Prepare raw/vocal/background audio on the GPU-capable machine."""
    video = _find_unique_video()
    print(f"Source video: {_display_path(video)}")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_AUDIO_FILE.exists():
        print(f"[SKIP] {_display_path(RAW_AUDIO_FILE)} already exists.")
    else:
        from core.asr_backend.audio_preprocess import convert_video_to_audio

        convert_video_to_audio(str(video))

    if args.no_demucs:
        print("[SKIP] --no-demucs was set; only raw audio was prepared.")
    elif VOCAL_AUDIO_FILE.exists() and BACKGROUND_AUDIO_FILE.exists():
        print(
            f"[SKIP] {_display_path(VOCAL_AUDIO_FILE)} and "
            f"{_display_path(BACKGROUND_AUDIO_FILE)} already exist."
        )
    else:
        from core.asr_backend.demucs_vl import demucs_audio

        _require_files([RAW_AUDIO_FILE], "Demucs audio separation")
        demucs_audio()

    required = [RAW_AUDIO_FILE]
    if not args.no_demucs:
        required += [VOCAL_AUDIO_FILE, BACKGROUND_AUDIO_FILE]
    _require_files(required, "prep-audio output")
    _print_file_status(required, "prep-audio output")


def _text_and_audio_steps() -> list[tuple[str, Callable[[], object]]]:
    """Match st.py's normal flow but stop after core/_11_merge_audio.py."""
    from core import (
        _2_asr,
        _3_1_split_nlp,
        _3_2_split_meaning,
        _4_1_summarize,
        _4_2_translate,
        _5_split_sub,
        _6_gen_sub,
        _8_1_audio_task,
        _8_2_dub_chunks,
        _9_refer_audio,
        _10_gen_audio,
        _11_merge_audio,
    )

    return [
        ("WhisperX word-level transcription", _2_asr.transcribe),
        (
            "Sentence segmentation using NLP and LLM",
            lambda: (
                _3_1_split_nlp.split_by_spacy(),
                _3_2_split_meaning.split_sentences_by_meaning(),
            ),
        ),
        (
            "Summarization and multi-step translation",
            lambda: (_4_1_summarize.get_summary(), _4_2_translate.translate_all()),
        ),
        ("Cut and align long subtitles", _5_split_sub.split_for_sub_main),
        ("Generate timeline and subtitles", _6_gen_sub.align_timestamp_main),
        ("Generate audio tasks and chunks", lambda: (_8_1_audio_task.gen_audio_task_main(), _8_2_dub_chunks.gen_dub_chunks())),
        ("Extract reference audio", _9_refer_audio.extract_refer_audio),
        ("Generate audio and merge into dub.mp3/dub.srt", lambda: (_10_gen_audio.gen_audio(), _11_merge_audio.merge_full_audio())),
    ]


def cmd_local_until_audio(args: argparse.Namespace) -> None:
    """Run the local VideoLingo pipeline and stop before final video render."""
    video = _find_unique_video()
    print(f"Source video: {_display_path(video)}")

    from core.utils.config_utils import load_key

    demucs_enabled = bool(load_key("demucs"))
    required_inputs: list[Path] = [video, RAW_AUDIO_FILE]
    if demucs_enabled:
        required_inputs += [VOCAL_AUDIO_FILE, BACKGROUND_AUDIO_FILE]
    else:
        print("[INFO] config demucs=false; vocal/background are not required for ASR.")
    _require_files(required_inputs, "local-stop-before-video input")

    _run_steps(_text_and_audio_steps())
    _require_files([DUB_AUDIO_FILE, DUB_SUB_FILE], "local-stop-before-video output")
    print("\n[OK] Local pipeline stopped before final video render.")
    _print_file_status([DUB_AUDIO_FILE, DUB_SUB_FILE], "local output")


def _manifest_for_render() -> list[Path]:
    video = _find_unique_video()
    files = [video, DUB_AUDIO_FILE, DUB_SUB_FILE]
    if BACKGROUND_AUDIO_FILE.exists():
        files.append(BACKGROUND_AUDIO_FILE)
    elif RAW_AUDIO_FILE.exists():
        # core/_12_dub_to_vid.py can fall back to raw.mp3 if background.mp3 is absent.
        files.append(RAW_AUDIO_FILE)
    else:
        files.append(BACKGROUND_AUDIO_FILE)
    return files


def cmd_pack_render_inputs(args: argparse.Namespace) -> None:
    """Validate/list files needed by the render machine; optionally zip them."""
    files = _manifest_for_render()
    _print_file_status(files, "render input manifest")
    _require_files(files, "pack-render-inputs")

    manifest = {
        "project_root": str(PROJECT_ROOT),
        "files": [_display_path(path) for path in files],
        "render_command": "python tools/split_pipeline.py remote-render",
        "notes": [
            "Copy these paths relative to the VideoLingo project root on the render machine.",
            "Keep exactly one source video in output/ before running remote-render.",
            "background.mp3 is preferred; raw.mp3 is accepted only as fallback by _12_dub_to_vid.py.",
        ],
    }
    print("\n[manifest]")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.zip:
        zip_path = Path(args.zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True) if zip_path.parent != Path("") else None
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, arcname=path.as_posix())
            zf.writestr("split_render_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"\n[OK] Wrote render input package: {_display_path(zip_path)}")


def _ffmpeg_probe() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found in PATH; remote-render cannot run.")
    result = subprocess.run([ffmpeg, "-version"], text=True, capture_output=True)
    first_line = (result.stdout or result.stderr).splitlines()[0] if (result.stdout or result.stderr) else ffmpeg
    print(f"ffmpeg: {first_line}")


def cmd_remote_render(args: argparse.Namespace) -> None:
    """Run only the final VideoLingo render stage on the render machine."""
    files = _manifest_for_render()
    _print_file_status(files, "remote-render input")
    _require_files(files, "remote-render input")
    _ffmpeg_probe()
    from core import _12_dub_to_vid

    _12_dub_to_vid.merge_video_audio()
    _require_files([DUB_VIDEO_FILE], "remote-render output")
    print(f"\n[OK] Rendered {_display_path(DUB_VIDEO_FILE)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VideoLingo as a split local/remote pipeline without changing the Streamlit UI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show current file-contract status.").set_defaults(func=cmd_status)

    prep = subparsers.add_parser(
        "prep-audio",
        aliases=["remote-demucs"],
        help="Create output/audio/raw.mp3 and Demucs vocal/background tracks.",
    )
    prep.add_argument("--no-demucs", action="store_true", help="Only create raw.mp3; skip vocal/background generation.")
    prep.set_defaults(func=cmd_prep_audio)

    local = subparsers.add_parser(
        "local-stop-before-video",
        aliases=["local-until-audio"],
        help="Run text/TTS/audio pipeline through _11_merge_audio.py, then stop before _12.",
    )
    local.set_defaults(func=cmd_local_until_audio)

    pack = subparsers.add_parser(
        "pack-render-inputs",
        help="Validate/list files to copy to the render machine, optionally creating a zip.",
    )
    pack.add_argument(
        "--zip",
        "--out",
        dest="zip",
        metavar="PATH",
        help="Optional zip package path, e.g. output/render_inputs.zip. --out is kept as a compatibility alias.",
    )
    pack.set_defaults(func=cmd_pack_render_inputs)

    render = subparsers.add_parser(
        "remote-render",
        aliases=["render"],
        help="Run only _12_dub_to_vid.py on prepared render inputs.",
    )
    render.set_defaults(func=cmd_remote_render)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
