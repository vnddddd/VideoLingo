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
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
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
# Windows legacy locales such as GBK cannot encode emoji/checkmark output used
# by rich in downstream core modules; replace unencodable characters instead of
# crashing the background Streamlit subprocess.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass


def _process_identity(pid: int) -> str | None:
    """Return a best-effort process identity token so PID reuse does not fool the watchdog.

    Windows: process creation FILETIME via ctypes (no external command, no wmic dependency).
    POSIX:   field 22 (starttime) of /proc/<pid>/stat.
    """
    if pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_t = wintypes.FILETIME()
                kernel_t = wintypes.FILETIME()
                user_t = wintypes.FILETIME()
                ok = kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_t),
                    ctypes.byref(kernel_t),
                    ctypes.byref(user_t),
                )
                if not ok:
                    return None
                return f"{creation.dwHighDateTime:08x}{creation.dwLowDateTime:08x}"
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None

    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat_text = stat_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except Exception:
        return "unknown"
    fields = stat_text.rsplit(")", 1)[-1].strip().split()
    if len(fields) >= 20:
        # /proc/[pid]/stat field 22 (starttime) is stable for the lifetime of the process.
        return fields[19]
    return "unknown"


def _process_alive(pid: int) -> bool:
    """Lightweight PID-only liveness probe used as a safe fallback."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except Exception:
            return True  # transient probe failure: assume alive, do not falsely kill child
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _process_matches(pid: int, identity: str | None) -> bool:
    if not _process_alive(pid):
        return False
    if identity is None or identity == "unknown":
        return True
    current_identity = _process_identity(pid)
    if current_identity is None or current_identity == "unknown":
        # PID exists but we can't fingerprint right now; prefer keeping child alive over spurious kill.
        return True
    return current_identity == identity


def _start_parent_watchdog() -> None:
    """Exit this split CLI if the Streamlit parent process disappears or its PID is reused."""
    parent_pid_text = os.environ.get("VIDEOLINGO_PARENT_PID")
    if not parent_pid_text:
        return
    try:
        parent_pid = int(parent_pid_text)
    except ValueError:
        return
    if parent_pid <= 0 or parent_pid == os.getpid():
        return

    parent_identity = os.environ.get("VIDEOLINGO_PARENT_IDENTITY") or _process_identity(parent_pid)

    def _watch() -> None:
        while True:
            time.sleep(2.0)
            if not _process_matches(parent_pid, parent_identity):
                print("[WARN] Parent VideoLingo process disappeared; stopping split pipeline.", file=sys.stderr, flush=True)
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    finally:
                        os._exit(130)
                else:
                    try:
                        os.killpg(os.getpid(), signal.SIGTERM)
                    except Exception:
                        os._exit(130)

    threading.Thread(target=_watch, name="videolingo-parent-watchdog", daemon=True).start()

DUB_AUDIO_FILE = Path("output/dub.mp3")
DUB_SUB_FILE = Path("output/dub.srt")
DUB_VIDEO_FILE = Path("output/output_dub.mp4")
OUTPUT_DIR = Path("output")
AUDIO_DIR = Path("output/audio")
RAW_AUDIO_FILE = Path("output/audio/raw.mp3")
VOCAL_AUDIO_FILE = Path("output/audio/vocal.mp3")
BACKGROUND_AUDIO_FILE = Path("output/audio/background.mp3")

# Output checkpoints used to make the local split pipeline resumable. These
# mirror core/utils/models.py and the subtitle/audio files produced by _6/_10/_11.
CLEANED_CHUNKS_FILE = Path("output/log/cleaned_chunks.xlsx")
SPLIT_BY_NLP_FILE = Path("output/log/split_by_nlp.txt")
SPLIT_BY_MEANING_FILE = Path("output/log/split_by_meaning.txt")
TERMINOLOGY_FILE = Path("output/log/terminology.json")
TRANSLATION_FILE = Path("output/log/translation_results.xlsx")
SPLIT_SUB_FILE = Path("output/log/translation_results_for_subtitles.xlsx")
REMERGED_FILE = Path("output/log/translation_results_remerged.xlsx")
AUDIO_TASK_FILE = Path("output/audio/tts_tasks.xlsx")
SRC_SUB_FILE = Path("output/src.srt")
TRANS_SUB_FILE = Path("output/trans.srt")
SRC_TRANS_SUB_FILE = Path("output/src_trans.srt")
TRANS_SRC_SUB_FILE = Path("output/trans_src.srt")
AUDIO_SRC_SUB_FILE = Path("output/audio/src_subs_for_audio.srt")
AUDIO_TRANS_SUB_FILE = Path("output/audio/trans_subs_for_audio.srt")
AUDIO_REFERS_DIR = Path("output/audio/refers")
AUDIO_SEGS_DIR = Path("output/audio/segs")


def _safe_print(message: str) -> None:
    """Print without crashing on legacy Windows encodings such as GBK."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_message)


def _rprint(message: str) -> None:
    try:
        from rich import print as rich_print
    except Exception:
        _safe_print(message)
    else:
        try:
            rich_print(message)
        except UnicodeError:
            _safe_print(message)

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


def _checkpoint_complete(outputs: Sequence[Path | str]) -> bool:
    """Return True only when every expected output exists and non-empty files are not zero-byte."""
    if not outputs:
        return False
    for output in outputs:
        path = Path(output)
        if not path.exists():
            return False
        if path.is_file() and path.stat().st_size == 0:
            return False
    return True


def _directory_has_files(path: Path | str, pattern: str = "*") -> bool:
    directory = Path(path)
    return directory.is_dir() and any(item.is_file() for item in directory.glob(pattern))


def _run_if_missing(label: str, outputs: Sequence[Path | str], func: Callable[[], object]) -> None:
    if _checkpoint_complete(outputs):
        output_text = ", ".join(_display_path(path) for path in outputs)
        print(f"[SKIP] {label}: {output_text} already exists.")
        return
    func()
    _require_files(outputs, f"{label} output")


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


LOCAL_STEP_ALIASES = {
    "asr": 0,
    "split": 1,
    "translate": 2,
    "subtitles": 3,
    "timeline": 4,
    "audio-tasks": 5,
    "reference-audio": 6,
    "tts-merge": 7,
}


def _read_xlsx_header_without_openpyxl(path: Path) -> set[str]:
    """Read the first-row cell values from a simple .xlsx without pandas/openpyxl."""
    try:
        with zipfile.ZipFile(path) as archive:
            sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="replace")
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_xml = archive.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
                shared_strings = [html.unescape(value) for value in re.findall(r"<t[^>]*>(.*?)</t>", shared_xml, flags=re.S)]
    except Exception:
        return set()

    row_match = re.search(r"<row[^>]*\br=[\"']1[\"'][^>]*>(.*?)</row>", sheet_xml, flags=re.S)
    if not row_match:
        return set()

    columns: set[str] = set()
    for cell in re.findall(r"<c\b([^>]*)>(.*?)</c>", row_match.group(1), flags=re.S):
        attrs, body = cell
        cell_type_match = re.search(r"\bt=[\"']([^\"']+)[\"']", attrs)
        cell_type = cell_type_match.group(1) if cell_type_match else ""
        value = ""
        if cell_type == "inlineStr":
            parts = re.findall(r"<t[^>]*>(.*?)</t>", body, flags=re.S)
            value = "".join(html.unescape(part) for part in parts)
        else:
            value_match = re.search(r"<v>(.*?)</v>", body, flags=re.S)
            if value_match:
                value = html.unescape(value_match.group(1))
                if cell_type == "s":
                    try:
                        value = shared_strings[int(value)]
                    except Exception:
                        pass
        if value:
            columns.add(value)
    return columns


def _audio_task_has_columns(required_columns: Sequence[str]) -> bool:
    if not _checkpoint_complete([AUDIO_TASK_FILE]):
        return False
    try:
        import pandas as pd

        columns = set(pd.read_excel(AUDIO_TASK_FILE, nrows=0).columns)
    except Exception:
        columns = _read_xlsx_header_without_openpyxl(AUDIO_TASK_FILE)
    return set(required_columns).issubset(columns)


def _require_audio_task_columns(required_columns: Sequence[str], context: str) -> None:
    if not _audio_task_has_columns(required_columns):
        missing = ", ".join(required_columns)
        raise RuntimeError(f"Missing required column(s) for {context}: {missing} in {_display_path(AUDIO_TASK_FILE)}")


def _expected_audio_segment_files() -> list[Path]:
    """Return output/audio/segs/*.wav files expected from tts_tasks.xlsx."""
    _require_audio_task_columns(["number", "lines"], "TTS segment checkpoint")
    import pandas as pd

    df = pd.read_excel(AUDIO_TASK_FILE)
    expected: list[Path] = []
    for _, row in df.iterrows():
        number = row["number"]
        lines = row["lines"]
        if isinstance(lines, str):
            lines = eval(lines)
        for line_index in range(len(lines)):
            expected.append(AUDIO_SEGS_DIR / f"{number}_{line_index}.wav")
    return expected


def _all_expected_audio_segments_exist() -> bool:
    try:
        expected = _expected_audio_segment_files()
    except Exception:
        return False
    return bool(expected) and _checkpoint_complete(expected)


def _require_expected_audio_segments(context: str) -> None:
    expected = _expected_audio_segment_files()
    if not expected:
        raise RuntimeError(f"No expected audio segment(s) found for {context} in {_display_path(AUDIO_TASK_FILE)}")
    _require_files(expected, context)


def _text_and_audio_steps() -> list[tuple[str, Callable[[], object]]]:
    """Match st.py's normal flow but stop after core/_11_merge_audio.py.

    Each step has an output checkpoint guard. Re-running the command after a
    pause/stop/crash skips completed work and resumes at the first missing
    checkpoint instead of starting from Whisper/translation again.
    """
    def _core_module(module_name: str):
        """Import a core step only when that guarded step actually needs it."""
        import importlib

        return importlib.import_module(f"core.{module_name}")

    def sentence_segmentation() -> None:
        _3_1_split_nlp = _core_module("_3_1_split_nlp")
        _3_2_split_meaning = _core_module("_3_2_split_meaning")
        _run_if_missing("NLP sentence split", [SPLIT_BY_NLP_FILE], _3_1_split_nlp.split_by_spacy)
        _run_if_missing("Meaning sentence split", [SPLIT_BY_MEANING_FILE], _3_2_split_meaning.split_sentences_by_meaning)

    def summarize_and_translate() -> None:
        _4_1_summarize = _core_module("_4_1_summarize")
        _4_2_translate = _core_module("_4_2_translate")
        _run_if_missing("Summarization", [TERMINOLOGY_FILE], _4_1_summarize.get_summary)
        _run_if_missing("Translation", [TRANSLATION_FILE], _4_2_translate.translate_all)

    def audio_tasks_and_chunks() -> None:
        # Avoid importing TTS task modules when the workbook is already complete;
        # their optional dependencies are only needed if this step must run.
        if _audio_task_has_columns(["lines", "src_lines"]):
            print(f"[SKIP] Audio task/chunk generation: {_display_path(AUDIO_TASK_FILE)} already has chunk metadata.")
            return

        if _checkpoint_complete([AUDIO_TASK_FILE]):
            print(f"[SKIP] Audio task generation: {_display_path(AUDIO_TASK_FILE)} already exists.")
        else:
            _8_1_audio_task = _core_module("_8_1_audio_task")
            _8_1_audio_task.gen_audio_task_main()
            _require_files([AUDIO_TASK_FILE], "Audio task generation output")

        # _8_2 augments the same workbook with chunk columns; require columns
        # used later by _11 rather than checking only that the workbook exists.
        if _audio_task_has_columns(["lines", "src_lines"]):
            print(f"[SKIP] Audio chunk generation: {_display_path(AUDIO_TASK_FILE)} already has chunk metadata.")
        else:
            _8_2_dub_chunks = _core_module("_8_2_dub_chunks")
            _8_2_dub_chunks.gen_dub_chunks()
            _require_audio_task_columns(["lines", "src_lines"], "Audio chunk generation output")

    def reference_audio() -> None:
        # When demucs is disabled, _9 intentionally skips extraction; do not
        # force the refers directory to exist in that mode.
        from core.utils.config_utils import load_key

        if not bool(load_key("demucs")):
            print("[SKIP] Reference audio extraction: config demucs=false.")
            return
        if _directory_has_files(AUDIO_REFERS_DIR, "*.wav"):
            print(f"[SKIP] Reference audio extraction: {_display_path(AUDIO_REFERS_DIR)} already contains wav files.")
            return
        _9_refer_audio = _core_module("_9_refer_audio")
        _9_refer_audio.extract_refer_audio_main()
        if not _directory_has_files(AUDIO_REFERS_DIR, "*.wav"):
            raise FileNotFoundError(f"Missing required file(s) for Reference audio extraction output:\n  - {_display_path(AUDIO_REFERS_DIR)}/*.wav")

    def tts_and_merge() -> None:
        if _checkpoint_complete([DUB_AUDIO_FILE, DUB_SUB_FILE]):
            print(f"[SKIP] TTS/audio merge: {_display_path(DUB_AUDIO_FILE)} and {_display_path(DUB_SUB_FILE)} already exist.")
            return
        if _all_expected_audio_segments_exist():
            print(f"[SKIP] TTS segment generation: all expected wav files already exist in {_display_path(AUDIO_SEGS_DIR)}.")
        else:
            _10_gen_audio = _core_module("_10_gen_audio")
            _10_gen_audio.gen_audio()
        _require_expected_audio_segments("TTS segment generation output")
        _11_merge_audio = _core_module("_11_merge_audio")
        _run_if_missing("Final dub audio/subtitle merge", [DUB_AUDIO_FILE, DUB_SUB_FILE], _11_merge_audio.merge_full_audio)

    def transcribe_step() -> None:
        _2_asr = _core_module("_2_asr")
        _run_if_missing("WhisperX word-level transcription", [CLEANED_CHUNKS_FILE], _2_asr.transcribe)

    def split_subtitles() -> None:
        _5_split_sub = _core_module("_5_split_sub")
        _run_if_missing("Cut and align long subtitles", [SPLIT_SUB_FILE, REMERGED_FILE], _5_split_sub.split_for_sub_main)

    def generate_timeline() -> None:
        _6_gen_sub = _core_module("_6_gen_sub")
        _run_if_missing("Generate timeline and subtitles", [SRC_SUB_FILE, TRANS_SUB_FILE, SRC_TRANS_SUB_FILE, TRANS_SRC_SUB_FILE, AUDIO_SRC_SUB_FILE, AUDIO_TRANS_SUB_FILE], _6_gen_sub.align_timestamp_main)

    return [
        ("WhisperX word-level transcription", transcribe_step),
        ("Sentence segmentation using NLP and LLM", sentence_segmentation),
        ("Summarization and multi-step translation", summarize_and_translate),
        ("Cut and align long subtitles", split_subtitles),
        ("Generate timeline and subtitles", generate_timeline),
        ("Generate audio tasks and chunks", audio_tasks_and_chunks),
        ("Extract reference audio", reference_audio),
        ("Generate audio and merge into dub.mp3/dub.srt", tts_and_merge),
    ]


def _local_step_by_alias(alias: str) -> tuple[str, Callable[[], object]]:
    steps = _text_and_audio_steps()
    try:
        index = LOCAL_STEP_ALIASES[alias]
    except KeyError as exc:
        choices = ", ".join(LOCAL_STEP_ALIASES)
        raise RuntimeError(f"Unknown local step '{alias}'. Valid choices: {choices}") from exc
    return steps[index]


def _require_local_inputs() -> None:
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


def cmd_local_step(args: argparse.Namespace) -> None:
    """Run one guarded local split-pipeline step by alias."""
    _require_local_inputs()
    label, func = _local_step_by_alias(args.step)
    _rprint(f"\n[bold cyan]▶ Local step: {label}[/bold cyan]")
    func()


def cmd_local_until_audio(args: argparse.Namespace) -> None:
    """Run the local VideoLingo pipeline and stop before final video render."""
    _require_local_inputs()
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

    local_step = subparsers.add_parser(
        "local-step",
        help="Run one guarded local split-pipeline step; used by the Streamlit split UI for resumable progress.",
    )
    local_step.add_argument("step", choices=tuple(LOCAL_STEP_ALIASES.keys()))
    local_step.set_defaults(func=cmd_local_step)

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
    _start_parent_watchdog()
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
