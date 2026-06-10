#!/usr/bin/env python3
"""Local HTTP bridge for the Chrome YouTube overlay extension.

The bridge intentionally uses only the Python standard library. It listens on
127.0.0.1 and runs one VideoLingo job at a time because the upstream pipeline
uses the shared output/ workspace and checkpoint files.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

HOST = "127.0.0.1"
PORT = 8765
OUTPUT_DIR = PROJECT_ROOT / "output"
RUNTIME_DIR = PROJECT_ROOT / "runtime" / "browser_bridge"
JOBS_DIR = RUNTIME_DIR / "jobs"
LOGS_DIR = RUNTIME_DIR / "logs"
INDEX_FILE = RUNTIME_DIR / "index.json"
COOKIE_FILE = PROJECT_ROOT / "cookies" / "browser_bridge_youtube.txt"
PIPELINE_SCRIPT = PROJECT_ROOT / "tools" / "split_pipeline.py"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
CONFIG_EXAMPLE_FILE = PROJECT_ROOT / "config.example.yaml"
PLUGIN_AUDIO_FILE = "dub_loudnorm.mp3"
WORK_OUTPUT_DIRNAME = "work_output"
JOB_COOKIE_FILE = "cookies.txt"
OUTPUT_JOB_MARKER = ".browser_bridge_job.json"
# Keep plugin playback loudness aligned with core/_12_dub_to_vid.py final video output.
PLUGIN_AUDIO_LOUDNORM_FILTER = "loudnorm=I=-13:TP=-1.5:LRA=11"

jobs_lock = threading.RLock()
run_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _project_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _project_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _job_work_output_dir(job_id: str) -> Path:
    return _job_dir(job_id) / WORK_OUTPUT_DIRNAME


def _job_cookie_file(job_id: str) -> Path:
    return _job_dir(job_id) / JOB_COOKIE_FILE


def _ensure_config_file() -> None:
    if CONFIG_FILE.exists():
        return
    if not CONFIG_EXAMPLE_FILE.exists():
        raise FileNotFoundError(f"{CONFIG_FILE.name} not found and {CONFIG_EXAMPLE_FILE.name} is missing")
    shutil.copy2(CONFIG_EXAMPLE_FILE, CONFIG_FILE)


def _output_marker_path() -> Path:
    return OUTPUT_DIR / OUTPUT_JOB_MARKER


def _output_marker_job_id() -> str | None:
    marker = _output_marker_path()
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    return str(data.get("job_id") or "") or None


def _write_output_marker(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id, {})
        payload = {
            "job_id": job_id,
            "video_key": job.get("video_key"),
            "video_id": job.get("video_id"),
            "updated_at": _now(),
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _output_marker_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    output_root = OUTPUT_DIR.resolve()
    runtime_root = RUNTIME_DIR.resolve()
    if resolved != output_root and runtime_root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside bridge workspace: {path}")
    if path.exists():
        shutil.rmtree(path)


def _serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"thread", "cookies"}}


def _persist_index_unlocked() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _now(),
        "jobs": [
            _serialize_job(job)
            for job in sorted(jobs.values(), key=lambda item: item.get("created_at", 0), reverse=True)
        ],
    }
    tmp_path = INDEX_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(INDEX_FILE)


def _queue_position_unlocked(job_id: str) -> int | None:
    queued = [
        job
        for job in sorted(jobs.values(), key=lambda item: item.get("created_at", 0))
        if job.get("status") == "queued"
    ]
    for index, job in enumerate(queued, start=1):
        if job.get("id") == job_id:
            return index
    return None


def _sanitize_video_id(video_id: str) -> str:
    clean = video_id.strip().split("?")[0].split("&")[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", clean):
        raise ValueError("Could not read a valid YouTube video id from this URL")
    return clean


def _youtube_video_from_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")):
        raise ValueError("Only YouTube URLs are supported")

    video_id = ""
    parts = [part for part in parsed.path.split("/") if part]
    if host == "youtu.be":
        video_id = parts[0] if parts else ""
    else:
        query = parse_qs(parsed.query)
        video_id = (query.get("v") or [""])[0]
        if not video_id and len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            video_id = parts[1]

    video_id = _sanitize_video_id(video_id)
    video_key = f"youtube:{video_id}"
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    return video_key, video_id, canonical_url


def _ensure_cached_job_ready(job: dict[str, Any]) -> bool:
    if job.get("status") != "done":
        return False
    job_id = str(job.get("id") or "")
    if not job_id:
        return False
    job_dir = JOBS_DIR / job_id
    subtitle = job_dir / "dub.srt"
    normalized_audio = job_dir / PLUGIN_AUDIO_FILE
    if subtitle.exists() and normalized_audio.exists():
        return True
    raw_audio = job_dir / "dub.mp3"
    if subtitle.exists() and raw_audio.exists():
        _normalize_plugin_audio(raw_audio, normalized_audio)
        return normalized_audio.exists()
    return False


def _load_index() -> None:
    if not INDEX_FILE.exists():
        return
    try:
        payload = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read {INDEX_FILE}: {exc}", flush=True)
        return

    changed = False
    loaded_jobs = payload.get("jobs") if isinstance(payload, dict) else []
    if not isinstance(loaded_jobs, list):
        return

    with jobs_lock:
        for item in loaded_jobs:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            job = dict(item)
            job.pop("thread", None)
            job.pop("cookies", None)
            original_status = job.get("status")
            jobs[str(job["id"])] = job
            if original_status == "running":
                try:
                    marker_job_id = _output_marker_job_id()
                    if marker_job_id is None or marker_job_id == str(job["id"]):
                        _snapshot_output_to_job_workspace(str(job["id"]))
                        job["resume_note"] = "Recovered partial output after bridge restart."
                    else:
                        job["resume_note"] = (
                            "Bridge restarted, but current output belongs to another job; "
                            "kept the previous saved work_output."
                        )
                except Exception as exc:  # noqa: BLE001
                    job["resume_note"] = f"Could not recover current output after bridge restart: {exc}"
                job["status"] = "queued"
                job["phase"] = "queued"
                job.pop("error", None)
                job["updated_at"] = _now()
                changed = True
            elif original_status == "queued":
                job["phase"] = "queued"
                job.pop("error", None)
                job["updated_at"] = _now()
                changed = True
            elif job.get("status") == "done" and not _ensure_cached_job_ready(job):
                job["status"] = "error"
                job["phase"] = "error"
                job["error"] = "Cached output files are missing. Submit this video again."
                job["updated_at"] = _now()
                changed = True
        if changed:
            _persist_index_unlocked()


def _find_latest_video_job_unlocked(video_key: str) -> dict[str, Any] | None:
    candidates = [
        job
        for job in jobs.values()
        if job.get("video_key") == video_key and job.get("status") in {"queued", "running", "done", "error"}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    for job in candidates:
        if job.get("status") != "done" or _ensure_cached_job_ready(job):
            return job
        job["status"] = "error"
        job["phase"] = "error"
        job["error"] = "Cached output files are missing. Submit this video again."
        job["updated_at"] = _now()
    _persist_index_unlocked()
    return None


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    data = {key: value for key, value in job.items() if key not in {"thread", "cookies"}}
    if job.get("status") == "queued":
        with jobs_lock:
            data["queue_position"] = _queue_position_unlocked(str(job.get("id") or ""))
    if job.get("status") == "done":
        job_id = job["id"]
        data["outputs"] = {
            "audio": f"/jobs/{job_id}/{PLUGIN_AUDIO_FILE}",
            "subtitle": f"/jobs/{job_id}/dub.srt",
            "log": f"/jobs/{job_id}/log.txt",
        }
    return data


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)
            jobs[job_id]["updated_at"] = _now()
            _persist_index_unlocked()


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    _write_cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(body)


def _write_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Private-Network", "true")
    handler.send_header("Access-Control-Max-Age", "86400")


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length_text = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_text)
    except ValueError:
        length = 0
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _active_job() -> dict[str, Any] | None:
    with jobs_lock:
        running = [
            job
            for job in jobs.values()
            if job.get("status") == "running"
        ]
        if running:
            return sorted(running, key=lambda item: item.get("updated_at", 0), reverse=True)[0]
        queued = [
            job
            for job in jobs.values()
            if job.get("status") == "queued"
        ]
        if queued:
            return sorted(queued, key=lambda item: item.get("created_at", 0))[0]
    return None


def _restore_job_workspace(job_id: str) -> None:
    work_output = _job_work_output_dir(job_id)
    _job_dir(job_id).mkdir(parents=True, exist_ok=True)
    _safe_rmtree(OUTPUT_DIR)
    if work_output.exists():
        shutil.copytree(work_output, OUTPUT_DIR)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_output_marker(job_id)


def _snapshot_output_to_job_workspace(job_id: str) -> None:
    work_output = _job_work_output_dir(job_id)
    _job_dir(job_id).mkdir(parents=True, exist_ok=True)
    _safe_rmtree(work_output)
    if OUTPUT_DIR.exists():
        shutil.copytree(OUTPUT_DIR, work_output)


def _raw_audio_file() -> Path:
    return OUTPUT_DIR / "audio" / "raw.mp3"


def _raw_audio_ready() -> bool:
    raw_audio = _raw_audio_file()
    return raw_audio.exists() and raw_audio.is_file() and raw_audio.stat().st_size > 0


def _cookie_domain(cookie: dict[str, Any]) -> str:
    domain = str(cookie.get("domain") or "")
    if not domain:
        return ".youtube.com"
    if cookie.get("httpOnly") and not domain.startswith("#HttpOnly_"):
        return "#HttpOnly_" + domain
    return domain


def _write_cookie_file(
    cookies: list[dict[str, Any]],
    cookie_file: Path | None = None,
    *,
    update_config: bool = True,
) -> Path | None:
    if not cookies:
        return None

    target = cookie_file or COOKIE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Netscape HTTP Cookie File",
        "# Generated by VideoLingo browser_bridge.py from Chrome extension cookies.",
    ]
    seen: set[tuple[str, str, str]] = set()
    for cookie in cookies:
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        if not name:
            continue
        domain = _cookie_domain(cookie)
        plain_domain = domain.removeprefix("#HttpOnly_")
        path = str(cookie.get("path") or "/")
        key = (plain_domain, path, name)
        if key in seen:
            continue
        seen.add(key)
        include_subdomains = "FALSE" if cookie.get("hostOnly") else "TRUE"
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = cookie.get("expirationDate")
        try:
            expires_text = str(int(float(expires))) if expires else "0"
        except (TypeError, ValueError):
            expires_text = "0"
        lines.append(
            "\t".join(
                [
                    domain,
                    include_subdomains,
                    path,
                    secure,
                    expires_text,
                    name,
                    value,
                ]
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if update_config:
        _update_config_cookie_path(target)
    return target


def _store_job_cookies(job_id: str, cookies: list[dict[str, Any]]) -> Path | None:
    cookie_path = _write_cookie_file(cookies, _job_cookie_file(job_id), update_config=False)
    if cookie_path is None:
        return None
    return cookie_path


def _update_config_cookie_path(cookie_file: Path) -> None:
    """Update config.yaml without importing VideoLingo's heavy core package."""
    try:
        _ensure_config_file()
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        data = yaml.load(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        youtube = data.setdefault("youtube", {})
        if isinstance(youtube, dict):
            youtube["cookies_path"] = str(cookie_file)
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                yaml.dump(data, f)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to update config.yaml youtube.cookies_path: {exc}", flush=True)


def _copy_outputs(job_id: str) -> dict[str, str]:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "audio": OUTPUT_DIR / "dub.mp3",
        "subtitle": OUTPUT_DIR / "dub.srt",
    }
    missing = [name for name, path in outputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected output(s): {', '.join(missing)}")

    copied: dict[str, str] = {}
    for name, src in outputs.items():
        dst = job_dir / src.name
        shutil.copy2(src, dst)
        copied[name] = str(dst.relative_to(PROJECT_ROOT)).replace("\\", "/")

    normalized_audio = job_dir / PLUGIN_AUDIO_FILE
    _normalize_plugin_audio(job_dir / "dub.mp3", normalized_audio)
    copied["plugin_audio"] = str(normalized_audio.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return copied


def _normalize_plugin_audio(src: Path, dst: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-filter:a",
        PLUGIN_AUDIO_LOUDNORM_FILTER,
        "-b:a",
        "96k",
        str(dst),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    if result.returncode != 0 or not dst.exists():
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Failed to normalize plugin audio with ffmpeg: {details}")


def _run_split_pipeline(args: list[str], log_file: Path) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["VIDEOLINGO_PARENT_PID"] = str(os.getpid())
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        *args,
    ]
    with log_file.open("ab") as log:
        log.write((" ".join(command) + "\n").encode("utf-8", errors="replace"))
        log.flush()
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _download_audio(url: str, cookie_path: Path | None, log_file: Path) -> None:
    command = ["download-audio", url]
    if cookie_path is not None:
        command.extend(["--cookies-path", str(cookie_path)])
    _run_split_pipeline(command, log_file)


def _run_pipeline(log_file: Path) -> None:
    _run_split_pipeline(["local-stop-before-video"], log_file)


def _run_job(job_id: str) -> None:
    with run_lock:
        with jobs_lock:
            job = jobs[job_id]
            url = job["url"]
            cookies = job.get("cookies") or []
            stored_cookie_path = _project_path(job.get("cookie_path"))

        log_file = LOGS_DIR / f"{job_id}.log"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _set_job(job_id, status="running", phase="preparing", log_path=str(log_file))

        try:
            _restore_job_workspace(job_id)
            if _raw_audio_ready():
                _set_job(job_id, phase="audio_ready")
                with log_file.open("ab") as log:
                    log.write(f"Reusing existing audio: {_raw_audio_file()}\n".encode("utf-8"))
                    log.flush()
            else:
                cookie_path = None
                if cookies:
                    cookie_path = _store_job_cookies(job_id, cookies)
                    if cookie_path is not None:
                        _set_job(job_id, cookie_path=_project_relative(cookie_path))
                elif stored_cookie_path and stored_cookie_path.exists():
                    cookie_path = stored_cookie_path
                if cookie_path is not None:
                    _update_config_cookie_path(cookie_path)

                _set_job(job_id, phase="downloading_audio")
                with log_file.open("ab") as log:
                    log.write(f"Downloading audio from: {url}\n".encode("utf-8"))
                    log.flush()
                _download_audio(url, cookie_path, log_file)
                _snapshot_output_to_job_workspace(job_id)

            _set_job(job_id, phase="transcribing_translating_dubbing")
            _run_pipeline(log_file)
            _snapshot_output_to_job_workspace(job_id)

            outputs = _copy_outputs(job_id)
            _set_job(job_id, status="done", phase="done", outputs=outputs, finished_at=_now())
        except Exception as exc:  # noqa: BLE001
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            try:
                _snapshot_output_to_job_workspace(job_id)
            except Exception:
                pass
            try:
                with log_file.open("ab") as log:
                    log.write(("\n[ERROR]\n" + traceback.format_exc()).encode("utf-8", errors="replace"))
            except Exception:
                pass
            _set_job(job_id, status="error", phase="error", error=error, finished_at=_now())


def _start_job_thread(job_id: str) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        thread = job.get("thread")
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(target=_run_job, args=(job_id,), name=f"browser-job-{job_id}", daemon=True)
        job["thread"] = thread
    thread.start()


def _start_queued_jobs() -> None:
    with jobs_lock:
        job_ids = [
            str(job["id"])
            for job in sorted(jobs.values(), key=lambda item: item.get("created_at", 0))
            if job.get("status") == "queued"
        ]
    for job_id in job_ids:
        _start_job_thread(job_id)


def _lookup_job(url: str) -> tuple[int, dict[str, Any]]:
    try:
        video_key, video_id, canonical_url = _youtube_video_from_url(url)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    with jobs_lock:
        job = _find_latest_video_job_unlocked(video_key)
        return 200, {
            "video_key": video_key,
            "video_id": video_id,
            "canonical_url": canonical_url,
            "job": _public_job(job) if job else None,
        }


def _start_job(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    url = str(payload.get("url") or "").strip()
    try:
        video_key, video_id, canonical_url = _youtube_video_from_url(url)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    cookies = payload.get("cookies") or []
    if not isinstance(cookies, list):
        return 400, {"error": "cookies must be a list"}

    with jobs_lock:
        existing = _find_latest_video_job_unlocked(video_key)
        if existing is not None:
            existing_status = existing.get("status")
            if existing_status in {"queued", "error"}:
                cookie_path = _store_job_cookies(str(existing["id"]), cookies)
                existing["cookies"] = cookies
                if cookie_path is not None:
                    existing["cookie_path"] = _project_relative(cookie_path)
                existing["title"] = str(payload.get("title") or existing.get("title") or "")
                existing["source_url"] = url
                existing["updated_at"] = _now()
                if existing_status == "error":
                    existing["status"] = "queued"
                    existing["phase"] = "queued"
                    existing.pop("error", None)
                _persist_index_unlocked()
            status = 200 if existing.get("status") == "done" else 202
            response = _public_job(existing)
            response["reused"] = True
            if existing.get("status") == "queued":
                _start_job_thread(str(existing["id"]))
            return status, response

    job_id = uuid.uuid4().hex[:12]
    cookie_path = _store_job_cookies(job_id, cookies)
    job = {
        "id": job_id,
        "url": canonical_url,
        "source_url": url,
        "video_key": video_key,
        "video_id": video_id,
        "title": str(payload.get("title") or ""),
        "status": "queued",
        "phase": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "cookies": cookies,
    }
    if cookie_path is not None:
        job["cookie_path"] = _project_relative(cookie_path)
    with jobs_lock:
        jobs[job_id] = job
        _persist_index_unlocked()
    _start_job_thread(job_id)
    return 202, _public_job(job)


def _job_file_path(job_id: str, filename: str) -> Path | None:
    allowed = {
        "dub.mp3": JOBS_DIR / job_id / "dub.mp3",
        PLUGIN_AUDIO_FILE: JOBS_DIR / job_id / PLUGIN_AUDIO_FILE,
        "dub.srt": JOBS_DIR / job_id / "dub.srt",
        "log.txt": LOGS_DIR / f"{job_id}.log",
    }
    path = allowed.get(filename)
    if path is None or not path.exists():
        return None
    return path


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "VideoLingoBrowserBridge/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        _write_cors_headers(self)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]

        if parsed.path == "/health":
            active = _active_job()
            _write_json(self, 200, {"ok": True, "active_job": _public_job(active) if active else None})
            return

        if parts == ["jobs"]:
            with jobs_lock:
                payload = [_public_job(job) for job in sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)]
            _write_json(self, 200, {"jobs": payload})
            return

        if parts == ["lookup"]:
            url = (parse_qs(parsed.query).get("url") or [""])[0]
            status, response = _lookup_job(url)
            _write_json(self, status, response)
            return

        if len(parts) == 2 and parts[0] == "jobs":
            job_id = parts[1]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                _write_json(self, 404, {"error": "job not found"})
                return
            _write_json(self, 200, _public_job(job))
            return

        if len(parts) == 3 and parts[0] == "jobs":
            job_id, filename = parts[1], parts[2]
            path = _job_file_path(job_id, filename)
            if path is None:
                _write_json(self, 404, {"error": "file not found"})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if filename.endswith(".srt") or filename.endswith(".txt"):
                content_type = "text/plain; charset=utf-8"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            _write_cors_headers(self)
            self.end_headers()
            self.wfile.write(data)
            return

        _write_json(self, 404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]

        try:
            payload = _read_json(self)
        except ValueError as exc:
            _write_json(self, 400, {"error": str(exc)})
            return

        if parts == ["jobs"]:
            status, response = _start_job(payload)
            _write_json(self, status, response)
            return

        _write_json(self, 404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local VideoLingo browser extension bridge.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _load_index()
    _start_queued_jobs()
    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"VideoLingo browser bridge listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping bridge...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
