"""Best-effort, local-only diagnostics for sentence segmentation.

One daemon per run writes bounded logs and watches for a lack of progress.
The pipeline never waits for disk writes or writes diagnostics to the console.
Stacks contain code locations only, never source text, arguments or locals.
"""

from __future__ import annotations

import functools
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from uuid import uuid4


LOG_DIR = Path("output/log")
EVENT_LOG = "split_diagnostics.jsonl"
STACK_LOG = "split_stalls.log"
STALL_SECONDS = 60.0
SNAPSHOT_INTERVAL_SECONDS = 120.0
MAX_SNAPSHOTS = 3
MAX_LOG_BYTES = 2 * 1024 * 1024  # Each log has at most one rotated backup.

_CURRENT = threading.local()
_FILE_LOCK = threading.Lock()  # Used only by diagnostic writers, never work threads.


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class _QuietHandler(RotatingFileHandler):
    def handleError(self, record):
        # logging's default error handler prints to stderr, which may be blocked.
        pass


class DiagnosticRun:
    def __init__(self, phase, *, stall_seconds=None):
        self.phase = phase
        self.run_id = uuid4().hex[:12]
        self.log_dir = LOG_DIR.resolve()
        self.stall_seconds = STALL_SECONDS if stall_seconds is None else stall_seconds
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.last_progress = self.started
        self.owner = threading.get_ident()
        self.active = {}
        self.sequence = 0
        self.dropped_events = 0
        self.snapshots = 0
        self.last_snapshot = float("-inf")
        self.lock = threading.Lock()
        self.events = queue.Queue(maxsize=256)
        self.stop = threading.Event()
        self.thread = None
        self.enabled = False

    def _event(self, event, **fields):
        if not self.enabled:
            return
        try:
            self.events.put_nowait({
                "time": _timestamp(), "pid": os.getpid(), "run_id": self.run_id,
                "phase": self.phase, "event": event, **fields,
            })
        except queue.Full:
            self.dropped_events += 1
        except Exception:
            pass

    def __enter__(self):
        self.previous = getattr(_CURRENT, "run", None)
        _CURRENT.run = self
        try:
            self.enabled = True
            self._event("run_started", thread_id=self.owner, stall_seconds=self.stall_seconds)
            self.thread = threading.Thread(
                target=self._watch, name=f"split-diagnostics-{self.phase}", daemon=True,
            )
            self.thread.start()
        except Exception:
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._event(
                "run_finished", status="error" if exc_type else "completed",
                error_type=exc_type.__name__ if exc_type else None,
                seconds=round(time.monotonic() - self.started, 6),
                process_cpu_seconds=round(time.process_time() - self.cpu_started, 6),
                dropped_events=self.dropped_events,
            )
            self.stop.set()
            if self.thread is not None and self.thread.is_alive():
                # A full or unavailable disk must not hold up the real task.
                self.thread.join(timeout=0.2)
        except Exception:
            pass
        finally:
            _CURRENT.run = self.previous
        return False

    def begin_step(self, name, details):
        try:
            started = time.monotonic()
            with self.lock:
                self.sequence += 1
                token = self.sequence
                self.active[token] = {
                    "step": name, "thread_id": threading.get_ident(),
                    "started": started, "details": dict(details),
                }
                self.last_progress = started
            self._event("step_started", step_id=token, step=name,
                        thread_id=threading.get_ident(), **details)
            return token
        except Exception:
            return None

    def end_step(self, token, cpu_started, error_type):
        try:
            now = time.monotonic()
            with self.lock:
                entry = self.active.pop(token, None)
                self.last_progress = now
            if entry is not None:
                self._event(
                    "step_finished", step_id=token, step=entry["step"],
                    thread_id=entry["thread_id"], status="error" if error_type else "completed",
                    error_type=error_type, seconds=round(now - entry["started"], 6),
                    thread_cpu_seconds=round(time.thread_time() - cpu_started, 6),
                    **entry["details"],
                )
        except Exception:
            pass

    def progress(self, token, details):
        try:
            with self.lock:
                if token in self.active:
                    self.active[token]["details"].update(details)
                self.last_progress = time.monotonic()
        except Exception:
            pass

    def _handler(self, filename):
        try:
            with _FILE_LOCK:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                handler = _QuietHandler(
                    self.log_dir / filename, maxBytes=MAX_LOG_BYTES,
                    backupCount=1, encoding="utf-8",
                )
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler
        except Exception:
            return None

    @staticmethod
    def _write(handler, message):
        if handler is None:
            return
        try:
            record = logging.LogRecord("split_diagnostics", logging.INFO, "", 0, message, (), None)
            with _FILE_LOCK:
                handler.handle(record)
        except Exception:
            pass

    def _stall_snapshot(self):
        now = time.monotonic()
        with self.lock:
            idle = now - self.last_progress
            if (idle < self.stall_seconds or self.snapshots >= MAX_SNAPSHOTS
                    or now - self.last_snapshot < SNAPSHOT_INTERVAL_SECONDS):
                return None
            self.snapshots += 1
            self.last_snapshot = now
            active = [
                {"step_id": token, "step": entry["step"], "thread_id": entry["thread_id"],
                 "seconds": round(now - entry["started"], 3), **entry["details"]}
                for token, entry in self.active.items()
            ]
        header = {
            "time": _timestamp(), "pid": os.getpid(), "run_id": self.run_id,
            "phase": self.phase, "event": "no_progress_snapshot",
            "no_progress_seconds": round(idle, 3), "owner_thread_id": self.owner,
            "process_cpu_seconds": round(time.process_time() - self.cpu_started, 3),
            "snapshot": self.snapshots, "active_steps": active,
        }
        lines = ["=== " + json.dumps(header, ensure_ascii=False) + " ==="]
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        frames = sys._current_frames()
        try:
            for ident, frame in frames.items():
                lines.append(f"Thread {ident} ({names.get(ident, 'unknown')}):")
                locations = []
                for _ in range(64):
                    if frame is None:
                        break
                    code = frame.f_code
                    locations.append(f"  {code.co_filename}:{frame.f_lineno} in {code.co_name}")
                    frame = frame.f_back
                lines.extend(reversed(locations))
        finally:
            # Do not keep live frames/locals alive after taking the snapshot.
            frames.clear()
        return header, "\n".join(lines) + "\n"

    def _watch(self):
        events_handler = stacks_handler = None
        try:
            events_handler = self._handler(EVENT_LOG)
            poll = min(1.0, max(0.01, self.stall_seconds / 4))
            while not self.stop.is_set() or not self.events.empty():
                try:
                    event = self.events.get(timeout=poll)
                except queue.Empty:
                    event = None
                if event is not None:
                    self._write(events_handler, json.dumps(event, ensure_ascii=False))
                    if event["event"] == "run_finished":
                        break
                if not self.stop.is_set():
                    snapshot = self._stall_snapshot()
                    if snapshot is not None:
                        if stacks_handler is None:
                            stacks_handler = self._handler(STACK_LOG)
                        header, stack = snapshot
                        self._write(events_handler, json.dumps(header, ensure_ascii=False))
                        self._write(stacks_handler, stack)
        except Exception:
            # Diagnostic failure must never escape into the pipeline.
            pass
        finally:
            for handler in (events_handler, stacks_handler):
                if handler is not None:
                    try:
                        handler.close()
                    except Exception:
                        pass


def diagnose_stage(phase):
    """Monitor the whole call, including an existing cache-check decorator."""
    def decorate(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                run = DiagnosticRun(phase)
            except Exception:
                return func(*args, **kwargs)
            with run:
                return func(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def diagnostic_step(name, **details):
    run = getattr(_CURRENT, "run", None)
    if run is None or not run.enabled:
        yield
        return
    previous = getattr(_CURRENT, "step", None)
    token = run.begin_step(name, details)
    _CURRENT.step = token
    cpu_started = time.thread_time()
    error_type = None
    try:
        yield
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        run.end_step(token, cpu_started, error_type)
        _CURRENT.step = previous


def diagnostic_progress(**details):
    """Cheap in-memory heartbeat; does not write a log line for each sentence."""
    run = getattr(_CURRENT, "run", None)
    if run is not None and run.enabled:
        run.progress(getattr(_CURRENT, "step", None), details)


def inherit_diagnostics(func):
    """Carry only diagnostic context into an existing executor's work item."""
    run = getattr(_CURRENT, "run", None)
    if run is None:
        return func

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        previous = getattr(_CURRENT, "run", None)
        _CURRENT.run = run
        try:
            return func(*args, **kwargs)
        finally:
            _CURRENT.run = previous
    return wrapped
