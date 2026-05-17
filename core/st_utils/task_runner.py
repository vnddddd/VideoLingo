"""
Background task runner for Streamlit with pause/resume/stop control.

Usage:
    runner = TaskRunner.get(st.session_state)
    runner.start(steps)  # list of (label, callable) tuples
    runner.pause() / runner.resume() / runner.stop()
    runner.state  # "idle" | "running" | "paused" | "stopped" | "completed" | "error"
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable


_CURRENT_RUNNER = threading.local()


def get_current_runner() -> "TaskRunner | None":
    """Return the TaskRunner currently executing on this worker thread, if any."""
    return getattr(_CURRENT_RUNNER, "runner", None)


class StopTask(Exception):
    """Raised when the task is stopped by user."""

    pass


@dataclass
class TaskRunner:
    """Manages a background thread that executes a sequence of steps with pause/stop control."""

    # Public read-only state
    state: str = "idle"  # idle | running | paused | stopped | completed | error
    current_step: int = -1  # 0-indexed, -1 = not started
    total_steps: int = 0
    current_label: str = ""
    error_msg: str = ""
    logs: list[str] = field(default_factory=list)
    max_log_lines: int = 500

    # Internal
    _pause_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _steps: list = field(default_factory=list)
    _stop_callbacks: list[Callable[[], None]] = field(default_factory=list)
    _stop_callbacks_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self._pause_event.set()  # not paused initially

    # ------ Singleton per session_state ------
    @staticmethod
    def get(session_state, key: str = "_task_runner") -> "TaskRunner":
        """Get or create a TaskRunner stored in Streamlit session_state."""
        if key not in session_state:
            session_state[key] = TaskRunner()
        return session_state[key]

    # ------ Control API ------

    def start(self, steps: list[tuple[str, Callable]]):
        """Start executing steps in a background thread.

        Args:
            steps: list of (label, callable) — each callable takes no args.
        """
        if self.state == "running" or self.state == "paused":
            return  # already running

        self._steps = steps
        self.total_steps = len(steps)
        self.current_step = -1
        self.current_label = ""
        self.error_msg = ""
        self.logs = []
        self.state = "running"

        self._pause_event.set()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        if self.state == "running":
            self._pause_event.clear()
            self.state = "paused"

    def resume(self):
        if self.state == "paused":
            self._pause_event.set()
            self.state = "running"

    def stop(self):
        """Request stop and interrupt the currently running step if it registered cleanup."""
        if self.state in ("running", "paused"):
            self._stop_event.set()
            self._pause_event.set()  # unblock if paused so thread can exit
            self.state = "stopped"
            callbacks = self._snapshot_stop_callbacks()
            for callback in callbacks:
                try:
                    callback()
                except Exception as exc:
                    self.log(f"[WARN] Stop cleanup failed: {exc}")

    def register_stop_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register cleanup to run when stop() is requested; returns an unregister function."""
        with self._stop_callbacks_lock:
            self._stop_callbacks.append(callback)
        if self._stop_event.is_set():
            callback()

        def unregister() -> None:
            with self._stop_callbacks_lock:
                try:
                    self._stop_callbacks.remove(callback)
                except ValueError:
                    pass

        return unregister

    def _snapshot_stop_callbacks(self) -> list[Callable[[], None]]:
        with self._stop_callbacks_lock:
            return list(self._stop_callbacks)

    def reset(self):
        """Reset to idle state (only when not running)."""
        if self.state not in ("running", "paused"):
            self.state = "idle"
            self.current_step = -1
            self.total_steps = 0
            self.current_label = ""
            self.error_msg = ""
            self.logs = []
            self._steps = []

    def log(self, message: object):
        """Append one or more log lines for the Streamlit task panel."""
        text = "" if message is None else str(message)
        for line in text.splitlines() or [""]:
            self.logs.append(line)
        if len(self.logs) > self.max_log_lines:
            self.logs = self.logs[-self.max_log_lines:]

    @property
    def is_active(self) -> bool:
        return self.state in ("running", "paused")

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_done(self) -> bool:
        return self.state in ("completed", "stopped", "error")

    @property
    def progress(self) -> float:
        """0.0 to 1.0"""
        if self.total_steps == 0:
            return 0.0
        return min((self.current_step + 1) / self.total_steps, 1.0)

    # ------ Internal ------

    def _run(self):
        """Execute steps sequentially in background thread."""
        try:
            for i, (label, func) in enumerate(self._steps):
                # Check stop before each step
                if self._stop_event.is_set():
                    self.state = "stopped"
                    return

                # Block if paused
                self._pause_event.wait()

                # Check stop again after resume
                if self._stop_event.is_set():
                    self.state = "stopped"
                    return

                self.current_step = i
                self.current_label = label
                self.log(f"[STEP {i + 1}/{self.total_steps}] {label}")
                _CURRENT_RUNNER.runner = self
                try:
                    func()
                finally:
                    _CURRENT_RUNNER.runner = None

            self.log("[DONE] Task completed")
            self.state = "completed"
        except StopTask as e:
            self.error_msg = str(e)
            if self.error_msg:
                self.log(f"[STOPPED] {self.error_msg}")
            self.state = "stopped"
        except Exception as e:
            self.error_msg = str(e)
            self.log(f"[ERROR] {self.error_msg}")
            self.state = "error"
