"""Smoke tests for split_pipeline child-process cleanup hooks.

Run from project root:  python tools/_test_proc_cleanup.py
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
import textwrap
import tempfile
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.st_utils.task_runner import StopTask, TaskRunner  # noqa: E402


def is_alive(pid: int) -> bool:
    if os.name == "nt":
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        return str(pid) in (r.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def hard_kill(pid: int) -> None:
    if not is_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def make_busy_script(tmp: pathlib.Path) -> pathlib.Path:
    p = tmp / "busy.py"
    p.write_text(textwrap.dedent("""
        import time
        while True:
            time.sleep(0.5)
    """).strip(), encoding="utf-8")
    return p


def run_stop_simulation() -> None:
    """TaskRunner.stop() should run the registered callback that kills the child."""
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        busy = make_busy_script(tmp)

        from core.st_utils.task_runner import _CURRENT_RUNNER

        runner = TaskRunner()
        proc_holder: dict = {}

        def step():
            popen_kwargs = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen([sys.executable, str(busy)], **popen_kwargs)
            proc_holder["proc"] = proc
            cb = runner.register_stop_callback(lambda: hard_kill(proc.pid))
            try:
                while True:
                    if runner.stop_requested:
                        hard_kill(proc.pid)
                        raise StopTask("stopped")
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
            finally:
                cb()
            if runner.stop_requested:
                raise StopTask("stopped")

        runner.start([("busy", step)])
        # Wait for child to spawn
        for _ in range(50):
            if "proc" in proc_holder:
                break
            time.sleep(0.05)
        assert "proc" in proc_holder, "child not spawned"
        child_pid = proc_holder["proc"].pid
        assert is_alive(child_pid), f"child {child_pid} should be alive"

        runner.stop()
        # Wait for runner thread to exit
        if runner._thread is not None:
            runner._thread.join(timeout=10)
        assert not is_alive(child_pid), f"child {child_pid} should be dead"
        assert runner.state == "stopped", f"state={runner.state}"
        print(f"[OK] Stop simulation: child PID {child_pid} killed, state={runner.state}")


def make_watchdog_child(tmp: pathlib.Path, env_overrides: dict) -> subprocess.Popen:
    """Spawn a fake CLI that uses split_pipeline's parent watchdog logic."""
    # Minimal child: imports the watchdog helpers and idles.
    child_script = tmp / "child.py"
    child_script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, r"{ROOT}")
        from tools.split_pipeline import _start_parent_watchdog
        _start_parent_watchdog()
        while True:
            time.sleep(0.5)
    """).strip(), encoding="utf-8")

    env = os.environ.copy()
    env.update(env_overrides)
    popen_kwargs = {"env": env}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen([sys.executable, str(child_script)], **popen_kwargs)


def run_watchdog_parent_dies_simulation() -> None:
    """When the parent process dies, watchdog should kill the child within a few seconds."""
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        parent_busy = tmp / "parent.py"
        parent_busy.write_text(textwrap.dedent("""
            import time
            while True:
                time.sleep(0.5)
        """).strip(), encoding="utf-8")
        parent = subprocess.Popen([sys.executable, str(parent_busy)])
        try:
            child = make_watchdog_child(tmp, {"VIDEOLINGO_PARENT_PID": str(parent.pid)})
            time.sleep(2.0)
            assert is_alive(child.pid), "child should be alive while parent alive"
            hard_kill(parent.pid)
            deadline = time.time() + 15
            while time.time() < deadline and is_alive(child.pid):
                time.sleep(0.5)
            if is_alive(child.pid):
                hard_kill(child.pid)
                raise AssertionError(f"watchdog child {child.pid} survived parent death")
            print(f"[OK] Watchdog parent-dies: child PID {child.pid} exited after parent died")
        finally:
            hard_kill(parent.pid)


def run_watchdog_identity_mismatch_simulation() -> None:
    """Pass a bogus identity; child should self-terminate quickly even though parent PID is alive."""
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        parent_busy = tmp / "parent.py"
        parent_busy.write_text(textwrap.dedent("""
            import time
            while True:
                time.sleep(0.5)
        """).strip(), encoding="utf-8")
        parent = subprocess.Popen([sys.executable, str(parent_busy)])
        try:
            env = {
                "VIDEOLINGO_PARENT_PID": str(parent.pid),
                "VIDEOLINGO_PARENT_IDENTITY": "BOGUS_IDENTITY_TOKEN_99999",
            }
            child = make_watchdog_child(tmp, env)
            deadline = time.time() + 15
            while time.time() < deadline and is_alive(child.pid):
                time.sleep(0.5)
            if is_alive(child.pid):
                hard_kill(child.pid)
                raise AssertionError(f"watchdog child {child.pid} ignored identity mismatch")
            print(f"[OK] Watchdog identity-mismatch: child PID {child.pid} exited even though parent PID alive")
        finally:
            hard_kill(parent.pid)


if __name__ == "__main__":
    run_stop_simulation()
    run_watchdog_parent_dies_simulation()
    run_watchdog_identity_mismatch_simulation()
    print("ALL OK")
