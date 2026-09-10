"""Bounded, offline UI-refresh/NLP comparison using a transcript copy.

Run: .venv/Scripts/python -X utf8 tools/_benchmark_ui_nlp.py
Each mode runs in its own process with a 55-second timeout. Only temporary
files are written. The existing en_core_web_md model is used, never downloaded.
No running Web UI is connected to, restarted, or changed by this benchmark.
"""

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PANEL = '''
import streamlit as st
from types import SimpleNamespace
from core.st_utils.timing_panel import render_timing_panel

@st.fragment(run_every=1)
def panel():
    data = {
        "media": {"duration_seconds": 1448.39},
        "stages": {"nlp": {"label": "NLP", "seconds": 20, "status": "running", "runs": 1}},
    }
    render_timing_panel(SimpleNamespace(timing_snapshot=lambda: data), "benchmark")

panel()
'''


def run_worker(mode, workspace):
    os.chdir(workspace)
    import spacy
    from streamlit import config as st_config
    from streamlit.runtime.scriptrunner.script_runner import ScriptRunner
    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1.local_script_runner import LocalScriptRunner
    from core import utils
    from core.utils import config_utils as config
    from translations import translations

    if mode == "before":
        # Exact pre-cache read behavior, using only the temporary example config.
        def uncached_key(key):
            with config.lock:
                config.ensure_config_file()
                with open(config.CONFIG_PATH, encoding="utf-8") as file:
                    value = config.yaml.load(file)
            for part in key.split("."):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    raise KeyError(f"Key '{part}' not found in configuration")
            return value

        def uncached_translations(language):
            with open(f"translations/{language}.json", encoding="utf-8") as file:
                return json.load(file)

        config.load_key = utils.load_key = uncached_key
        translations._get_translations = uncached_translations

    from core import _3_1_split_nlp as pipeline
    nlp = spacy.load("en_core_web_md")
    # Model initialization is excluded in both modes; the four real NLP passes
    # and their diagnostic logging are identical to the application.
    pipeline.init_nlp = lambda: nlp
    app = AppTest.from_string(PANEL)
    st_config.set_option("runner.postScriptGC", False)
    app.run(timeout=15)  # Warm UI imports/caches in both processes.
    if app.exception:
        raise AssertionError("Panel warmup failed")
    gc.collect(2)
    st_config.set_option("runner.postScriptGC", mode == "before")

    counters = {"yaml_parses": 0, "json_parses": 0, "forced_gc_calls": 0, "forced_gc_seconds": 0.0}
    original_yaml_load = config.yaml.load
    original_json_load = json.load
    original_collect = gc.collect

    def yaml_load(*args, **kwargs):
        counters["yaml_parses"] += 1
        return original_yaml_load(*args, **kwargs)

    def json_load(*args, **kwargs):
        counters["json_parses"] += 1
        return original_json_load(*args, **kwargs)

    def collect(generation=2):
        started = time.perf_counter()
        result = original_collect(generation)
        if generation == 2:
            counters["forced_gc_calls"] += 1
            counters["forced_gc_seconds"] += time.perf_counter() - started
        return result

    finished = threading.Event()
    stats = {}
    errors = []

    def work():
        started, cpu = time.perf_counter(), time.thread_time()
        try:
            pipeline.split_by_spacy()
        except BaseException as exc:
            errors.append(exc)
        finally:
            stats["nlp_seconds"] = time.perf_counter() - started
            stats["nlp_thread_cpu_seconds"] = time.thread_time() - cpu
            finished.set()

    refresh_seconds = []
    # AppTest normally overrides _on_script_finished and omits GC! Exercise
    # the real Streamlit finish hook so this reproduces the actual contention.
    with (
        patch.object(LocalScriptRunner, "_on_script_finished", ScriptRunner._on_script_finished),
        patch.object(config.yaml, "load", yaml_load),
        patch.object(json, "load", json_load),
        patch.object(gc, "collect", collect),
        redirect_stdout(io.StringIO()),
    ):
        worker = threading.Thread(target=work, name="benchmark-nlp", daemon=True)
        started, cpu = time.perf_counter(), time.process_time()
        worker.start()
        while not finished.is_set():
            refreshed = time.perf_counter()
            app.run(timeout=30)
            if app.exception:
                raise AssertionError("Concurrent panel refresh failed")
            refresh_seconds.append(time.perf_counter() - refreshed)
            finished.wait(max(0, 1 - (time.perf_counter() - refreshed)))
        worker.join(timeout=2)
        stats["total_seconds"] = time.perf_counter() - started
        stats["process_cpu_seconds"] = time.process_time() - cpu
    if errors:
        raise errors[0]
    output = (workspace / "output/log/split_by_nlp.txt").read_bytes()
    stats.update(counters)
    stats.update({
        "mode": mode,
        "ui_refreshes": len(refresh_seconds),
        "max_refresh_seconds": max(refresh_seconds, default=0),
        "automatic_gc_enabled": gc.isenabled(),
        "output_lines": len(output.splitlines()),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    })
    assert stats["ui_refreshes"] >= 2, "Workload did not overlap enough refreshes"
    if mode == "after":
        assert counters["forced_gc_calls"] == 0
        assert counters["yaml_parses"] == counters["json_parses"] == 0
    else:
        assert counters["forced_gc_calls"] == stats["ui_refreshes"]
    assert gc.isenabled()
    print(json.dumps(stats), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/log/cleaned_chunks.xlsx")
    parser.add_argument("--worker-mode", choices=("before", "after"), help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_mode:
        with patch("socket.socket.connect", side_effect=AssertionError("Network disabled in benchmark")):
            run_worker(args.worker_mode, args.workspace.resolve())
        return

    results = []
    with tempfile.TemporaryDirectory(prefix="videolingo-ui-nlp-") as temporary:
        base = Path(temporary)
        snapshot = base / "input.xlsx"
        shutil.copy2(args.source, snapshot)
        for mode in ("before", "after"):
            workspace = base / mode
            (workspace / "output/log").mkdir(parents=True)
            (workspace / "translations").mkdir()
            shutil.copy2(snapshot, workspace / "output/log/cleaned_chunks.xlsx")
            shutil.copy2(ROOT / "config.example.yaml", workspace / "config.yaml")
            for language in ("en", "zh-CN"):
                shutil.copy2(ROOT / f"translations/{language}.json", workspace / f"translations/{language}.json")
            print(f"Running {mode}: real NLP + one panel refresh per second (55s limit)...", flush=True)
            try:
                run = subprocess.run(
                    [sys.executable, "-X", "utf8", __file__, "--worker-mode", mode, "--workspace", str(workspace)],
                    cwd=ROOT, capture_output=True, encoding="utf-8", timeout=55, check=True,
                )
            except subprocess.TimeoutExpired:
                print(json.dumps({"mode": mode, "timed_out_seconds": 55}), flush=True)
                continue
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"{mode} benchmark failed:\n{exc.stderr}") from exc
            result = json.loads(run.stdout)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    after = next((item for item in results if item["mode"] == "after"), None)
    if after is None:
        raise AssertionError("Patched concurrent run did not finish within the time limit")
    if len(results) == 2:
        assert results[0]["output_sha256"] == after["output_sha256"], "NLP output changed"
        print("NLP output is byte-identical across both concurrent runs.", flush=True)


if __name__ == "__main__":
    main()
