"""Run with .venv/Scripts/python tools/_test_split_diagnostics.py.

All logs go to temporary folders; no real LLM calls or model downloads.
"""

import importlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.utils import step_diagnostics as diag


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="videolingo-split-diagnostics-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "logs"
        patcher = patch.object(diag, "LOG_DIR", self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def events(self):
        path = self.directory / diag.EVENT_LOG
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def finish(self, run):
        run.thread.join(timeout=2)
        self.assertFalse(run.thread.is_alive(), "Diagnostic worker did not exit")

    def test_normal_steps_record_duration_and_exit(self):
        result = object()
        with diag.DiagnosticRun("nlp") as run:
            with diag.diagnostic_step("nlp.model_init"):
                actual = result
            with diag.diagnostic_step("nlp.split_by_mark", sentence_count=239):
                diag.diagnostic_progress(sentence_index=17)
        self.finish(run)
        self.assertIs(actual, result)
        events = self.events()
        self.assertEqual(events[0]["event"], "run_started")
        self.assertEqual(events[-1]["event"], "run_finished")
        self.assertEqual(events[-1]["status"], "completed")
        self.assertEqual(len({item["run_id"] for item in events}), 1)
        steps = [item for item in events if item["event"] == "step_finished"]
        self.assertEqual([item["step"] for item in steps], ["nlp.model_init", "nlp.split_by_mark"])
        self.assertEqual(steps[-1]["sentence_index"], 17)
        self.assertTrue(all(item["seconds"] >= 0 and item["thread_cpu_seconds"] >= 0 for item in steps))
        self.assertFalse((self.directory / diag.STACK_LOG).exists())
        self.assertIsNone(getattr(diag._CURRENT, "run", None))

    def test_original_exception_propagates_without_logging_its_values(self):
        original = ValueError("private-transcript-and-token")
        run = diag.DiagnosticRun("nlp")
        with self.assertRaises(ValueError) as caught:
            with run:
                with diag.diagnostic_step("nlp.split_by_root"):
                    raise original
        self.finish(run)
        self.assertIs(caught.exception, original)
        events = self.events()
        self.assertEqual(events[-1]["error_type"], "ValueError")
        self.assertEqual(events[-2]["status"], "error")
        self.assertNotIn(str(original), (self.directory / diag.EVENT_LOG).read_text(encoding="utf-8"))

    def test_disk_failure_does_not_break_work(self):
        called = []
        with patch.object(diag.DiagnosticRun, "_handler", return_value=None):
            with diag.DiagnosticRun("nlp") as run:
                with diag.diagnostic_step("nlp.split_by_mark"):
                    called.append(True)
            self.finish(run)
        self.assertEqual(called, [True])

    def test_thread_start_failure_does_not_break_work(self):
        @diag.diagnose_stage("nlp")
        def work(value):
            with diag.diagnostic_step("nlp.split_by_mark"):
                return value

        with patch.object(threading.Thread, "start", side_effect=RuntimeError("thread unavailable")):
            self.assertEqual(work(42), 42)
        self.assertIsNone(getattr(diag._CURRENT, "run", None))

    def test_slow_writer_and_full_queue_do_not_block_work(self):
        entered = threading.Event()
        release = threading.Event()

        def blocked_write(*args):
            entered.set()
            release.wait(3)

        run = diag.DiagnosticRun("nlp")
        try:
            with patch.object(diag.DiagnosticRun, "_write", side_effect=blocked_write):
                started = time.monotonic()
                with run:
                    self.assertTrue(entered.wait(2))
                    for index in range(600):
                        with diag.diagnostic_step("nlp.test", sentence_index=index):
                            pass
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertGreater(run.dropped_events, 0)
        finally:
            release.set()
            self.finish(run)

    def test_stall_captures_blocked_frame_without_locals_or_source(self):
        captured = threading.Event()
        write = diag.DiagnosticRun._write

        def observe_write(handler, message):
            write(handler, message)
            if message.startswith("=== "):
                captured.set()

        def blocked_work():
            private_token = "sensitive-key-not-for-diagnostics"
            private_subtitle = "a-private-sentence-not-for-diagnostics"
            with diag.diagnostic_step("nlp.split_by_connector", sentence_index=17):
                self.assertTrue(captured.wait(3), "No stall snapshot was written")
            return bool(private_token and private_subtitle)

        run = diag.DiagnosticRun("nlp", stall_seconds=0.04)
        with patch.object(diag.DiagnosticRun, "_write", side_effect=observe_write):
            with run:
                self.assertTrue(blocked_work())
            self.finish(run)
        text = (self.directory / diag.STACK_LOG).read_text(encoding="utf-8")
        self.assertIn("in blocked_work", text)
        self.assertIn("nlp.split_by_connector", text)
        self.assertIn('"sentence_index": 17', text)
        self.assertNotIn("sensitive-key-not-for-diagnostics", text)
        self.assertNotIn("a-private-sentence-not-for-diagnostics", text)
        self.assertEqual(run.snapshots, 1)

    def test_progress_resets_stall_deadline_and_snapshots_are_capped(self):
        now = [0.0]
        clock = SimpleNamespace(monotonic=lambda: now[0], process_time=lambda: 0.0)
        with patch.object(diag, "time", clock):
            run = diag.DiagnosticRun("nlp")
            token = run.begin_step("nlp.split_by_comma", {})
            now[0] = 59
            run.progress(token, {"sentence_index": 12})
            now[0] = 100
            self.assertIsNone(run._stall_snapshot())
            now[0] = 120
            snapshot = run._stall_snapshot()
            self.assertEqual(snapshot[0]["active_steps"][0]["sentence_index"], 12)
            now[0] = 121
            self.assertIsNone(run._stall_snapshot())
            now[0] = 240
            self.assertIsNotNone(run._stall_snapshot())
            now[0] = 360
            self.assertIsNotNone(run._stall_snapshot())
            now[0] = 10000
            self.assertIsNone(run._stall_snapshot())
            self.assertEqual(run.snapshots, diag.MAX_SNAPSHOTS)

    def test_cached_call_is_still_monitored_without_rerunning_work(self):
        from core.utils import decorator
        marker = self.directory.parent / "cached.txt"
        marker.touch()
        calls = []

        @diag.diagnose_stage("nlp")
        @decorator.check_file_exists(str(marker))
        def work():
            calls.append(True)

        with patch.object(decorator, "rprint"):
            work()
        self.assertEqual(calls, [])
        self.assertEqual(self.events()[-1]["status"], "completed")

    def test_rotation_retains_only_one_backup(self):
        with patch.object(diag, "MAX_LOG_BYTES", 2048):
            with diag.DiagnosticRun("nlp") as run:
                for index in range(60):
                    with diag.diagnostic_step("nlp.test", sentence_index=index):
                        pass
            self.finish(run)
        self.assertTrue((self.directory / (diag.EVENT_LOG + ".1")).exists())
        self.assertFalse((self.directory / (diag.EVENT_LOG + ".2")).exists())
        self.assertTrue(all(path.stat().st_size <= 2048 for path in self.directory.iterdir()))

    def test_nlp_call_order_and_arguments_are_unchanged(self):
        from core import _3_1_split_nlp as nlp
        model = object()
        order = []
        patchers = []
        for name in ("split_by_mark", "split_by_comma_main", "split_sentences_main", "split_long_by_root_main"):
            patcher = patch.object(nlp, name, side_effect=lambda received, name=name: (
                self.assertIs(received, model), order.append(name)))
            patcher.start()
            patchers.append(patcher)
        try:
            with patch.object(nlp, "init_nlp", return_value=model):
                with diag.DiagnosticRun("nlp") as run:
                    # Bypass only the already-tested cache and outer run decorators.
                    nlp.split_by_spacy.__wrapped__.__wrapped__()
                self.finish(run)
        finally:
            for patcher in patchers:
                patcher.stop()
        self.assertEqual(order, ["split_by_mark", "split_by_comma_main", "split_sentences_main", "split_long_by_root_main"])
        self.assertEqual(len([event for event in self.events() if event["event"] == "step_finished"]), 5)

    def test_model_download_fallback_is_not_changed(self):
        loader = importlib.import_module("core.spacy_utils.load_nlp_model")
        model = object()
        with (
            patch.object(loader, "load_key", return_value="en"),
            patch.object(loader, "rprint"),
            patch.object(loader.spacy, "load", side_effect=[OSError("missing model"), model]) as load,
            patch.object(loader, "download") as download,
        ):
            with diag.DiagnosticRun("nlp") as run:
                self.assertIs(loader.init_nlp(), model)
            self.finish(run)
        self.assertEqual(load.call_count, 2)
        download.assert_called_once_with("en_core_web_md")
        steps = [event for event in self.events() if event["event"] == "step_finished"]
        self.assertEqual([event["step"] for event in steps], [
            "nlp.spacy_load", "nlp.model_download", "nlp.spacy_load_after_download",
        ])
        self.assertEqual(steps[0]["status"], "error")

    def test_llm_calls_output_order_and_worker_context_are_unchanged(self):
        from core import _3_2_split_meaning as meaning
        sentences = ["alpha beta gamma delta epsilon zeta", "short sentence", "one two three four five six"]
        calls = []

        def fake_request(prompt, **kwargs):
            calls.append((prompt, kwargs["resp_type"], kwargs["log_title"]))
            words = prompt.strip().split()
            return {"choice": 1, "split1": " ".join(words[:3]) + " [br] " + " ".join(words[3:])}

        with (
            patch.object(meaning, "tokenize_sentence", side_effect=lambda sentence, nlp: sentence.split()),
            patch.object(meaning, "get_split_prompt", side_effect=lambda sentence, *args: sentence),
            patch.object(meaning, "ask_gpt", side_effect=fake_request),
            patch.object(meaning, "load_key", return_value="en"),
            patch.object(meaning, "get_joiner", return_value=" "),
            patch.object(meaning.console, "print"),
        ):
            expected = meaning.parallel_split_sentences(sentences, 4, 2, None)
            expected_calls = sorted(calls)
            calls.clear()
            with diag.DiagnosticRun("llm") as run:
                actual = meaning.parallel_split_sentences(sentences, 4, 2, None)
            self.finish(run)
        self.assertEqual(actual, expected)
        self.assertEqual(sorted(calls), expected_calls)
        requests = [event for event in self.events() if event.get("step") == "llm.request" and event["event"] == "step_finished"]
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(event["thread_id"] != threading.get_ident() for event in requests))
        self.assertEqual({event["sentence_index"] for event in requests}, {0, 2})
        self.assertNotIn(sentences[0], (self.directory / diag.EVENT_LOG).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
