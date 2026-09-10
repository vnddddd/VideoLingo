"""Run with .venv/Scripts/python tools/_test_stage_timings.py.

Uses temporary logs and dummy work; no transcription, translation, or TTS APIs.
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.st_utils import task_runner
from core.st_utils.task_runner import StopTask, TaskRunner
from core.utils import stage_timer


class StageTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="videolingo-timings-")
        self.addCleanup(self.temp.cleanup)
        self.log_path = Path(self.temp.name) / "log" / "stage_timings.json"
        self.now = 0.0
        for target, name, value in (
            (stage_timer, "TIMINGS_FILE", self.log_path),
            (stage_timer, "_source_duration", lambda: 120.0),
            (task_runner, "time", SimpleNamespace(monotonic=lambda: self.now)),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.runner = TaskRunner()

    def advance(self, seconds):
        self.now += seconds

    def finish(self):
        self.runner._thread.join(timeout=5)
        self.assertFalse(self.runner._thread.is_alive(), "Task did not finish")

    def test_success_survives_reset_and_new_session(self):
        self.runner.start([
            ("asr", "Transcription", lambda: self.advance(12.5)),
            ("translate", "Translation", lambda: self.advance(30)),
        ])
        self.finish()
        self.assertEqual(self.runner.state, "completed")
        self.assertTrue(any("[TIME]" in line for line in self.runner.logs))
        self.runner.reset()
        data = TaskRunner().timing_snapshot()
        self.assertEqual(list(data["stages"]), ["asr", "translate"])
        self.assertEqual(data["stages"]["asr"]["seconds"], 12.5)
        self.assertEqual(data["stages"]["translate"]["seconds"], 30)
        self.assertEqual(data["stages"]["asr"]["status"], "completed")
        self.assertEqual(data["media"]["duration_seconds"], 120)
        self.assertFalse(self.log_path.with_suffix(".json.tmp").exists())

    def test_live_retry_does_not_double_count(self):
        stage_timer.record_stage("tts", "TTS", 12, status="error")
        entered, release = threading.Event(), threading.Event()

        def work():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test step was not released")

        self.runner.start([("tts", "TTS", work)])
        try:
            self.assertTrue(entered.wait(5))
            self.advance(5)
            for _ in range(2):
                live = self.runner.timing_snapshot()["stages"]["tts"]
                self.assertEqual(live["seconds"], 17)
                self.assertEqual(live["runs"], 2)
                self.assertEqual(live["status"], "running")
            self.assertEqual(stage_timer.load_timings()["stages"]["tts"]["seconds"], 12)
        finally:
            release.set()
            self.finish()
        saved = stage_timer.load_timings()["stages"]["tts"]
        self.assertEqual(saved["seconds"], 17)
        self.assertEqual(saved["runs"], 2)
        self.assertEqual(saved["status"], "completed")

    def test_failure_and_stop_save_elapsed_and_skip_later_steps(self):
        for error, expected in ((ValueError("test failure"), "error"), (StopTask("pick voices"), "stopped")):
            with self.subTest(status=expected):
                self.runner = TaskRunner()
                later = []

                def work():
                    self.advance(3.25)
                    raise error

                self.runner.start([
                    (expected, expected, work),
                    ("later", "Later", lambda: later.append(True)),
                ])
                self.finish()
                self.assertEqual(self.runner.state, expected)
                if expected == "error":
                    self.assertIn("ValueError: test failure", self.runner.error_details)
                    error_log = self.log_path.with_name("task_errors.log").read_text(encoding="utf-8")
                    self.assertIn("ValueError: test failure", error_log)
                    self.assertIn("in work", error_log)
                saved = stage_timer.load_timings()["stages"][expected]
                self.assertEqual(saved["seconds"], 3.25)
                self.assertEqual(saved["status"], expected)
                self.assertEqual(later, [])

    def test_wait_between_steps_is_excluded(self):
        saved = threading.Event()
        original_record = stage_timer.record_stage

        def record(*args, **kwargs):
            original_record(*args, **kwargs)
            saved.set()

        def first():
            self.advance(8)
            self.runner.pause()

        with patch.object(stage_timer, "record_stage", side_effect=record):
            self.runner.start([
                ("asr", "Transcription", first),
                ("translate", "Translation", lambda: self.advance(3)),
            ])
            try:
                self.assertTrue(saved.wait(5))
                self.advance(1000)
                self.assertEqual(self.runner.timing_snapshot()["stages"]["asr"]["seconds"], 8)
            finally:
                self.runner.resume()
                self.finish()
        self.assertEqual(sum(e["seconds"] for e in stage_timer.load_timings()["stages"].values()), 11)

    def test_stop_on_last_step_is_not_reported_as_complete(self):
        def work():
            self.advance(4)
            self.runner.stop()

        self.runner.start([("tts", "TTS", work)])
        self.finish()
        self.assertEqual(self.runner.state, "stopped")
        self.assertEqual(stage_timer.load_timings()["stages"]["tts"]["status"], "stopped")

    def test_log_failure_does_not_mask_task_error(self):
        def fail():
            raise ValueError("original error")

        with patch.object(stage_timer, "record_stage", side_effect=OSError("disk unavailable")):
            self.runner.start([("tts", "TTS", fail)])
            self.finish()
        self.assertEqual(self.runner.error_msg, "original error")
        self.assertIn("ValueError: original error", self.runner.error_details)
        self.assertTrue(any("Could not save stage timing" in line for line in self.runner.logs))
        self.runner.reset()
        self.assertEqual(self.runner.error_details, "")

    def test_legacy_tasks_and_browser_timer_remain_compatible(self):
        called = []
        self.runner.start([("Untimed", lambda: called.append(True))])
        self.finish()
        self.assertEqual(called, [True])
        self.assertFalse(self.log_path.exists())
        with patch.object(stage_timer, "time", SimpleNamespace(monotonic=lambda: self.now)):
            with self.assertRaises(ValueError):
                with stage_timer.timed_stage("tts-merge", "TTS and merge"):
                    self.advance(9)
                    raise ValueError("test")
        self.assertEqual(stage_timer.load_timings()["stages"]["tts-merge"]["seconds"], 9)
        self.assertTrue(any("Pipeline timings" in line for line in stage_timer.summary_lines()))

    def test_corrupt_log_can_be_replaced_by_new_timing(self):
        self.log_path.parent.mkdir(parents=True)
        for content in ('{"stages":', 'null', '{"stages": null, "media": []}'):
            with self.subTest(content=content):
                self.log_path.write_text(content, encoding="utf-8")
                self.assertEqual(stage_timer.load_timings()["stages"], {})
                stage_timer.record_stage("asr", "Transcription", 2)
                self.assertEqual(stage_timer.load_timings()["stages"]["asr"]["seconds"], 2)

    def test_panel_renders_saved_results_and_download_in_chinese(self):
        from streamlit.testing.v1 import AppTest
        from core.st_utils import timing_panel

        translations = json.loads((ROOT / "translations" / "zh-CN.json").read_text(encoding="utf-8"))
        script = """
from core.st_utils.task_runner import TaskRunner
from core.st_utils.timing_panel import render_timing_panel
render_timing_panel(TaskRunner(), 'test')
"""
        with patch.object(timing_panel, "t", side_effect=lambda key: translations[key]):
            app = AppTest.from_string(script).run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(app.expander[0].label, "耗时统计")
            self.assertIn("此前未记录", app.caption[0].value)
            stage_timer.record_stage("asr", "语音识别", 61, status="completed")
            stage_timer.record_stage("tts", "语音生成", 123, status="error")
            app.run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual([m.value for m in app.metric], ["3:04", "2:00", "1.53x"])
            self.assertEqual(app.dataframe[0].value["状态"].tolist(), ["已完成", "任务出错"])
            self.assertEqual(app.get("download_button")[0].label, "下载耗时日志")

    def test_main_ui_keeps_timings_after_completion_rerun(self):
        from streamlit.testing.v1 import AppTest
        import st as webui

        self.runner.start([("tts", "TTS", lambda: self.advance(7))])
        self.finish()
        app = AppTest.from_string("import st as webui\nwebui.translation_dubbing_section()")
        app.session_state["_translation_runner"] = self.runner
        with (
            patch.object(webui, "_get_translation_dubbing_steps", return_value=[]),
            patch.object(webui, "_render_subtitle_outputs", return_value=False),
            patch.object(webui, "_render_dubbing_outputs", return_value=False),
        ):
            app.run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(self.runner.state, "idle")
            self.assertEqual(app.metric[0].value, "0:07")
            self.assertEqual(len(app.dataframe[0].value), 1)
            app.run(timeout=10)
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "0:07")


if __name__ == "__main__":
    unittest.main(verbosity=2)
