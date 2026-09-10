"""Regression checks for chunk timing; all API calls are replaced with test doubles.

Run: .venv/Scripts/python tools/_test_audio_chunk_fit.py
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydub.generators import Sine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import _10_gen_audio as audio


def task(number, start, end, lines, *, cut_off=1, gap=0):
    return {
        "number": number, "start_time": start, "end_time": end,
        "lines": list(lines), "text": " ".join(lines),
        "gap": gap, "tolerance": 0, "cut_off": cut_off,
    }


def failing_chunk():
    frame = pd.DataFrame([
        task(134, "00:06:57.269", "00:07:00.269", ["First", "Second", "Third"], cut_off=0),
        task(137, "00:07:00.269", "00:07:04.529", ["Fourth", "Long sentence A"], cut_off=0),
        task(139, "00:07:04.529", "00:07:07.709", ["Long sentence B", "Short", "Last short sentence"]),
    ])
    durations = {
        (134, 0): 1.246208, (134, 1): 1.177333, (134, 2): 1.312958,
        (137, 0): 1.657, (137, 1): 2.682542,
        (139, 0): 2.370458, (139, 1): 0.618625, (139, 2): 0.890792,
    }
    return frame, durations


class ChunkFitTests(unittest.TestCase):
    def setUp(self):
        config = {"speed_factor.max": 1.5, "speed_factor.accept": 1.2, "speed_factor.min": 1.0, "tts_method": "test"}
        for name, value in (
            ("load_key", lambda key: config[key]),
            ("load_positive_int", lambda *args, **kwargs: 2),
            ("rprint", lambda *args, **kwargs: None),
        ):
            patcher = patch.object(audio, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        blocker = patch.object(audio, "tts_main", side_effect=AssertionError("Unexpected TTS request"))
        blocker.start()
        self.addCleanup(blocker.stop)
        blocker = patch.object(audio, "ask_gpt", side_effect=AssertionError("Unexpected LLM request"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_original_failure_gets_positive_shared_targets(self):
        frame, durations = failing_chunk()
        old_last_target = durations[(139, 2)] - (sum(durations.values()) - 10.44)
        self.assertLess(old_last_target, 0)

        def fit(frame, row, line, path, target, speed, **kwargs):
            self.assertGreater(target, audio.MIN_SEGMENT_DURATION_MS / 1000)
            self.assertEqual(speed, 1.5)
            self.assertFalse(kwargs["allow_overrun"])
            return target

        with patch.object(audio, "fit_or_shorten_line", side_effect=fit) as fitter:
            keep = audio.fit_chunk_audio(frame, 0, 2, durations, 10.44, 1.5, False)
        self.assertFalse(keep)
        self.assertGreater(fitter.call_count, 1)
        self.assertLessEqual(sum(durations.values()), 10.441)
        self.assertTrue(all(value > 0 for value in durations.values()))

    def test_optional_gaps_are_reclaimed_before_refitting_speech(self):
        frame = pd.DataFrame([
            task(1, "00:00:00.000", "00:00:01.000", ["One"], cut_off=0, gap=2),
            task(2, "00:00:03.000", "00:00:04.000", ["Two"]),
        ])
        durations = {(1, 0): 1.0, (2, 0): 1.0}
        with patch.object(audio, "fit_or_shorten_line") as fitter:
            self.assertFalse(audio.fit_chunk_audio(frame, 0, 1, durations, 3, 1, True))
            fitter.assert_not_called()
        self.assertEqual(sum(durations.values()), 2)

    def test_fitting_chunk_is_unchanged(self):
        frame, durations = failing_chunk()
        before = durations.copy()
        with patch.object(audio, "fit_or_shorten_line") as fitter:
            self.assertTrue(audio.fit_chunk_audio(frame, 0, 2, durations, 15, 1.5, True))
            fitter.assert_not_called()
        self.assertEqual(durations, before)

    def test_overrun_limit_is_for_the_chunk_not_each_line(self):
        frame = pd.DataFrame([task(1, "00:00:00.000", "00:00:02.700", ["A", "B", "C"])])
        durations = {(1, i): 1.4 for i in range(3)}
        with (
            patch.object(audio, "fit_or_shorten_line", side_effect=audio.AudioFitTooFastError("test", 2, 1.5)),
            patch.object(audio, "get_audio_duration", return_value=1.4),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot fit audio chunk"):
                audio.fit_chunk_audio(frame, 0, 0, durations, 2.7, 1.5, False)

    def test_failed_refit_remeasures_the_changed_file(self):
        frame = pd.DataFrame([task(1, "00:00:00.000", "00:00:03.500", ["A", "B"])])
        durations = {(1, 0): 3, (1, 1): 2}

        def fit(frame, row, line, path, target, speed, **kwargs):
            if line == 0:
                raise audio.AudioFitTooFastError(path, 2, 1.5)
            return target

        with (
            patch.object(audio, "fit_or_shorten_line", side_effect=fit),
            patch.object(audio, "get_audio_duration", return_value=2.5),
        ):
            audio.fit_chunk_audio(frame, 0, 0, durations, 3.5, 1.5, False)
        self.assertEqual(durations[(1, 0)], 2.5)
        self.assertAlmostEqual(sum(durations.values()), 3.5)

    def test_strict_line_fit_cannot_silently_allow_one_second(self):
        frame = pd.DataFrame([task(1, "00:00:00.000", "00:00:01.000", ["Short"])])
        with (
            patch.object(audio, "get_audio_duration", return_value=1.1),
            patch.object(audio, "try_native_speed_refit", return_value=None),
            patch.object(audio, "shorten_text_for_audio_fit", return_value="Short"),
        ):
            self.assertEqual(audio.fit_or_shorten_line(frame, 0, 0, "test.wav", 1, 1.5), 1.1)
            with self.assertRaises(audio.AudioFitTooFastError):
                audio.fit_or_shorten_line(frame, 0, 0, "test.wav", 1, 1.5, allow_overrun=False)

    def test_speed_limit_is_not_bypassed(self):
        with (
            patch.object(audio, "get_audio_duration", return_value=3),
            patch.object(audio, "adjust_audio_speed") as adjust,
        ):
            with self.assertRaises(audio.AudioFitTooFastError):
                audio.fit_audio_to_duration("test.wav", 2, 1.5)
            adjust.assert_not_called()

    def merge_without_io(self, frame, durations, fit):
        def read(path):
            stem = Path(path).stem.split("_")
            return durations[(int(stem[0]), int(stem[1]))]

        with (
            patch.object(audio, "process_chunk", return_value=(1.5, False)),
            patch.object(audio, "native_speed_enabled", return_value=False),
            patch.object(audio, "adjust_audio_speed"),
            patch.object(audio, "get_audio_duration", side_effect=read),
            patch.object(audio, "fit_or_shorten_line", side_effect=fit),
        ):
            return audio.merge_chunks(frame)

    def test_all_timestamps_are_rebuilt_after_chunk_refit(self):
        frame, durations = failing_chunk()

        def fit(frame, row, line, path, target, speed, **kwargs):
            durations[(frame.at[row, "number"], line)] = target
            return target

        result = self.merge_without_io(frame, durations, fit)
        previous_end = 417.269
        for _, row in result.iterrows():
            for line, (start, end) in enumerate(row["new_sub_times"]):
                self.assertAlmostEqual(start, previous_end)
                self.assertGreater(end, start)
                self.assertAlmostEqual(end - start, durations[(row["number"], line)])
                previous_end = end
        self.assertLessEqual(previous_end, 427.710)

    def test_tolerated_overrun_does_not_overlap_the_next_chunk(self):
        frame = pd.DataFrame([
            task(1, "00:00:00.000", "00:00:01.000", ["One"]),
            task(2, "00:00:01.000", "00:00:04.000", ["Two"]),
        ])
        durations = {(1, 0): 1.4, (2, 0): 2.0}

        def cannot_fit(*args, **kwargs):
            raise audio.AudioFitTooFastError("test", 2, 1.5)

        result = self.merge_without_io(frame, durations, cannot_fit)
        first = result.at[0, "new_sub_times"][0]
        second = result.at[1, "new_sub_times"][0]
        self.assertAlmostEqual(first[1], 1.4)
        self.assertAlmostEqual(second[0], first[1])
        self.assertLessEqual(second[1], 4)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is unavailable")
    def test_real_wavs_fit_with_available_speed_headroom(self):
        frame, durations = failing_chunk()
        with tempfile.TemporaryDirectory(prefix="videolingo-chunk-fit-") as directory:
            template = str(Path(directory) / "{}.wav")
            for key, duration in durations.items():
                Sine(440, sample_rate=16000).to_audio_segment(duration=round(duration * 1000)).export(
                    template.format(f"{key[0]}_{key[1]}"), format="wav",
                ).close()
            with patch.object(audio, "OUTPUT_FILE_TEMPLATE", template):
                audio.fit_chunk_audio(frame, 0, 2, durations, 10.44, 1.0, False)
            self.assertLessEqual(sum(durations.values()), 10.44 + audio.SAFE_TIMELINE_OVERRUN_SECONDS)
            self.assertTrue(all(value > 0 for value in durations.values()))
            for key in durations:
                self.assertTrue(Path(template.format(f"{key[0]}_{key[1]}")).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
