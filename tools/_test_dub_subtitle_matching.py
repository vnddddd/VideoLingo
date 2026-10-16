"""Regression checks for dubbing text alignment; no API calls or real output writes.

Run: .venv/Scripts/python tools/_test_dub_subtitle_matching.py
"""

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import _8_1_audio_task as tasks
from core import _8_2_dub_chunks as chunks


def write_srt(path, texts):
    path.write_text("\n\n".join(
        f"{i + 1}\n00:00:{i * 5:02d},000 --> 00:00:{(i + 1) * 5:02d},000\n{text}"
        for i, text in enumerate(texts)
    ), encoding="utf-8")


class SubtitleMatchingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="videolingo-subtitle-matching-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.workbook = self.directory / "tasks.xlsx"
        self.audio_subtitles = self.directory / "audio.srt"
        self.audio_source = self.directory / "audio_source.srt"
        self.subtitles = self.directory / "trans.srt"
        self.source = self.directory / "src.srt"
        for module, name, value in (
            (tasks, "TRANS_SUBS_FOR_AUDIO_FILE", self.audio_subtitles),
            (tasks, "SRC_SUBS_FOR_AUDIO_FILE", self.audio_source),
            (tasks, "_AUDIO_SPEAKERS", self.directory / "missing_speakers.xlsx"),
            (tasks, "load_key", lambda key: 2.5),
            (tasks, "ask_gpt", Mock(side_effect=AssertionError("Unexpected API call"))),
            (chunks, "_8_1_AUDIO_TASK", self.workbook),
            (chunks, "TRANS_SRT", self.subtitles),
            (chunks, "SRC_SRT", self.source),
            (chunks, "analyze_subtitle_timing_and_speed", lambda df: df),
            (chunks, "process_cutoffs", lambda df: df),
            (chunks, "rprint", lambda *args, **kwargs: None),
        ):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self, audio_texts, display_texts):
        write_srt(self.audio_subtitles, audio_texts)
        write_srt(self.audio_source, [f"Source {i}" for i in range(len(audio_texts))])
        write_srt(self.subtitles, display_texts)
        write_srt(self.source, [f"Source ({i})" for i in range(len(display_texts))])
        frame = tasks.process_srt()
        frame.to_excel(self.workbook, index=False)
        return frame

    def test_formula_across_subtitle_boundary_preserves_content_and_alignment(self):
        for left, right in (("（", "）"), ("(", ")")):
            with self.subTest(parentheses=left + right):
                first = f"也就是 1 减去 S 等于{left}δ＋"
                second = f"δ的十次方{right}除以{left}1＋δ{right}"
                frame = self.prepare([first + " " + second, "下一句"], [first, second, "下一句"])
                self.assertEqual(frame.iloc[0]["text"], "也就是 1 减去 S 等于δ＋ δ的十次方除以1＋δ")
                expected = pd.read_excel(self.workbook)
                chunks.gen_dub_chunks()
                result = pd.read_excel(self.workbook)
                self.assertEqual(ast.literal_eval(result.iloc[0]["lines"]), [
                    "也就是 1 减去 S 等于δ＋", "δ的十次方除以1＋δ",
                ])
                self.assertEqual(ast.literal_eval(result.iloc[0]["src_lines"]), ["Source 0", "Source 1"])
                self.assertEqual(ast.literal_eval(result.iloc[1]["lines"]), ["下一句"])
                pd.testing.assert_frame_equal(result[expected.columns], expected)

    def test_parenthetical_and_nested_content_is_not_silently_lost(self):
        self.prepare(["（δ的十次方）", "收益（也就是（利润））"], ["（δ的十次方）", "收益（也就是（利润））"])
        chunks.gen_dub_chunks()
        result = pd.read_excel(self.workbook)
        self.assertEqual(result["text"].tolist(), ["δ的十次方", "收益也就是利润"])
        self.assertEqual(ast.literal_eval(result.iloc[0]["lines"]), ["δ的十次方"])

    def test_plain_text_and_existing_hyphen_handling(self):
        self.prepare(["-Hello, world!", "下一句。"], ["-Hello,", "world!", "下一句。"])
        chunks.gen_dub_chunks()
        result = pd.read_excel(self.workbook)
        self.assertEqual(result["text"].tolist(), ["Hello, world!", "下一句。"])
        self.assertEqual(ast.literal_eval(result.iloc[0]["lines"]), ["Hello,", "world!"])
        self.assertEqual(ast.literal_eval(result.iloc[1]["src_lines"]), ["Source 2"])

    def test_unrelated_text_mismatch_still_fails_without_saving_bad_alignment(self):
        self.prepare(["甲乙"], ["甲丙"])
        original = self.workbook.read_bytes()
        with self.assertRaisesRegex(ValueError, "Matching failed"):
            chunks.gen_dub_chunks()
        self.assertEqual(self.workbook.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
