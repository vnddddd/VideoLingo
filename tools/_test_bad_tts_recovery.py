"""Regression checks for unrecoverably long TTS output.

Run: .venv/Scripts/python -X utf8 tools/_test_bad_tts_recovery.py

The reported failure: Fish Audio sometimes returns a 20-40 s clip for a one
word line by looping. That clip used to survive as a "shortest fallback", so
fit_chunk_audio could not squeeze it into a sub-second window and raised
"Cannot fit audio chunk", aborting the entire dubbing run.

Two behaviours are pinned here:
  * a looping clip is cut back to its leading utterance, not discarded;
  * when nothing usable survives, the line is reported loudly, never
    silently replaced by silence without a trace.
"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydub.generators import Sine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tts = importlib.import_module("core.tts_backend.tts_main")
gen = importlib.import_module("core._10_gen_audio")
vad = importlib.import_module("core.tts_backend.tts_vad")


def speech(path, seconds, sample_rate=16000):
    """Write a real WAV of the requested length so the probes measure it."""
    Sine(440, sample_rate=sample_rate).to_audio_segment(
        duration=round(seconds * 1000)
    ).export(str(path), format="wav")
    return str(path)


def looped_clip(path, repeats=5):
    """A clip shaped like a looping hallucination: real speech, N times over.

    A synthetic tone is not enough here: silero-vad detects no speech in a sine
    wave, so the salvage path would never run on it. Use the project's own
    per-sentence reference clip as the utterance.
    """
    from pydub import AudioSegment

    fixture = ROOT / "output" / "audio" / "refers" / "2619.wav"
    if not fixture.exists():
        raise unittest.SkipTest("no per-line speech fixture available")

    # NOTE: AudioSegment.__iadd__ mutates in place, so `combined += unit` inside a
    # loop would just keep extending the SAME segment. Build the list and join.
    unit = AudioSegment.from_file(fixture)[:700]
    gap = AudioSegment.silent(duration=300, frame_rate=unit.frame_rate)
    parts = []
    for index in range(repeats):
        parts.append(unit)
        if index < repeats - 1:
            parts.append(gap)
    combined = sum(parts[1:], parts[0])
    combined.export(str(path), format="wav")
    return str(path)


class BadTTSRaisesInsteadOfPoisoningTheTimeline(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="videolingo-badtts-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = {"tts_method": "fish_audio_tts"}
        self._stub(tts, "load_key", new=lambda key: self.config[key])
        self._stub(tts, "rprint")
        self._stub(tts, "_TTS_ESTIMATOR", new=object())
        self._stub(tts, "estimate_duration", new=lambda text, est=None: 0.5)
        self._stub(tts, "ensure_non_empty_wav")
        self._stub(tts, "vad_compact_tts_audio")
        self._stub(tts, "ask_gpt", new=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no LLM")))
        self._stub(gen, "TEMP_FILE_TEMPLATE", new=str(self.root / "{}_temp.wav"))
        self._stub(tts, "_BAD_TTS_REPORT", new=str(self.root / "failures.jsonl"))

    def _stub(self, module, name, **kwargs):
        patcher = patch.object(module, name, **kwargs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_looping_clip_is_salvaged_not_discarded(self):
        """The leading utterance survives; the loop tail is cut away."""

        def backend(text, save_as, voice_cfg=None, number=None):
            looped_clip(save_as, repeats=6)

        def fake_trim(path):
            # Real VAD would compact; the stub keeps the fixture's shape.
            return False

        with (
            patch.object(tts, "fish_audio_tts", side_effect=backend),
            patch.object(tts, "vad_compact_tts_audio", side_effect=fake_trim),
            patch.object(tts, "_salvage_leading_speech", wraps=tts._salvage_leading_speech),
        ):
            try:
                tts.tts_main("谢谢", str(self.root / "line.wav"), 2619, None)
            except tts.BadTTSAudioError:
                pass

        target = self.root / "line.wav"
        self.assertTrue(target.exists(), "a salvaged line must leave audio behind")
        kept = tts.get_audio_duration(str(target))
        self.assertLess(kept, 3.0, "the loop tail must be cut off so the line fits")

    def test_unsalvageable_line_raises_and_is_reported(self):
        """Nothing usable in, nothing silent out: raise and record it."""
        def backend(text, save_as, voice_cfg=None, number=None):
            speech(save_as, 9.0)

        with (
            patch.object(tts, "fish_audio_tts", side_effect=backend),
            patch.object(tts, "_salvage_leading_speech", return_value=None),
        ):
            with self.assertRaises(tts.BadTTSAudioError):
                tts.tts_main("谢谢", str(self.root / "line.wav"), 2619, None)

        report = self.root / "failures.jsonl"
        self.assertTrue(report.exists(), "an unvoiced line must be recorded")
        record = json.loads(report.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(record["action"], "dropped")
        self.assertIn("谢谢", record["text"])

    def test_unvoiced_line_is_surfaced_not_silently_dropped(self):
        """The caller keeps the cue but announces the defect."""
        row = pd.Series({
            "number": 2619,
            "lines": ["不客气", "谢谢你 Suri"],
            "speaker_id": None,
        })

        def fake_tts(text, save_as, number, tasks_df, speaker_id=None):
            if text == "谢谢你 Suri":
                raise tts.BadTTSAudioError("model kept looping")
            speech(save_as, 0.4)

        gen._UNVOICED_LINES.clear()
        printed = []
        with (
            patch.object(gen, "tts_main", side_effect=fake_tts),
            patch.object(gen, "rprint", side_effect=lambda *a, **k: printed.append(str(a))),
            patch.object(gen, "get_row_speaker_id", return_value=None),
        ):
            number, total = gen.process_row(row, pd.DataFrame())
            gen.print_unvoiced_summary()

        self.assertEqual(number, 2619)
        self.assertAlmostEqual(total, 0.7, places=2)
        self.assertEqual(len(gen._UNVOICED_LINES), 1)
        joined = " ".join(printed)
        self.assertIn("NOT be dubbed", joined, "the defect must be stated, not hidden")

    def test_salvage_rejects_a_first_segment_that_is_still_too_long(self):
        """If even the leading utterance overruns, there is nothing to keep."""
        clip = looped_clip(self.root / "long.wav", repeats=3)
        self.assertIsNone(tts._salvage_leading_speech(clip, limit_s=0.5))

    def test_salvage_ignores_a_single_clean_utterance(self):
        """A normal clip is not a loop and must not be truncated."""
        clip = speech(self.root / "clean.wav", 0.8)
        self.assertIsNone(tts._salvage_leading_speech(clip, limit_s=2.0))

    def test_salvage_keeps_exactly_one_utterance(self):
        """Six repeats and three repeats both reduce to one utterance."""
        three = tts._salvage_leading_speech(looped_clip(self.root / "a.wav", 3), 5.0)
        six = tts._salvage_leading_speech(looped_clip(self.root / "b.wav", 6), 5.0)
        self.assertIsNotNone(three)
        self.assertIsNotNone(six)
        self.assertAlmostEqual(three, six, delta=0.35)


if __name__ == "__main__":
    unittest.main(verbosity=2)