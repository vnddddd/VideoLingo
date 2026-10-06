"""Speaker-picker regression checks using temporary audio, ASR and config.

Run: .venv/Scripts/python tools/_test_speaker_preview.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import _3_speaker_preview as preview


class SpeakerPreviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.chunks = root / "cleaned_chunks.xlsx"
        preview_dir = root / "preview"
        self.config = {"multi_speaker_enabled": True, "speaker_voice_map": {}}
        source = root / "source.wav"
        with source.open("wb") as stream:
            AudioSegment.silent(duration=10000).export(stream, format="wav")
        for name, value in (
            ("_2_CLEANED_CHUNKS", str(self.chunks)),
            ("PREVIEW_DIR", preview_dir),
            ("PENDING_FLAG", preview_dir / ".pending"),
            ("MANIFEST_FILE", preview_dir / "manifest.json"),
            ("CONFIRMED_STATE_FILE", preview_dir / ".confirmed_voice_map.json"),
            ("_source_audio_path", lambda: str(source)),
            ("_source_media_identity", lambda: {"name": "test-video.mp4"}),
            ("load_key", lambda key: self.config[key]),
            ("update_key", lambda key, value: self.config.__setitem__(key, value)),
            ("rprint", lambda *args, **kwargs: None),
        ):
            patcher = patch.object(preview, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_asr(self, extra_count=1, unusable=False):
        rows = []
        for sid, count in (("1", 6), ("2", 5), ("4", extra_count)):
            for _ in range(count):
                start = len(rows) * 0.3
                end = start if sid == "4" and unusable else start + 0.2
                rows.append({"speaker_id": sid, "start": start, "end": end, "text": "word"})
        pd.DataFrame(rows).to_excel(self.chunks, index=False)

    def clone_picks(self):
        return {
            item["speaker_id"]: {"mode": "clone", "ref_wav": item["wav"]}
            for item in preview.read_manifest()
        }

    def test_short_speaker_does_not_block_save_or_prompt_again(self):
        self.write_asr()
        manifest = preview.generate_previews()
        self.assertEqual([item["speaker_id"] for item in manifest], ["1", "2"])
        self.assertTrue(preview.is_pending())
        picks = self.clone_picks()
        preview.confirm_picks(picks)
        saved = self.config["speaker_voice_map"]
        self.assertEqual(saved["4"], {"mode": "default"})
        self.assertEqual({sid: saved[sid] for sid in picks}, picks)
        self.assertNotIn("4", picks)
        self.assertFalse(preview.is_pending())
        self.assertFalse(preview.is_required())

    def test_speaker_without_usable_timing_uses_default(self):
        self.write_asr(extra_count=5, unusable=True)
        preview.generate_previews()
        preview.confirm_picks(self.clone_picks())
        self.assertEqual(self.config["speaker_voice_map"]["4"], {"mode": "default"})
        self.assertFalse(preview.is_required())

    def test_missing_selectable_speaker_still_rejected_without_writes(self):
        self.write_asr(extra_count=5)
        preview.generate_previews()
        picks = self.clone_picks()
        del picks["4"]
        with self.assertRaisesRegex(ValueError, "missing current speaker"):
            preview.confirm_picks(picks)
        self.assertEqual(self.config["speaker_voice_map"], {})
        self.assertTrue(preview.is_pending())
        self.assertFalse(preview.CONFIRMED_STATE_FILE.exists())

    def test_stale_manifest_cannot_hide_a_selectable_speaker(self):
        self.write_asr()
        preview.generate_previews()
        picks = self.clone_picks()
        self.write_asr(extra_count=5)
        with self.assertRaisesRegex(ValueError, "missing current speaker"):
            preview.confirm_picks(picks)

    def test_explicit_short_speaker_choice_is_preserved(self):
        self.write_asr()
        preview.generate_previews()
        picks = self.clone_picks()
        picks["4"] = {"mode": "fixed", "voice": "chosen-voice"}
        preview.confirm_picks(picks)
        self.assertEqual(self.config["speaker_voice_map"], picks)

    def test_new_video_still_requires_confirmation(self):
        self.write_asr()
        preview.generate_previews()
        preview.confirm_picks(self.clone_picks())
        with patch.object(preview, "_source_media_identity", return_value={"name": "new-video.mp4"}):
            self.assertTrue(preview.is_required())


if __name__ == "__main__":
    unittest.main(verbosity=2)
