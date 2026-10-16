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


class _PreviewTestBase(unittest.TestCase):
    """Shared fixtures: temp audio, temp ASR workbook, temp config."""

    # Subclasses decide whether the global engine clones (auto-clone path) or
    # not (manual picker path).
    CLONE_MODEL = None

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

        # None -> preset/unsupported engine -> the manual picker runs.
        # A model name -> the global engine clones -> the picker is skipped.
        clone = patch.object(preview, "global_clone_model",
                             return_value=type(self).CLONE_MODEL)
        clone.start()
        self.addCleanup(clone.stop)

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


class SpeakerPreviewTests(_PreviewTestBase):
    """Manual picker flow: a preset (non-cloning) global engine."""

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


class AutoCloneTests(_PreviewTestBase):
    """When the global engine clones, the picker must never appear."""

    CLONE_MODEL = "s2.1-pro-free"

    def test_auto_clone_fills_map_and_never_pends(self):
        self.write_asr()
        manifest = preview.generate_previews()
        self.assertEqual([item["speaker_id"] for item in manifest], ["1", "2"])
        # No picker: the flag stays down even on a brand-new video.
        self.assertFalse(preview.is_pending())
        saved = self.config["speaker_voice_map"]
        for item in manifest:
            self.assertEqual(saved[item["speaker_id"]], {
                "mode": "clone", "ref_wav": item["wav"],
            })
        # The short speaker has no usable clip -> global voice, not an error.
        self.assertEqual(saved["4"], {"mode": "default"})

    def test_confirmation_sidecar_is_written_so_reruns_stay_quiet(self):
        self.write_asr()
        preview.generate_previews()
        self.assertTrue(preview.CONFIRMED_STATE_FILE.exists())
        self.assertFalse(preview.is_required())

    def test_pending_flag_from_earlier_run_is_cleared(self):
        # Simulate an older run that paused before this shortcut existed.
        self.write_asr()
        preview.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        preview._set_pending({"speaker_ids": ["1"]})
        preview.generate_previews()
        self.assertFalse(preview.is_pending())

    def test_preset_engine_still_asks_for_confirmation(self):
        # A preset voice is the one global setting that means "do not clone".
        with patch.object(preview, "global_clone_model", return_value=None):
            self.write_asr()
            preview.generate_previews()
            self.assertTrue(preview.is_pending())

    def test_switching_to_a_clone_engine_replaces_fixed_picks(self):
        # A map confirmed for a non-clone engine must be refreshed, not honored.
        self.write_asr()
        self.config["speaker_voice_map"] = {
            "1": {"mode": "fixed", "voice": "zh-CN-XiaoxiaoNeural"},
            "2": {"mode": "clone", "ref_wav": "old.wav"},
            "4": {"mode": "default"},
        }
        self.assertTrue(preview.is_required())
        preview.generate_previews()
        self.assertFalse(preview.is_pending())
        saved = self.config["speaker_voice_map"]
        self.assertEqual(saved["1"]["mode"], "clone")
        self.assertEqual(saved["4"], {"mode": "default"})

    def test_default_pick_does_not_force_a_refresh(self):
        # 'default' is not a conflict: it just means "use the global voice".
        self.write_asr()
        preview.generate_previews()
        saved = self.config["speaker_voice_map"]
        self.assertEqual(saved["4"], {"mode": "default"})
        self.assertFalse(preview.is_required())

    def test_single_usable_speaker_still_uses_global_voice(self):
        self.write_asr(extra_count=0)
        self.config["speaker_voice_map"] = {}
        self.write_asr_one_speaker()
        manifest = preview.generate_previews()
        self.assertLess(len(manifest), preview.MIN_SPEAKERS_TO_TRIGGER)
        self.assertFalse(preview.is_pending())

    def write_asr_one_speaker(self):
        rows = [
            {"speaker_id": "1", "start": i * 0.3, "end": i * 0.3 + 0.2, "text": "word"}
            for i in range(6)
        ]
        pd.DataFrame(rows).to_excel(self.chunks, index=False)


class GlobalCloneModelTests(unittest.TestCase):
    """The trigger rule shared by the Web UI, the plugin and the pipeline."""

    def setUp(self):
        from core.utils import speaker_router

        self.router = speaker_router
        self.config = {}
        # The base fixtures patch module attributes; here only config matters.
        patcher = patch.object(
            speaker_router, "load_key",
            side_effect=lambda key: self.config.get(key),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def model(self, **config):
        self.config = config
        return self.router.global_clone_model()

    def test_fish_custom_and_dynamic_clone(self):
        for mode in ("custom", "dynamic"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.model(**{
                        "tts_method": "fish_audio_tts",
                        "fish_audio_tts.mode": mode,
                        "fish_audio_tts.model": "s2.1-pro-free",
                    }),
                    "s2.1-pro-free",
                )

    def test_fish_preset_keeps_the_picker(self):
        self.assertIsNone(self.model(**{
            "tts_method": "fish_audio_tts", "fish_audio_tts.mode": "preset",
            "fish_audio_tts.model": "s2.1-pro-free",
        }))

    def test_fish_s1_is_paid_only_and_not_multi_speaker(self):
        self.assertIsNone(self.model(**{
            "tts_method": "fish_audio_tts", "fish_audio_tts.mode": "custom",
            "fish_audio_tts.model": "s1",
        }))

    def test_soniox_clone_mode_clones(self):
        self.assertEqual(self.model(**{
            "tts_method": "soniox_tts", "soniox_tts.mode": "clone",
            "soniox_tts.model": "tts-rt-v2",
        }), "tts-rt-v2")

    def test_soniox_preset_keeps_the_picker(self):
        self.assertIsNone(self.model(**{
            "tts_method": "soniox_tts", "soniox_tts.mode": "preset",
            "soniox_tts.model": "tts-rt-v2",
        }))

    def test_mimo_only_clones_on_the_voiceclone_model(self):
        self.assertEqual(self.model(**{
            "tts_method": "mimo_tts", "mimo_tts.model": "mimo-v2.5-tts-voiceclone",
        }), "mimo-v2.5-tts-voiceclone")
        for other in ("mimo-v2.5-tts", "mimo-v2.5-tts-voicedesign"):
            with self.subTest(model=other):
                self.assertIsNone(self.model(**{
                    "tts_method": "mimo_tts", "mimo_tts.model": other,
                }))

    def test_sf_fish_follows_its_mode(self):
        self.assertEqual(self.model(**{
            "tts_method": "sf_fish_tts", "sf_fish_tts.mode": "custom",
        }), "sf_fish_tts")
        self.assertIsNone(self.model(**{
            "tts_method": "sf_fish_tts", "sf_fish_tts.mode": "preset",
        }))

    def test_engines_without_preset_concept_always_clone(self):
        for method in ("gpt_sovits", "sf_cosyvoice2", "f5tts", "indextts2"):
            with self.subTest(method=method):
                self.assertEqual(self.model(**{"tts_method": method}), method)

    def test_non_clone_engines_keep_the_picker(self):
        for method in ("edge_tts", "qwen3_tts", "azure_tts", "openai_tts", None):
            with self.subTest(method=method):
                self.assertIsNone(self.model(**{"tts_method": method}))

    def test_missing_mode_keys_use_their_documented_defaults(self):
        # fish defaults to custom (clones); soniox defaults to preset (asks).
        self.assertEqual(self.model(**{"tts_method": "fish_audio_tts"}), "fish_audio_tts")
        self.assertIsNone(self.model(**{"tts_method": "soniox_tts"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
