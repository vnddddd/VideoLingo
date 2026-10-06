"""Check speaker routing through TTS dispatch and Fish request construction.

Run: .venv/Scripts/python -X utf8 tools/_test_speaker_clone_routing.py
Cloud calls are replaced with test doubles; only temporary WAVs are written.
"""

import importlib
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

router = importlib.import_module("core.utils.speaker_router")
tts = importlib.import_module("core.tts_backend.tts_main")
fish = importlib.import_module("core.tts_backend.fish_audio_tts")
long_ref = importlib.import_module("core.utils._long_ref_extractor")


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    return buffer.getvalue()


class SpeakerCloneRoutingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.audio = wav_bytes()
        self.refs = {}
        for sid in ("1", "2"):
            path = self.root / f"speaker_{sid}.wav"
            path.write_bytes(self.audio)
            self.refs[sid] = path
        self.config = {
            "multi_speaker_enabled": True,
            "tts_method": "fish_audio_tts",
            "speaker_voice_map": {
                sid: {"mode": "clone", "ref_wav": str(path)}
                for sid, path in self.refs.items()
            },
            "fish_audio_tts.api_key": "test-key",
            "fish_audio_tts.model": "s2.1-pro-free",
            "fish_audio_tts.mode": "custom",
        }
        for module in (router, tts, fish):
            self.patch(module, "load_key", side_effect=lambda key: self.config[key])
            self.patch(module, "rprint")
        self.patch(tts, "_TTS_ESTIMATOR", object())
        self.patch(tts, "estimate_duration", return_value=1.0)
        self.patch(tts, "vad_compact_tts_audio")
        self.soniox = self.patch(tts, "soniox_tts", side_effect=AssertionError("Unexpected Soniox call"))
        self.patch(tts, "ask_gpt", side_effect=AssertionError("Unexpected LLM call"))
        self.global_ref = self.patch(long_ref, "ensure_long_ref", side_effect=AssertionError("Unexpected shared reference"))
        self.patch(fish.requests.sessions.Session, "request", side_effect=AssertionError("Unexpected network call"))
        self.clone = self.patch(fish, "ensure_cloned_voice", side_effect=lambda path, key: f"voice-{path.stem}")
        self.post = self.patch(fish, "_post", return_value=self.audio)

    def patch(self, target, name, *args, **kwargs):
        patcher = patch.object(target, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def synthesize(self, sid, filename=None):
        path = self.root / (filename or f"output_{sid}.wav")
        tts.tts_main("Hello world", str(path), 1, None, speaker_id=sid)
        with wave.open(str(path), "rb") as stream:
            self.assertGreater(stream.getnframes(), 0)

    def test_fish_model_and_individual_references_reach_request(self):
        for model in ("s2.1-pro-free", "s2.1-pro"):
            with self.subTest(model=model):
                self.config["fish_audio_tts.model"] = model
                for sid in ("1", "2"):
                    self.synthesize(sid, f"{model}_{sid}.wav")
                    self.clone.assert_called_with(self.refs[sid], "test-key")
                    payload, headers, msgpack = self.post.call_args.args
                    self.assertEqual(headers["model"], model)
                    self.assertEqual(payload["reference_id"], f"voice-speaker_{sid}")
                    self.assertNotIn("references", payload)
                    self.assertFalse(msgpack)
        self.assertEqual(self.post.call_count, 4)
        self.soniox.assert_not_called()
        self.global_ref.assert_not_called()

    def test_dynamic_mode_keeps_each_speakers_inline_audio(self):
        self.config["fish_audio_tts.mode"] = "dynamic"
        self.patch(fish, "_prepare_reference_clip", side_effect=lambda path: path.stem.encode())
        self.patch(fish, "_reference_text_for", side_effect=lambda path: f"text-{path.stem}")
        for sid in ("1", "2"):
            self.synthesize(sid)
            payload, headers, msgpack = self.post.call_args.args
            self.assertEqual(payload["references"], [{
                "audio": f"speaker_{sid}".encode(), "text": f"text-speaker_{sid}",
            }])
            self.assertEqual(headers["model"], "s2.1-pro-free")
            self.assertTrue(msgpack)
        self.clone.assert_not_called()
        self.global_ref.assert_not_called()

    def test_explicit_clone_overrides_global_fish_preset(self):
        self.config["fish_audio_tts.mode"] = "preset"
        self.config["fish_audio_tts.reference_id"] = "global-preset"
        self.synthesize("1")
        self.assertEqual(self.post.call_args.args[0]["reference_id"], "voice-speaker_1")

    def test_switching_global_backend_changes_existing_clone_picks(self):
        for method in ("fish_audio_tts", "soniox_tts", "mimo_tts", "indextts2",
                       "gpt_sovits", "sf_fish_tts", "sf_cosyvoice2", "f5tts"):
            with self.subTest(method=method):
                self.config["tts_method"] = method
                voice = router.resolve_voice_cfg("1")
                self.assertEqual(voice["method"], method)
                self.assertEqual(voice["ref_wav"], str(self.refs["1"]))
                self.assertTrue(voice["is_clone"])

    def test_unsupported_backend_is_not_silently_dispatched(self):
        self.config["tts_method"] = "edge_tts"
        edge = self.patch(tts, "edge_tts")
        with self.assertRaisesRegex(router.UnsupportedCloneBackendError, "edge_tts"):
            self.synthesize("1")
        edge.assert_not_called()
        self.post.assert_not_called()
        self.soniox.assert_not_called()

    def test_default_and_disabled_modes_still_use_global_path(self):
        self.config["speaker_voice_map"]["1"] = {"mode": "default"}
        self.assertIsNone(router.resolve_voice_cfg("1"))
        self.config["multi_speaker_enabled"] = False
        self.assertIsNone(router.resolve_voice_cfg("2"))

    def test_numeric_excel_speaker_id_keeps_own_reference(self):
        for sid in (1, 1.0, "1.0"):
            self.assertEqual(router.resolve_voice_cfg(sid)["ref_wav"], str(self.refs["1"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
