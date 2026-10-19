"""Regression checks for Fish Audio voice-clone identity.

Run: .venv/Scripts/python tools/_test_fish_voice_identity.py

The bug these tests pin down: voice models were titled by a per-video
"family" id plus a running index, so every speaker of every video collapsed
onto one family and reuse matched on title alone. A female speaker could be
handed a male speaker's voice from an unrelated, much older run.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.tts_backend import fish_audio_tts as fish


class VoiceNameTests(unittest.TestCase):
    """A title must be a function of the reference audio and nothing else."""

    def test_different_clips_get_different_names(self):
        female = fish._voice_name(b"audio-of-a-female-speaker")
        male = fish._voice_name(b"audio-of-a-male-speaker")
        self.assertNotEqual(female, male)

    def test_same_clip_gets_the_same_name(self):
        clip = b"identical-reference-bytes"
        self.assertEqual(fish._voice_name(clip), fish._voice_name(clip))

    def test_name_has_the_project_prefix(self):
        self.assertTrue(fish._voice_name(b"x").startswith(fish.VOICE_NAME_PREFIX))

    def test_name_does_not_depend_on_account_state(self):
        # The old scheme derived the index from how many models already existed,
        # so the same speaker got a different name on every rerun.
        clip = b"stable-reference"
        with patch.object(fish, "list_voice_models", return_value=[]):
            first = fish._voice_name(clip)
        with patch.object(fish, "list_voice_models", return_value=[{"title": f"vl_{i}"} for i in range(50)]):
            second = fish._voice_name(clip)
        self.assertEqual(first, second)


class ClonedVoiceReuseTests(unittest.TestCase):
    """Reuse must follow the clip, never a stale same-titled model."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.female = self.root / 'spk_0.wav'
        self.male = self.root / 'spk_1.wav'
        self.female.write_bytes(b"female-clip")
        self.male.write_bytes(b"male-clip")
        # Reset the in-process id cache between tests.
        fish._clone_cache.clear()
        self.addCleanup(fish._clone_cache.clear)
        for name, value in (
            ('_load_api_key', lambda: 'key'),
            ('_wait_until_trained', lambda *a, **k: None),
            ('rprint', lambda *a, **k: None),
        ):
            patcher = patch.object(fish, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_no_cross_speaker_reuse(self):
        """The exact reported bug: speaker 1 must not receive speaker 2's voice."""
        created = {}

        def fake_create(name, clip, api_key):
            model_id = f'id-{len(created)}'
            created[model_id] = {'_id': model_id, 'title': name, 'created_at': '2026-01-01'}
            return model_id

        def fake_list(api_key=None):
            return list(created.values())

        with patch.object(fish, '_create_voice_model', side_effect=fake_create), \
             patch.object(fish, 'list_voice_models', side_effect=fake_list), \
             patch.object(fish, '_prepare_reference_clip', side_effect=lambda p: Path(p).read_bytes()):
            female_id = fish.ensure_cloned_voice(self.female)
            male_id = fish.ensure_cloned_voice(self.male)

        self.assertNotEqual(female_id, male_id, 'two speakers shared one voice model')
        self.assertEqual(len(created), 2)

    def test_same_speaker_reuses_its_own_model(self):
        created = []

        def fake_create(name, clip, api_key):
            created.append(name)
            return f'id-{len(created)}'

        def fake_list(api_key=None):
            return [{'_id': f'id-{i+1}', 'title': n, 'created_at': '2026-01-01'}
                    for i, n in enumerate(created)]

        with patch.object(fish, '_create_voice_model', side_effect=fake_create), \
             patch.object(fish, 'list_voice_models', side_effect=fake_list), \
             patch.object(fish, '_prepare_reference_clip', side_effect=lambda p: Path(p).read_bytes()):
            first = fish.ensure_cloned_voice(self.female)
            # A new file, same audio bytes: must resolve to the existing model.
            twin = self.root / 'spk_0_copy.wav'
            twin.write_bytes(b'female-clip')
            second = fish.ensure_cloned_voice(twin)

        self.assertEqual(first, second)
        self.assertEqual(len(created), 1, 'duplicate voice model was created')


class PruneSafetyTests(unittest.TestCase):
    """Automatic deletion must never run behind the user's back."""

    def test_prune_never_deletes_anything(self):
        deleted = []
        models = [{'_id': f'id-{i}', 'title': f'vl_{i}', 'created_at': '2026-01-01'} for i in range(80)]
        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish, 'list_voice_models', return_value=models), \
             patch.object(fish, 'delete_voice_model', side_effect=lambda *a, **k: deleted.append(a)), \
             patch.object(fish, '_load_opt', side_effect=lambda k, d=None: 0 if k == 'fish_audio_tts.max_voice_slots' else d), \
             patch.object(fish, 'rprint', lambda *a, **k: None):
            removed = fish.prune_voice_models()
        self.assertEqual(removed, 0)
        self.assertEqual(deleted, [], 'prune deleted user voices')


class Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
    def json(self):
        return self._payload


class PaginationTests(unittest.TestCase):
    """The real endpoint ignores offset and always says has_more=true."""

    def setUp(self):
        fish._invalidate_voice_list_cache()
        self.addCleanup(fish._invalidate_voice_list_cache)

    def test_single_page_account(self):
        page = [{'_id': f'a{i}', 'title': f'vl_a{i}'} for i in range(5)]
        calls = []

        def fake_get(url, headers=None, params=None, timeout=None):
            calls.append(dict(params))
            return Resp({'items': page, 'total': 5, 'has_more': True})

        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish.requests, 'get', side_effect=fake_get):
            got = fish.list_voice_models()

        self.assertEqual(len(got), 5)
        self.assertEqual(len(calls), 1, 'short page should stop after one request')

    def test_ignores_offset_and_has_more_never_loops(self):
        # Reproduces the live endpoint: same 100 models every time, has_more=true
        # forever. The old pagination looped on this and hung every TTS line.
        page = [{'_id': f'a{i}', 'title': f'vl_a{i}'} for i in range(100)]
        calls = []

        def fake_get(url, headers=None, params=None, timeout=None):
            calls.append(dict(params))
            return Resp({'items': page, 'total': 101, 'has_more': True})

        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish.requests, 'get', side_effect=fake_get):
            got = fish.list_voice_models()

        self.assertEqual(len(got), 100)
        self.assertEqual(len(calls), 2, 'must stop once a page brings nothing new')

    def test_second_page_of_real_models_is_added(self):
        page1 = [{'_id': f'a{i}', 'title': f'vl_a{i}'} for i in range(100)]
        page2 = [{'_id': f'b{i}', 'title': f'vl_b{i}'} for i in range(1)]
        responses = iter([
            {'items': page1, 'total': 101, 'has_more': True},
            {'items': page2, 'total': 101, 'has_more': True},
        ])

        def fake_get(url, headers=None, params=None, timeout=None):
            return Resp(next(responses))

        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish.requests, 'get', side_effect=fake_get):
            got = fish.list_voice_models()

        self.assertEqual(len(got), 101)

    def test_result_is_cached_between_calls(self):
        calls = []

        def fake_get(url, headers=None, params=None, timeout=None):
            calls.append(1)
            return Resp({'items': [{'_id': 'a', 'title': 'vl_a'}], 'total': 1, 'has_more': False})

        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish.requests, 'get', side_effect=fake_get):
            fish.list_voice_models()
            fish.list_voice_models()
            fish.list_voice_models()

        self.assertEqual(len(calls), 1, 'voice list should be cached, not refetched per line')


if __name__ == "__main__":
    unittest.main(verbosity=2)
