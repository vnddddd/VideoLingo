"""Regression checks for Fish Audio voice-clone identity.

Run: .venv/Scripts/python tools/_test_fish_voice_identity.py

The bug these tests pin down: voice models were titled by a per-video
"family" id plus a running index, so every speaker of every video collapsed
onto one family and reuse matched on title alone. A female speaker could be
handed a male speaker's voice from an unrelated, much older run.
"""

import hashlib
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

        # ensure_cloned_voice hashes canonical PCM, so stub the (bytes, digest)
        # pair: the digest is what must differ between speakers.
        with patch.object(fish, '_create_voice_model', side_effect=fake_create), \
             patch.object(fish, 'list_voice_models', side_effect=fake_list), \
             patch.object(fish, '_prepare_reference',
                          side_effect=lambda p: (Path(p).read_bytes(), hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16])):
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
             patch.object(fish, '_prepare_reference',
                          side_effect=lambda p: (Path(p).read_bytes(), hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16])):
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

    def test_recycle_refuses_to_delete(self):
        """The create-limit is a credit limit; deleting frees nothing."""
        deleted = []
        models = [{'_id': 'id-1', 'title': 'vl_old', 'created_at': '2026-01-01'}]
        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish, 'list_voice_models', return_value=models), \
             patch.object(fish, 'delete_voice_model', side_effect=lambda *a, **k: deleted.append(a)), \
             patch.object(fish, 'rprint', lambda *a, **k: None):
            result = fish.recycle_oldest_voice_model()
        self.assertFalse(result)
        self.assertEqual(deleted, [], 'recycle deleted a voice model')


class CreateLimitTests(unittest.TestCase):
    """A full allowance must fail loudly, never delete anything."""

    def test_create_limit_does_not_delete_and_explains_credit(self):
        import tempfile, io, wave
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        ref = Path(temp.name) / 'spk.wav'
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b'\0\0' * 16000)
        ref.write_bytes(buffer.getvalue())
        fish._clone_cache.clear()
        self.addCleanup(fish._clone_cache.clear)

        deleted, errors = [], []
        limit_error = Exception(
            'Fish Audio voice create failed 403: {"status":403,"message":'
            '"API model creation limit reached (102/0). Top up API credit to raise the limit."}',
        )

        with patch.object(fish, '_load_api_key', return_value='key'), \
             patch.object(fish, 'list_voice_models', return_value=[]), \
             patch.object(fish, '_prepare_reference_clip', return_value=b'clip'), \
             patch.object(fish, '_create_voice_model', side_effect=limit_error), \
             patch.object(fish, 'delete_voice_model', side_effect=lambda *a, **k: deleted.append(a)), \
             patch.object(fish, 'rprint', side_effect=lambda *a, **k: errors.append(' '.join(str(x) for x in a))):
            with self.assertRaises(Exception) as ctx:
                fish.ensure_cloned_voice(ref)

        self.assertEqual(deleted, [], 'a create failure deleted voice models')
        message = str(ctx.exception)
        self.assertIn('API credit', message, 'error must name the real cause')
        self.assertIn('No voice model was deleted', message)


class DynamicReferenceTests(unittest.TestCase):
    """Dynamic mode must clone each line from that line's own voice."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root, ignore_errors=True))
        self.refers = self.root / 'refers'
        self.refers.mkdir()
        for n in (1, 2, 3):
            (self.refers / f'{n}.wav').write_bytes(b'x' * 32)
        (self.refers / '_long_ref.wav').write_bytes(b'y' * 32)
        patcher = patch.object(fish, '_AUDIO_REFERS_DIR', str(self.refers))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_line_uses_its_own_clip(self):
        got = fish._dynamic_reference(None, 2)
        self.assertEqual(got.name, '2.wav', 'line 2 did not use its own reference')

    def test_router_override_wins(self):
        cfg = {'is_clone': True, 'ref_wav': str(self.root / 'spk_1.wav')}
        got = fish._dynamic_reference(cfg, 2)
        self.assertEqual(got.name, 'spk_1.wav', 'multi-speaker override was ignored')

    def test_missing_line_falls_back_to_long_ref(self):
        with patch('core.utils._long_ref_extractor.ensure_long_ref',
                   return_value=str(self.refers / '_long_ref.wav')):
            got = fish._dynamic_reference(None, 999)
        self.assertEqual(got.name, '_long_ref.wav')

    def test_empty_line_clip_falls_back(self):
        (self.refers / '4.wav').write_bytes(b'')
        with patch('core.utils._long_ref_extractor.ensure_long_ref',
                   return_value=str(self.refers / '_long_ref.wav')):
            got = fish._dynamic_reference(None, 4)
        self.assertEqual(got.name, '_long_ref.wav', 'a zero-byte clip must not be sent')


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
