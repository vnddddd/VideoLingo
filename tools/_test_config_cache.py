"""Run with .venv/Scripts/python -X utf8 tools/_test_config_cache.py.

Only temporary configuration/translation files are changed; no API calls.
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.utils import config_utils as config
from translations import translations as translation


CONFIG = '''# Keep this comment
display_language: en
backend:
  model: "old"  # Keep these quotes
  values: &defaults [alpha, beta]
alias: *defaults
items:
  - name: first
'''


class TemporaryFilesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="videolingo-config-cache-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config_path = self.directory / "config.yaml"
        self.example_path = self.directory / "config.example.yaml"
        self.config_path.write_text(CONFIG, encoding="utf-8")
        self.example_path.write_text(CONFIG, encoding="utf-8")
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.directory)
        parser = YAML()
        parser.preserve_quotes = True
        for name, value in (
            ("CONFIG_PATH", str(self.config_path)),
            ("CONFIG_EXAMPLE_PATH", str(self.example_path)),
            ("_config_cache", None),
            ("yaml", parser),
        ):
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        translation._read_translations.cache_clear()
        self.addCleanup(translation._read_translations.cache_clear)
        self.language_dir = self.directory / "translations"
        self.language_dir.mkdir()
        self.write_language("en", {"hello": "Hello", "payload": {"items": [1]}})
        self.write_language("zh-CN", {"hello": "你好"})

    def write_language(self, language, data):
        path = self.language_dir / f"{language}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def change_same_size_file(self, path, old, new):
        previous = path.stat()
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
        self.assertEqual(path.stat().st_size, previous.st_size)
        # Don't depend on the filesystem clock resolution or use sleeps.
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))


class ConfigCacheTests(TemporaryFilesTest):
    def test_repeated_reads_parse_once(self):
        with patch.object(config.yaml, "load", wraps=config.yaml.load) as parse:
            for _ in range(100):
                self.assertEqual(config.load_key("backend.model"), "old")
                self.assertEqual(config.load_key("backend.values"), ["alpha", "beta"])
        self.assertEqual(parse.call_count, 1)

    def test_concurrent_cold_reads_share_one_parse(self):
        barrier = threading.Barrier(8)

        def read(_):
            barrier.wait(timeout=5)
            for _ in range(50):
                self.assertEqual(config.load_key("backend.model"), "old")
                config.load_key("items")[0]["name"] = "private mutation"

        with patch.object(config.yaml, "load", wraps=config.yaml.load) as parse:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(read, range(8)))
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(config.load_key("items"), [{"name": "first"}])

    def test_returned_nested_values_and_aliases_are_independent(self):
        backend = config.load_key("backend")
        backend["model"] = "changed"
        backend["values"].append("changed")
        config.load_key("alias").clear()
        self.assertEqual(config.load_key("backend.model"), "old")
        self.assertEqual(config.load_key("backend.values"), ["alpha", "beta"])
        self.assertEqual(config.load_key("alias"), ["alpha", "beta"])

    def test_update_is_immediate_and_preserves_yaml_formatting(self):
        config.load_key("backend")
        self.assertTrue(config.update_key("backend.model", "new"))
        self.assertIsNone(config._config_cache)
        self.assertEqual(config.load_key("backend.model"), "new")
        saved = self.config_path.read_text(encoding="utf-8")
        self.assertIn("# Keep this comment", saved)
        self.assertIn('model: "new"', saved)
        self.assertIn("# Keep these quotes", saved)
        self.assertIn("*defaults", saved)

    def test_external_same_size_edit_invalidates_cache(self):
        self.assertEqual(config.load_key("backend.model"), "old")
        self.change_same_size_file(self.config_path, '"old"', '"new"')
        self.assertEqual(config.load_key("backend.model"), "new")

    def test_direct_yaml_writer_using_shared_lock_is_detected(self):
        config.load_key("backend.model")
        # Matches the existing sidebar settings migrations' write path.
        with config.lock:
            with self.config_path.open(encoding="utf-8") as file:
                data = config.yaml.load(file)
            data["added_by_sidebar"] = True
            with self.config_path.open("w", encoding="utf-8") as file:
                config.yaml.dump(data, file)
        self.assertTrue(config.load_key("added_by_sidebar"))

    def test_missing_keys_keep_original_error_behavior(self):
        before = self.config_path.read_bytes()
        with self.assertRaisesRegex(KeyError, "Key 'missing' not found"):
            config.load_key("backend.missing")
        with self.assertRaisesRegex(KeyError, "Key 'child' not found"):
            config.load_key("backend.model.child")
        self.assertFalse(config.update_key("missing.child", 1))
        with self.assertRaisesRegex(KeyError, "Key 'missing' not found"):
            config.update_key("backend.missing", 1)
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_invalid_yaml_is_not_replaced_by_cached_values(self):
        config.load_key("backend.model")
        self.config_path.write_text("backend: [\n", encoding="utf-8")
        for _ in range(2):
            with self.assertRaises(YAMLError):
                config.load_key("backend.model")
        self.config_path.write_text(CONFIG.replace('"old"', '"repaired"'), encoding="utf-8")
        self.assertEqual(config.load_key("backend.model"), "repaired")

    def test_missing_config_is_recreated_from_example(self):
        config.load_key("backend.model")
        self.example_path.write_text(CONFIG.replace('"old"', '"example"'), encoding="utf-8")
        self.config_path.unlink()
        self.assertEqual(config.load_key("backend.model"), "example")
        self.assertEqual(self.config_path.read_bytes(), self.example_path.read_bytes())

    def test_missing_config_and_example_still_raise(self):
        config.load_key("backend.model")
        self.config_path.unlink()
        self.example_path.unlink()
        with self.assertRaises(FileNotFoundError):
            config.load_key("backend.model")

    def test_changing_config_path_reloads(self):
        config.load_key("backend.model")
        alternate = self.directory / "alternate.yaml"
        alternate.write_text(CONFIG.replace('"old"', '"new"'), encoding="utf-8")
        with patch.object(config, "CONFIG_PATH", str(alternate)):
            self.assertEqual(config.load_key("backend.model"), "new")
        self.assertEqual(config.load_key("backend.model"), "old")

    def test_relative_path_is_scoped_to_working_directory(self):
        other = self.directory / "other"
        other.mkdir()
        (other / "config.yaml").write_text(CONFIG.replace('"old"', '"other"'), encoding="utf-8")
        with patch.object(config, "CONFIG_PATH", "config.yaml"):
            self.assertEqual(config.load_key("backend.model"), "old")
            os.chdir(other)
            self.assertEqual(config.load_key("backend.model"), "other")

    def test_concurrent_updates_and_reads_remain_consistent(self):
        barrier = threading.Barrier(5)

        def read(_):
            barrier.wait(timeout=5)
            for _ in range(100):
                self.assertIn(config.load_key("backend.model"), ("old", "new"))

        with ThreadPoolExecutor(max_workers=5) as pool:
            readers = [pool.submit(read, i) for i in range(4)]
            barrier.wait(timeout=5)
            for i in range(12):
                config.update_key("backend.model", "old" if i % 2 == 0 else "new")
            for reader in readers:
                reader.result(timeout=10)
        self.assertEqual(config.load_key("backend.model"), "new")

    def test_failed_write_invalidates_cached_snapshot(self):
        config.load_key("backend.model")
        with patch.object(config.yaml, "dump", side_effect=OSError("test write failure")):
            with self.assertRaises(OSError):
                config.update_key("backend.model", "new")
        self.assertIsNone(config._config_cache)
        with self.assertRaises(KeyError):
            config.load_key("backend.model")


class TranslationCacheTests(TemporaryFilesTest):
    def test_many_ui_labels_parse_each_file_once(self):
        with (
            patch.object(config.yaml, "load", wraps=config.yaml.load) as yaml_parse,
            patch.object(translation.json, "load", wraps=translation.json.load) as json_parse,
        ):
            for _ in range(100):
                self.assertEqual(translation.translate("hello"), "Hello")
        self.assertEqual(yaml_parse.call_count, 1)
        self.assertEqual(json_parse.call_count, 1)

    def test_concurrent_translation_readers_share_parses(self):
        barrier = threading.Barrier(8)

        def read(_):
            barrier.wait(timeout=5)
            for _ in range(30):
                self.assertEqual(translation.translate("hello"), "Hello")

        with patch.object(translation.json, "load", wraps=translation.json.load) as parse:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(read, range(8)))
        self.assertEqual(parse.call_count, 1)

    def test_language_switch_is_immediate(self):
        with patch.object(translation.json, "load", wraps=translation.json.load) as parse:
            self.assertEqual(translation.translate("hello"), "Hello")
            config.update_key("display_language", "zh-CN")
            self.assertEqual(translation.translate("hello"), "你好")
            config.update_key("display_language", "en")
            self.assertEqual(translation.translate("hello"), "Hello")
        self.assertEqual(parse.call_count, 2)

    def test_same_size_json_edit_is_detected(self):
        self.assertEqual(translation.translate("hello"), "Hello")
        self.change_same_size_file(self.language_dir / "en.json", "Hello", "World")
        self.assertEqual(translation.translate("hello"), "World")

    def test_mutating_public_results_does_not_change_cached_translations(self):
        data = translation.load_translations()
        data["hello"] = "changed"
        data["payload"]["items"].append(2)
        translation.translate("payload")["items"].append(3)
        self.assertEqual(translation.translate("hello"), "Hello")
        self.assertEqual(translation.translate("payload"), {"items": [1]})

    def test_invalid_or_missing_language_keeps_fallback(self):
        self.assertEqual(translation.translate("hello"), "Hello")
        path = self.language_dir / "en.json"
        path.write_text("{", encoding="utf-8")
        for _ in range(2):
            with self.assertRaises(json.JSONDecodeError):
                translation.load_translations()
            self.assertEqual(translation.translate("hello"), "hello")
        path.unlink()
        self.assertEqual(translation.translate("hello"), "hello")
        self.write_language("en", {"hello": "Repaired"})
        self.assertEqual(translation.translate("hello"), "Repaired")
        with patch("builtins.print"):
            self.assertEqual(translation.translate("missing"), "missing")

    def test_translation_path_is_scoped_to_working_directory(self):
        self.assertEqual(translation.translate("hello"), "Hello")
        other = self.directory / "other"
        (other / "translations").mkdir(parents=True)
        (other / "translations" / "en.json").write_text('{"hello": "Other"}', encoding="utf-8")
        os.chdir(other)
        self.assertEqual(translation.translate("hello"), "Other")

    def test_language_cache_has_a_fixed_bound(self):
        for i in range(24):
            self.write_language(f"test-{i}", {"hello": str(i)})
            self.assertEqual(translation.load_translations(f"test-{i}")["hello"], str(i))
        info = translation._read_translations.cache_info()
        self.assertEqual(info.maxsize, 16)
        self.assertLessEqual(info.currsize, 16)


class StreamlitGCConfigTests(unittest.TestCase):
    def test_project_disables_only_post_script_full_gc(self):
        result = subprocess.run(
            [sys.executable, "-c", "import gc, json; from streamlit import config; "
             "print(json.dumps([config.get_option('runner.postScriptGC'), gc.isenabled()]))"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=20,
        )
        self.assertEqual(json.loads(result.stdout), [False, True])
        self.assertTrue(gc.isenabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
