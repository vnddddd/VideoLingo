import json
import os
import threading
from copy import deepcopy
from functools import lru_cache

DISPLAY_LANGUAGES = {
    "🇬🇧 English": "en",
    "🇨🇳 简体中文": "zh-CN",
    "🇭🇰 繁体中文": "zh-HK",
    "🇯🇵 日本語": "ja",
    "🇪🇸 Español": "es",
    "🇷🇺 Русский": "ru",
    "🇫🇷 Français": "fr",
}

# Load the language file based on user selection
_translations_lock = threading.Lock()


@lru_cache(maxsize=16)
def _read_translations(path, signature):
    """Keep only a bounded number of language-file versions in memory."""
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _get_translations(language):
    path = os.path.abspath(f'translations/{language}.json')
    with _translations_lock:
        stat = os.stat(path)
        signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
        return _read_translations(path, signature)


# Keep the public loader's independent, mutable return value.
def load_translations(language="en"):
    return deepcopy(_get_translations(language))

# Function to fetch the translation
def translate(key):
    from core.utils.config_utils import load_key
    try:
        display_language = load_key("display_language")
        translations = _get_translations(display_language)
        translation = translations.get(key)
        if translation is None:
            print(f"Warning: Translation not found for key '{key}' in language '{display_language}'")
            return key
        return deepcopy(translation)
    except:
        return key
