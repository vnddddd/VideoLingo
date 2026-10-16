"""Multi-speaker voice router.

Resolves the **effective TTS voice config** for a given speaker_id by
consulting `config.yaml` keys `multi_speaker_enabled` and `speaker_voice_map`
(both populated by the Streamlit picker via `core._3_speaker_preview`).

Return-value contract
---------------------
`resolve_voice_cfg(speaker_id)` returns one of:

* ``None``
    Caller should fall back to the **legacy global path**:
    use ``load_key('tts_method')`` + that backend's own global voice field.
    This is the case when:
      - multi-speaker mode is off, OR
      - speaker_id is empty / unknown to the map, OR
      - the picker stored ``mode == 'default'`` for this speaker, OR
      - the entry exists but is incomplete (missing voice / ref_wav).

* ``dict`` with shape::

    {
        "method":   <tts_method to dispatch to, e.g. 'edge_tts' / 'gpt_sovits'>,
        "voice":    <voice id string for that backend>  | None,
        "ref_wav":  <abs path to reference clip>        | None,
        "is_clone": <bool>,
    }

Backends accept this dict as ``voice_cfg=`` kwarg; when present, they should
honour ``voice_cfg['voice']`` (overrides their global voice config) and,
for clone-capable backends, ``voice_cfg['ref_wav']``.

Global clone shortcut
----------------------
``global_clone_model()`` reports the model the *global* dubbing settings
would use when cloning every speaker. When it returns a model, the picker has
nothing left to ask: core/_3_speaker_preview fills speaker_voice_map with a
clone entry per detected speaker and the pipeline runs end-to-end from a
single "translate and dub" click. Only an explicit *preset* choice (Fish,
SiliconFlow or Soniox preset mode, or a MiMo non-clone model) keeps the
picker, because that is the one global setting that says "do not clone".

Picker shape (input to this module) — see ``core._3_speaker_preview.confirm_picks``::

    speaker_voice_map = {
        "<sid>": {"mode": "fixed",            "voice": "<tts_voice_name>"},
        "<sid>": {"mode": "qwen3_tts",        "voice": "<qwen3_voice_name>"},
        "<sid>": {"mode": "mimo_voicedesign", "voice_description": "<prompt>"},
        "<sid>": {"mode": "clone",            "ref_wav": "<abs path>"},
        "<sid>": {"mode": "default"},
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import math
import re

from rich import print as rprint

from core.utils.config_utils import load_key


# Modes accepted in speaker_voice_map entries. Kept in sync with
# core/st_utils/speaker_picker.py constants.
_MODE_FIXED = "fixed"
_MODE_QWEN3_TTS = "qwen3_tts"
_MODE_MIMO_VOICE_DESIGN = "mimo_voicedesign"
_MODE_CLONE = "clone"
_MODE_DEFAULT = "default"

# These backends consume the per-speaker reference instead of a global voice.
# The selected backend retains control of its model and cloning settings.
_CLONE_BACKENDS = {
    "fish_audio_tts", "soniox_tts", "gpt_sovits", "sf_fish_tts",
    "sf_cosyvoice2", "f5tts", "mimo_tts", "indextts2",
}
_QWEN3_BACKEND = "qwen3_tts"
_MIMO_BACKEND = "mimo_tts"
_MIMO_VOICE_DESIGN_MODEL = "mimo-v2.5-tts-voicedesign"
_MIMO_CLONE_MODEL = "mimo-v2.5-tts-voiceclone"

# Global switches that mean: clone every speaker with this model.
# `preset` is the only setting that explicitly asks for a library voice
# instead of a clone, so it keeps the per-speaker picker. Backends with no
# preset concept (gpt_sovits, sf_cosyvoice2, f5tts, indextts2) always dub
# from a reference, so cloning is implied by the engine choice itself.
_CLONE_MODES = {"custom", "dynamic"}
# Fish's `s1` is paid-only and does not support multi-speaker tagging, so it
# cannot back the automatic per-speaker clone path (the sidebar labels it as
# such). Keep the picker for it instead of starting a run that would fail.
_FISH_SINGLE_SPEAKER_MODELS = {"s1"}
_FISH_MODE_KEY = "fish_audio_tts.mode"
_SF_FISH_MODE_KEY = "sf_fish_tts.mode"
_SONIOX_MODE_KEY = "soniox_tts.mode"
_MODEL_KEYS = {
    "fish_audio_tts": "fish_audio_tts.model",
    "soniox_tts": "soniox_tts.model",
    "mimo_tts": "mimo_tts.model",
}


class UnsupportedCloneBackendError(ValueError):
    """The selected global backend cannot clone a speaker's reference."""


def _safe_load(key: str, default: Any = None) -> Any:
    """`load_key` that returns *default* instead of raising on missing keys."""
    try:
        return load_key(key)
    except KeyError:
        return default


def _multi_enabled() -> bool:
    return bool(_safe_load("multi_speaker_enabled", False))


def _voice_map() -> Dict[str, Dict]:
    raw = _safe_load("speaker_voice_map", None)
    return raw if isinstance(raw, dict) else {}


def _normalise_speaker_id(value: Any) -> Optional[str]:
    """Return the canonical string id used by speaker_voice_map.

    The speaker id travels through JSON and Excel/Pandas before TTS.  Excel
    commonly round-trips ids like "1" as numeric 1/1.0, while YAML config keys
    are stored as strings.  Canonicalising here keeps the router as the single
    source of truth and prevents silent fallback to the global/default voice.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return str(value).strip()

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None

    # Pandas/openpyxl may surface integral ids as strings such as "1.0".
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]

    return text


def _normalised_voice_map() -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    for raw_sid, entry in _voice_map().items():
        sid = _normalise_speaker_id(raw_sid)
        if sid is not None:
            result[sid] = entry
    return result


def global_clone_model() -> Optional[str]:
    """Return the global clone model, or ``None`` when cloning is not implied.

    The answer mirrors exactly what resolve_voice_cfg does for a speaker whose
    pick is mode == 'clone': the configured tts_method is used with that
    speaker's own reference clip. So whenever this returns a value, every
    detected speaker can be cloned without asking the user anything.

    Returns a display name: the model id, or the backend name for backends
    that have no separate model field. Only truthiness and the log line matter.
    """
    method = _safe_load("tts_method", None)
    if method not in _CLONE_BACKENDS:
        return None

    def _mode(key: str, default: str) -> str:
        return str(_safe_load(key, default) or default).strip().lower()

    if method == "fish_audio_tts":
        if _mode(_FISH_MODE_KEY, "custom") not in _CLONE_MODES:
            return None
        if str(_safe_load("fish_audio_tts.model", "") or "").strip() in _FISH_SINGLE_SPEAKER_MODELS:
            return None
    elif method == "sf_fish_tts":
        if _mode(_SF_FISH_MODE_KEY, "preset") not in _CLONE_MODES:
            return None
    elif method == "soniox_tts":
        if _mode(_SONIOX_MODE_KEY, "preset") != _MODE_CLONE:
            return None
    elif method == "mimo_tts":
        if str(_safe_load("mimo_tts.model", "") or "").strip() != _MIMO_CLONE_MODEL:
            return None

    model_key = _MODEL_KEYS.get(method)
    model = str(_safe_load(model_key, "") or "").strip() if model_key else ""
    return model or method


def resolve_voice_cfg(speaker_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve effective TTS config for *speaker_id*; see module docstring."""
    if not _multi_enabled():
        return None

    sid = _normalise_speaker_id(speaker_id)
    if not sid:
        return None

    entry = _normalised_voice_map().get(sid)
    if not isinstance(entry, dict):
        return None

    mode = entry.get("mode", _MODE_DEFAULT)

    if mode == _MODE_DEFAULT:
        return None

    if mode == _MODE_FIXED:
        voice = (entry.get("voice") or "").strip()
        if not voice:
            rprint(
                f"[yellow]🎤 speaker_router: speaker '{speaker_id}' is "
                f"'fixed' but voice is empty; falling back to global voice.[/yellow]"
            )
            return None
        method = _safe_load("tts_method", None)
        if not method:
            return None
        return {
            "method": method,
            "voice": voice,
            "ref_wav": None,
            "is_clone": False,
        }

    if mode == _MODE_QWEN3_TTS:
        voice = (entry.get("voice") or "").strip()
        if not voice:
            rprint(
                f"[yellow]🎤 speaker_router: speaker '{speaker_id}' is "
                f"'qwen3_tts' but voice is empty; falling back to global voice.[/yellow]"
            )
            return None
        return {
            "method": _QWEN3_BACKEND,
            "voice": voice,
            "ref_wav": None,
            "is_clone": False,
        }

    if mode == _MODE_MIMO_VOICE_DESIGN:
        voice_description = (entry.get("voice_description") or "").strip()
        if not voice_description:
            rprint(
                f"[yellow]🎤 speaker_router: speaker '{speaker_id}' is "
                f"'mimo_voicedesign' but voice_description is empty; "
                f"falling back to global voice.[/yellow]"
            )
            return None
        return {
            "method": _MIMO_BACKEND,
            "voice": None,
            "ref_wav": None,
            "is_clone": False,
            "model": _MIMO_VOICE_DESIGN_MODEL,
            "voice_description": voice_description,
        }

    if mode == _MODE_CLONE:
        method = _safe_load("tts_method", None)
        if method not in _CLONE_BACKENDS:
            raise UnsupportedCloneBackendError(
                f"当前配音引擎 {method!r} 不支持按说话人克隆。"
                "请在配音设置中选择支持克隆的引擎（如 Fish Audio 或 Soniox），"
                "或更改该说话人的配音策略。"
            )
        ref = entry.get("ref_wav")
        if not ref or not Path(ref).is_file():
            rprint(
                f"[yellow]🎤 speaker_router: speaker '{speaker_id}' clone "
                f"ref missing ({ref!r}); falling back to global voice.[/yellow]"
            )
            return None
        return {
            "method": method,
            "voice": None,
            "ref_wav": str(ref),
            "is_clone": True,
        }

    # Unknown mode → safe fallback.
    rprint(
        f"[yellow]🎤 speaker_router: speaker '{speaker_id}' has unknown "
        f"mode {mode!r}; falling back to global voice.[/yellow]"
    )
    return None


__all__ = ["resolve_voice_cfg", "global_clone_model", "UnsupportedCloneBackendError"]
