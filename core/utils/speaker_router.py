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

Picker shape (input to this module) — see ``core._3_speaker_preview.confirm_picks``::

    speaker_voice_map = {
        "<sid>": {"mode": "fixed",   "voice":   "<tts_voice_name>"},
        "<sid>": {"mode": "clone",   "ref_wav": "<abs path>"},
        "<sid>": {"mode": "default"},
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich import print as rprint

from core.utils.config_utils import load_key


# Modes accepted in speaker_voice_map entries. Kept in sync with
# core/st_utils/speaker_picker.py constants.
_MODE_FIXED = "fixed"
_MODE_CLONE = "clone"
_MODE_DEFAULT = "default"

# Backend used whenever a speaker is configured for voice cloning. Matches
# the plan (C4 step in plan_multispeaker): clone -> gpt_sovits + spk{i}.wav.
_CLONE_BACKEND = "gpt_sovits"


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


def resolve_voice_cfg(speaker_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve effective TTS config for *speaker_id*; see module docstring."""
    if not _multi_enabled():
        return None
    if not speaker_id:
        return None

    entry = _voice_map().get(speaker_id)
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

    if mode == _MODE_CLONE:
        ref = entry.get("ref_wav")
        if not ref or not Path(ref).is_file():
            rprint(
                f"[yellow]🎤 speaker_router: speaker '{speaker_id}' clone "
                f"ref missing ({ref!r}); falling back to global voice.[/yellow]"
            )
            return None
        return {
            "method": _CLONE_BACKEND,
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


__all__ = ["resolve_voice_cfg"]
