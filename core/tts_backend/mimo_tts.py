"""Xiaomi MiMo TTS backend for VideoLingo.

Supports 3 models with the same OpenAI-compatible chat.completions endpoint:
  - mimo-v2.5-tts             : preset voice (9 voices, e.g. Chloe / 冰糖)
  - mimo-v2.5-tts-voicedesign : natural-language voice prompt
  - mimo-v2.5-tts-voiceclone  : reference-audio (dataURL) voice cloning

Endpoint default: https://token-plan-sgp.xiaomimimo.com/v1   (subscription cluster)
Auth: Authorization: Bearer <api_key>   (standard OpenAI protocol)

Response audio path: choices[0].message.audio.data  (base64 wav, 24kHz mono 16bit)

Design notes
------------
* ``_call_mimo_api()`` is a config-free low-level function — useful for unit
  tests that do not depend on VideoLingo's ``load_key``/config.yaml.
* ``mimo_tts_for_videolingo()`` is the production entry point that matches
  the signature of ``sf_cosyvoice2.cosyvoice_tts_for_videolingo`` so
  ``tts_main`` can route to it uniformly.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from core.utils import load_key, except_handler, load_timeout


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"

# Built-in preset voices for mimo-v2.5-tts (model A).
# Source: live API response (2026-05-18). The earlier docs list
# (Sophia/Hannah/Jacob/Owen/Ethan/可乐) is obsolete — those voices
# have been retired server-side; mimo_default/苏打/白桦/Mia/Milo/Dean
# are the new additions.
PRESET_VOICES = [
    "mimo_default",
    # Chinese
    "冰糖", "茉莉", "苏打", "白桦",
    # English female
    "Mia", "Chloe",
    # English male
    "Milo", "Dean",
]

SUPPORTED_MODELS = (
    "mimo-v2.5-tts",
    "mimo-v2.5-tts-voicedesign",
    "mimo-v2.5-tts-voiceclone",
)


# ---------------------------------------------------------------------------
# Helpers (config-free)
# ---------------------------------------------------------------------------

def _wav_to_dataurl(wav_path: Path) -> str:
    """Read a wav file and return ``data:audio/wav;base64,...`` dataURL."""
    with open(wav_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _build_payload(
    text: str,
    model: str,
    *,
    voice: str | None = None,
    voice_description: str | None = None,
    voice_ref_dataurl: str | None = None,
) -> dict:
    """Construct a chat.completions request payload for a MiMo TTS model.

    The three models differ only in messages / audio fields:
      - mimo-v2.5-tts            : audio.voice = preset name
      - mimo-v2.5-tts-voicedesign: messages[user]=description, audio.optimize_text_preview=False
      - mimo-v2.5-tts-voiceclone : audio.voice = data:audio/wav;base64,...
    """
    if model == "mimo-v2.5-tts":
        return {
            "model": model,
            "messages": [{"role": "assistant", "content": text}],
            "audio": {"format": "wav", "voice": voice or "Chloe"},
        }
    if model == "mimo-v2.5-tts-voicedesign":
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": voice_description or "A natural, neutral voice."},
                {"role": "assistant", "content": text},
            ],
            # IMPORTANT: VL needs the literal text to keep subtitle alignment.
            # Letting the model auto-rewrite the text would break sub timing.
            "audio": {"format": "wav", "optimize_text_preview": False},
        }
    if model == "mimo-v2.5-tts-voiceclone":
        if not voice_ref_dataurl:
            raise ValueError("voiceclone model requires `voice_ref_dataurl` (data:audio/wav;base64,...)")
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": text},
            ],
            "audio": {"format": "wav", "voice": voice_ref_dataurl},
        }
    raise ValueError(f"Unknown MiMo TTS model: {model!r}. Supported: {SUPPORTED_MODELS}")


def _normalize_base_url(base_url: str) -> str:
    """Normalize MiMo OpenAI-compatible base URL.

    MiMo's docs use endpoints like ``https://token-plan-sgp.xiaomimimo.com/v1``.
    Users often configure only the host (``https://token-plan-sgp.xiaomimimo.com``),
    which would otherwise post to ``/chat/completions`` and return openresty 404.
    """
    if not base_url:
        base_url = DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1") or "/v1/" in base_url:
        return base_url
    return base_url + "/v1"


def _call_mimo_api(
    text: str,
    save_as: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    timeout: int = 60,
    **kwargs,
) -> int:
    """Low-level API caller. Returns number of bytes written.

    This function does NOT depend on ``load_key``/config.yaml — it can be
    invoked directly in unit tests by passing api_key/base_url/model
    explicitly. ``mimo_tts_for_videolingo`` is a thin wrapper that pulls
    these from config.
    """
    if not api_key:
        raise ValueError("api_key is empty")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model!r}; expected one of {SUPPORTED_MODELS}")

    payload = _build_payload(text, model, **kwargs)
    url = _normalize_base_url(base_url) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    # Don't blindly raise_for_status — read body for diagnostics.
    if resp.status_code != 200:
        snippet = resp.text[:500] if resp.text else "<empty body>"
        raise RuntimeError(f"MiMo TTS HTTP {resp.status_code}: {snippet}")

    try:
        data = resp.json()
    except ValueError as e:
        raise RuntimeError(f"MiMo TTS returned non-JSON: {resp.text[:300]}") from e

    if "error" in data:
        raise RuntimeError(f"MiMo TTS API error: {json.dumps(data['error'], ensure_ascii=False)}")

    try:
        audio_b64 = data["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected MiMo TTS response shape: {json.dumps(data, ensure_ascii=False)[:400]}"
        ) from e

    wav_bytes = base64.b64decode(audio_b64)
    save_path = Path(save_as)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(wav_bytes)
    return len(wav_bytes)


# ---------------------------------------------------------------------------
# Config-bound helpers
# ---------------------------------------------------------------------------

def _safe_load_key(key: str, default=None):
    """Return ``load_key(key)`` or ``default`` if the key path is missing."""
    try:
        v = load_key(key)
        return v if v not in (None, "") else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# VideoLingo entry point
# ---------------------------------------------------------------------------

@except_handler("Failed to generate audio using Xiaomi MiMo TTS")
def mimo_tts_for_videolingo(text, save_as, number=0, task_df=None):
    """Production entry. Signature aligned with sf_cosyvoice2 family for
    ``tts_main`` dispatch.

    Behaviour depends on ``mimo_tts.model`` in config.yaml:
      * "mimo-v2.5-tts"            → uses ``mimo_tts.voice`` (preset name)
      * "mimo-v2.5-tts-voicedesign"→ uses ``mimo_tts.voice_description``
      * "mimo-v2.5-tts-voiceclone" → auto-loads output/audio/refers/{number}.wav
        (re-using the reference audio already extracted by VL step _9_refer_audio)

    ``task_df`` is accepted for signature compat but unused (MiMo voiceclone,
    unlike SiliconFlow CosyVoice2, does NOT require a reference transcript).
    """
    api_key = load_key("mimo_tts.api_key")
    if not api_key or api_key == "YOUR_MIMO_API_KEY":
        raise ValueError(
            "mimo_tts.api_key is not set in config.yaml. "
            "Get a token from https://platform.xiaomimimo.com/ (subscription)."
        )

    base_url = _safe_load_key("mimo_tts.base_url", DEFAULT_BASE_URL)
    model = _safe_load_key("mimo_tts.model", "mimo-v2.5-tts")

    kwargs = {}
    if model == "mimo-v2.5-tts":
        kwargs["voice"] = _safe_load_key("mimo_tts.voice", "Chloe")
    elif model == "mimo-v2.5-tts-voicedesign":
        kwargs["voice_description"] = _safe_load_key(
            "mimo_tts.voice_description", "A natural, neutral voice."
        )
    elif model == "mimo-v2.5-tts-voiceclone":
        # Reuse the reference audio that VL pipeline already extracted in
        # step _9_refer_audio. Same fallback chain as sf_cosyvoice2.
        current_dir = Path.cwd()
        ref_audio_path = current_dir / f"output/audio/refers/{number}.wav"
        if not ref_audio_path.exists():
            ref_audio_path = current_dir / "output/audio/refers/1.wav"
            if not ref_audio_path.exists():
                try:
                    from core._9_refer_audio import extract_refer_audio_main
                    print(f"参考音频文件不存在，尝试提取: {ref_audio_path}")
                    extract_refer_audio_main()
                except Exception as e:
                    print(f"提取参考音频失败: {e}")
                    raise
        kwargs["voice_ref_dataurl"] = _wav_to_dataurl(ref_audio_path)
    else:
        raise ValueError(
            f"mimo_tts.model={model!r} is not supported. "
            f"Choose one of: {SUPPORTED_MODELS}"
        )

    n_bytes = _call_mimo_api(
        text,
        save_as,
        api_key,
        base_url,
        model,
        timeout=load_timeout("tts", 60),
        **kwargs,
    )
    print(f"MiMo TTS [{model}] audio saved to {save_as} ({n_bytes} bytes)")
    return True


if __name__ == "__main__":
    # Smoke test — requires mimo_tts.api_key to be set in config.yaml
    out = "./mimo_smoke_output.wav"
    mimo_tts_for_videolingo("你好，这是小米 MiMo TTS 的烟雾测试。", out)
    print(f"OK: {out}")
