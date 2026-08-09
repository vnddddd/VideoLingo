from pathlib import Path
import io
import requests
from pydub import AudioSegment
from core.utils import load_key, except_handler, load_timeout

# Soniox Text-to-Speech backend.
# Docs: https://soniox.com/docs/tts/rest-api/generate-speech
#
# Single-shot REST flow: the response body IS the raw audio, so there is no
# JSON envelope and no second download hop like Qwen3/DashScope needs.
#
# The Soniox key already configured for ASR (whisper.soniox_api_key) works here
# too, so soniox_tts.api_key is optional and falls back to it.
_API_URL = "https://tts-rt.soniox.com/tts"
_DEFAULT_MODEL = "tts-rt-v1"

# Native speaking-rate control. Soniox rejects anything outside this range with
# an `invalid_request` error, so clamp rather than let the request fail.
SPEED_MIN = 0.7
SPEED_MAX = 1.3

# Built-in voices work across all 60+ languages and keep a single identity when
# the language changes. Not enforced, so voices added later keep working; a UUID
# in `voice` is resolved as a cloned voice instead of a built-in name.
KNOWN_VOICES = [
    "Maya", "Daniel", "Noah", "Nina", "Emma", "Jack", "Adrian", "Claire",
    "Grace", "Owen", "Mina", "Kenji", "Rafael", "Mateo", "Lucia", "Sofia",
    "Oliver", "Arthur",
]

# Placeholders shipped in config.example.yaml; treat them as "not configured"
# so the fallback to the ASR key still kicks in.
_PLACEHOLDER_KEYS = {"", "YOUR_API_KEY", "your_soniox_api_key"}


def _load_opt(key, default=None):
    """load_key raises KeyError on missing keys; config blocks may be partial."""
    try:
        value = load_key(key)
    except KeyError:
        return default
    return default if value is None else value


def _load_api_key() -> str:
    """Prefer a dedicated TTS key, else reuse the ASR one (same Soniox account)."""
    for key in ("soniox_tts.api_key", "whisper.soniox_api_key"):
        value = str(_load_opt(key, "")).strip()
        if value and value not in _PLACEHOLDER_KEYS:
            return value
    raise ValueError(
        "Soniox TTS: no API key. Set soniox_tts.api_key, or whisper.soniox_api_key "
        "if you already use Soniox for ASR."
    )


def clamp_speed(speed):
    """Clamp to the rate Soniox accepts. Returns None when speed is unset."""
    if speed is None:
        return None
    return min(SPEED_MAX, max(SPEED_MIN, float(speed)))


def configured_speed() -> float:
    """Baseline speaking rate every clip is generated at (1.0 when unset)."""
    return clamp_speed(_load_opt("soniox_tts.speed", 1.0)) or 1.0


@except_handler("Failed to generate audio using Soniox TTS", retry=3, delay=1)
def soniox_tts(text, save_as, voice_cfg=None, speed=None):
    """Soniox Text-to-Speech (REST, single request/response).

    voice_cfg: optional dict from the C4 speaker router. When provided and
    voice_cfg["voice"] is truthy, it overrides the global soniox_tts.voice for
    this call. A UUID selects a cloned voice; anything else is a built-in name.

    speed: optional per-call override of soniox_tts.speed. The timeline fitter
    in _10_gen_audio.py uses it to re-synthesise a line at a faster rate instead
    of stretching the existing wav with ffmpeg atempo.
    """
    api_key = _load_api_key()

    if voice_cfg and voice_cfg.get("voice"):
        voice = voice_cfg["voice"]
    else:
        voice = load_key("soniox_tts.voice")

    payload = {
        "model": _load_opt("soniox_tts.model") or _DEFAULT_MODEL,
        "voice": voice,
        "audio_format": "wav",
        "text": text,
    }

    # Optional: any voice speaks any language, and Soniox detects it from the
    # text, so only pin `language` when the user asked for a specific one.
    language = _load_opt("soniox_tts.language")
    if language:
        payload["language"] = str(language).strip()

    speed = clamp_speed(configured_speed() if speed is None else speed)
    if speed is not None and abs(speed - 1.0) >= 0.001:
        payload["speed"] = round(speed, 3)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(_API_URL, headers=headers, json=payload,
                         timeout=load_timeout("tts", 60))
    if resp.status_code != 200:
        raise Exception(f"Soniox TTS API error {resp.status_code}: {resp.text[:500]}")

    # Errors raised before audio starts streaming come back as JSON instead of
    # audio bytes, with a 200 in some proxy setups; branch on content type.
    if resp.headers.get("Content-Type", "").startswith("application/json"):
        raise Exception(f"Soniox TTS error: {resp.text[:500]}")

    audio_bytes = resp.content
    if not audio_bytes:
        raise Exception("Soniox TTS returned an empty response body")

    # Re-encode through pydub: the pipeline probes every clip with the `wave`
    # module and rejects anything that is not plain PCM WAV.
    speech_file_path = Path(save_as)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    seg.export(str(speech_file_path), format="wav")
    print(f"Audio saved to {speech_file_path}")


if __name__ == "__main__":
    soniox_tts("你好，欢迎使用 VideoLingo！", "test_soniox.wav")
