from pathlib import Path
import hashlib
import io
import threading
import time
import requests
from pydub import AudioSegment
from core.utils import load_key, except_handler, load_timeout, rprint

# Soniox Text-to-Speech backend.
# Docs: https://soniox.com/docs/tts/rest-api/generate-speech
#
# Single-shot REST flow: the response body IS the raw audio, so there is no
# JSON envelope and no second download hop like Qwen3/DashScope needs.
#
# The Soniox key already configured for ASR (whisper.soniox_api_key) works here
# too, so soniox_tts.api_key is optional and falls back to it.
_API_URL = "https://tts-rt.soniox.com/tts"
_VOICES_API_URL = "https://api.soniox.com/v1/voices"
_DEFAULT_MODEL = "tts-rt-v1"

# Voice cloning. Processing rejects reference clips over 20s with
# voice_audio_too_long, and the project's shared _long_ref.wav targets 22s, so
# trimming is the normal path rather than an edge case.
CLONE_REF_MAX_SECONDS = 18.0
CLONE_READY_TIMEOUT = 90
CLONE_POLL_INTERVAL = 2.0

# Cloned voices are created once and reused on later runs, keyed by the content
# hash of the reference clip. The lock stops concurrent TTS workers racing to
# create the same voice — names are unique per project, so the loser would 400.
_clone_cache = {}
_clone_lock = threading.Lock()

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


def _voice_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def list_voices(api_key=None) -> list:
    """Every custom voice in the project."""
    api_key = api_key or _load_api_key()
    resp = requests.get(_VOICES_API_URL, headers=_voice_headers(api_key),
                        timeout=load_timeout("tts", 60))
    if resp.status_code != 200:
        raise Exception(f"Soniox voice list failed {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("voices") or []


def delete_voice(voice_id, api_key=None) -> None:
    """Free one of the 20 voice slots an organisation gets."""
    api_key = api_key or _load_api_key()
    resp = requests.delete(f"{_VOICES_API_URL}/{voice_id}",
                           headers=_voice_headers(api_key),
                           timeout=load_timeout("tts", 60))
    if resp.status_code not in (200, 204):
        raise Exception(f"Soniox voice delete failed {resp.status_code}: {resp.text[:300]}")


def _prepare_reference_clip(ref_wav: Path) -> bytes:
    """Trim a reference clip to what Soniox accepts and return wav bytes."""
    seg = AudioSegment.from_file(ref_wav)
    limit_ms = int(CLONE_REF_MAX_SECONDS * 1000)
    if len(seg) > limit_ms:
        rprint(
            f"[yellow]Reference clip is {len(seg) / 1000:.1f}s; trimming to "
            f"{CLONE_REF_MAX_SECONDS:.0f}s for Soniox voice cloning[/yellow]"
        )
        seg = seg[:limit_ms]
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def _create_voice(name: str, clip: bytes, api_key: str) -> str:
    resp = requests.post(
        _VOICES_API_URL,
        headers=_voice_headers(api_key),
        data={"name": name},
        files={"file": (f"{name}.wav", clip, "audio/wav")},
        timeout=load_timeout("tts", 120),
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"Soniox voice create failed {resp.status_code}: {resp.text[:400]}")
    return resp.json()["id"]


def _wait_until_ready(voice_id: str, model: str, api_key: str) -> None:
    """Uploading only queues the clip; processing is async but usually seconds."""
    deadline = time.time() + CLONE_READY_TIMEOUT
    while True:
        for voice in list_voices(api_key):
            if voice.get("id") != voice_id:
                continue
            for entry in voice.get("models") or []:
                if entry.get("model") != model:
                    continue
                status = entry.get("status")
                if status == "ready":
                    return
                # A failed voice is terminal; recompute cannot recover it.
                if status == "failed":
                    raise Exception(
                        f"Soniox voice {voice_id} failed to process for {model}: "
                        f"{entry.get('error_message')}"
                    )
            break
        if time.time() >= deadline:
            raise Exception(
                f"Soniox voice {voice_id} still not ready for {model} after "
                f"{CLONE_READY_TIMEOUT}s"
            )
        time.sleep(CLONE_POLL_INTERVAL)


def ensure_cloned_voice(ref_wav, model=None) -> str:
    """Return the voice ID for a reference clip, creating it on first use.

    Voices are named after the clip's content hash, so a repeat run — or another
    speaker sharing the same reference — reuses the existing voice rather than
    burning one of the 20 slots an organisation gets.

    Clone quality tracks the reference closely: the model copies speaking speed,
    accent, breathing and any background noise, so a clean single-speaker clip
    matters more than a long one.
    """
    ref_path = Path(ref_wav)
    if not ref_path.exists():
        raise ValueError(f"Soniox voice clone: reference audio not found: {ref_path}")
    model = model or (_load_opt("soniox_tts.model") or _DEFAULT_MODEL)

    # Cheap in-process key. This runs once per generated line, so hashing the
    # clip every time would mean decoding the reference a few hundred times per
    # video just to look up an ID we already have.
    stat = ref_path.stat()
    cache_key = (str(ref_path.resolve()), stat.st_mtime_ns, stat.st_size, model)

    with _clone_lock:
        cached = _clone_cache.get(cache_key)
        if cached:
            return cached

        clip = _prepare_reference_clip(ref_path)
        # The content hash names the voice, so a repeat run — or another speaker
        # sharing the reference — reuses it rather than burning a slot.
        name = f"vl_{hashlib.md5(clip).hexdigest()[:12]}"

        api_key = _load_api_key()
        for voice in list_voices(api_key):
            if voice.get("name") == name:
                _wait_until_ready(voice["id"], model, api_key)
                _clone_cache[cache_key] = voice["id"]
                rprint(f"[cyan]Reusing Soniox cloned voice '{name}'[/cyan]")
                return voice["id"]

        rprint(f"[cyan]Creating Soniox cloned voice '{name}' from {ref_path.name}[/cyan]")
        voice_id = _create_voice(name, clip, api_key)
        _wait_until_ready(voice_id, model, api_key)
        _clone_cache[cache_key] = voice_id
        return voice_id


def _resolve_voice(voice_cfg):
    """Pick the voice for one call: router override, cloned voice, or config."""
    # The multi-speaker router wins, and its clone mode carries its own reference.
    if voice_cfg:
        if voice_cfg.get("is_clone") and voice_cfg.get("ref_wav"):
            return ensure_cloned_voice(voice_cfg["ref_wav"])
        if voice_cfg.get("voice"):
            return voice_cfg["voice"]

    if str(_load_opt("soniox_tts.mode", "preset")).strip().lower() == "clone":
        # Single-voice clone mode reuses the merged reference the pipeline already
        # builds for other cloning backends.
        from core.utils._long_ref_extractor import ensure_long_ref
        return ensure_cloned_voice(ensure_long_ref())

    return load_key("soniox_tts.voice")


@except_handler("Failed to generate audio using Soniox TTS", retry=3, delay=1)
def soniox_tts(text, save_as, voice_cfg=None, speed=None):
    """Soniox Text-to-Speech (REST, single request/response).

    voice_cfg: optional dict from the C4 speaker router. When provided and
    voice_cfg["voice"] is truthy, it overrides the global soniox_tts.voice for
    this call. A UUID selects a cloned voice; anything else is a built-in name.
    A router entry in clone mode instead clones its own reference clip.

    speed: optional per-call override of soniox_tts.speed. The timeline fitter
    in _10_gen_audio.py uses it to render a line at the rate it needs instead of
    stretching the result with ffmpeg atempo.
    """
    api_key = _load_api_key()
    voice = _resolve_voice(voice_cfg)

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
