import os
import json
import time
import tempfile
import requests
import librosa
import soundfile as sf
from rich import print as rprint
from core.utils import *
from core.asr_backend.soniox_format import soniox_to_whisper

# ----------------------------------------
# Constants (Soniox async transcription API)
# https://soniox.com/docs/stt/api-reference/
# ----------------------------------------

BASE_URL = "https://api.soniox.com"
MODEL = "stt-async-v4"
POLL_INTERVAL_SEC = 2.0
def _poll_timeout_sec():
    return load_timeout("asr_poll_total", 600)  # 10 min default upper bound for one chunk

# ----------------------------------------
# Low-level helpers (one helper per HTTP call)
# ----------------------------------------

def _auth_headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}

def _upload_file(api_key, audio_path):
    """POST /v1/files -> {id, ...}"""
    with open(audio_path, "rb") as fp:
        files = {"file": (os.path.basename(audio_path), fp, "audio/mpeg")}
        r = requests.post(
            f"{BASE_URL}/v1/files",
            headers=_auth_headers(api_key),
            files=files,
            timeout=load_timeout("asr_upload", 180),
        )
    r.raise_for_status()
    return r.json()["id"]

def _create_transcription(api_key, file_id, language=None, diarize=False):
    """POST /v1/transcriptions -> {id, status, ...}"""
    body = {"file_id": file_id, "model": MODEL}
    if language:
        body["language_hints"] = [language]
    if diarize:
        body["enable_speaker_diarization"] = True
    headers = _auth_headers(api_key)
    headers["Content-Type"] = "application/json"
    r = requests.post(
        f"{BASE_URL}/v1/transcriptions",
        headers=headers,
        json=body,
        timeout=load_timeout("asr_request", 30),
    )
    r.raise_for_status()
    return r.json()["id"]

def _poll_transcription(api_key, tx_id):
    """GET /v1/transcriptions/{id} until status in (completed, error). Returns final body on completed."""
    poll_timeout_sec = _poll_timeout_sec()
    deadline = time.time() + poll_timeout_sec
    while time.time() < deadline:
        r = requests.get(
            f"{BASE_URL}/v1/transcriptions/{tx_id}",
            headers=_auth_headers(api_key),
            timeout=load_timeout("asr_request", 30),
        )
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(
                f"Soniox transcription failed: "
                f"{data.get('error_message') or data.get('error') or data}"
            )
        # queued | processing -> keep waiting
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(f"Soniox transcription timeout after {poll_timeout_sec}s (tx_id={tx_id})")

def _fetch_transcript(api_key, tx_id):
    """GET /v1/transcriptions/{id}/transcript -> {text, tokens, ...}"""
    r = requests.get(
        f"{BASE_URL}/v1/transcriptions/{tx_id}/transcript",
        headers=_auth_headers(api_key),
        timeout=load_timeout("asr_fetch", 60),
    )
    r.raise_for_status()
    return r.json()

def _best_effort_delete(api_key, tx_id, file_id):
    """Cleanup remote resources; never raises."""
    headers = _auth_headers(api_key)
    if tx_id:
        try:
            requests.delete(f"{BASE_URL}/v1/transcriptions/{tx_id}", headers=headers, timeout=load_timeout("asr_cleanup", 30))
        except Exception:
            pass
    if file_id:
        try:
            requests.delete(f"{BASE_URL}/v1/files/{file_id}", headers=headers, timeout=load_timeout("asr_cleanup", 30))
        except Exception:
            pass

def _pick_detected_language(tokens):
    """Soniox per-token language is already ISO-639-1; pick most frequent non-empty value."""
    counts = {}
    for tk in tokens:
        lang = tk.get("language")
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]

# ----------------------------------------
# Public API (signature aligned with transcribe_audio_elevenlabs)
# ----------------------------------------

def transcribe_audio_soniox(raw_audio_path, vocal_audio_path, start=None, end=None):
    """
    Soniox async STT adapter for VideoLingo.
    - Slice [start, end] from vocal_audio_path (librosa+soundfile, mp3 16kHz)
    - Upload -> create transcription -> poll -> fetch transcript
    - Map Soniox tokens to whisper-style segments via soniox_to_whisper(time_offset=start)
    - Cache result at output/log/soniox_transcribe_{start}_{end}.json
    Returns: {"segments": [{"text", "start", "end", "speaker_id", "words":[{"word","start","end"}]}, ...]}
    """
    rprint(f"[cyan]🎤 Soniox ASR start, file: {vocal_audio_path}[/cyan]")

    LOG_FILE = f"output/log/soniox_transcribe_{start}_{end}.json"
    if os.path.exists(LOG_FILE):
        rprint(f"[yellow]📁 Cached result hit: {LOG_FILE}[/yellow]")
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- audio slice ----
    y, sr = librosa.load(vocal_audio_path, sr=16000)
    audio_duration = len(y) / sr
    if start is None:
        start = 0
    if end is None:
        end = audio_duration
    start_sample = int(float(start) * sr)
    end_sample = int(float(end) * sr)
    y_slice = y[start_sample:end_sample]

    tmp_path = None
    tmp_fd = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tmp_path = tmp_fd.name
        tmp_fd.close()
        sf.write(tmp_path, y_slice, sr, format="MP3")

        # ---- config ----
        api_key = load_key("whisper.soniox_api_key")
        if not api_key:
            raise RuntimeError(
                "whisper.soniox_api_key is empty. "
                "Please set it in config.yaml (get one from https://console.soniox.com)."
            )
        try:
            diarize = bool(load_key("whisper.soniox_diarize"))
        except KeyError:
            diarize = False
        language = load_key("whisper.language")
        if language == "auto":
            language = None

        t0 = time.time()
        tx_id = file_id = None
        try:
            # 3-step Soniox async API
            rprint("[cyan]📤 [1/4] Upload to Soniox /v1/files ...[/cyan]")
            file_id = _upload_file(api_key, tmp_path)
            rprint(f"[green]    ✓ file_id={file_id}[/green]")

            rprint(
                f"[cyan]🛠  [2/4] Create transcription "
                f"(model={MODEL}, lang={language!r}, diarize={diarize}) ...[/cyan]"
            )
            tx_id = _create_transcription(api_key, file_id, language=language, diarize=diarize)
            rprint(f"[green]    ✓ transcription_id={tx_id}[/green]")

            rprint(f"[cyan]⏳ [3/4] Polling status (interval={POLL_INTERVAL_SEC}s, timeout={_poll_timeout_sec()}s) ...[/cyan]")
            _poll_transcription(api_key, tx_id)
            rprint(f"[green]    ✓ completed in {time.time()-t0:.2f}s[/green]")

            rprint("[cyan]📥 [4/4] Fetch transcript ...[/cyan]")
            api_resp = _fetch_transcript(api_key, tx_id)
        finally:
            _best_effort_delete(api_key, tx_id, file_id)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # ---- detected language write-back (parity with elevenlabs_asr) ----
    # Soniox token.language is null when language_hints is given (and no top-level
    # language field). Fallback to the explicit config language so downstream
    # modules that read whisper.detected_language behave the same as ElevenLabs.
    tokens = api_resp.get("tokens", [])
    detected = _pick_detected_language(tokens)
    if not detected:
        cfg_lang = load_key("whisper.language")
        if cfg_lang and cfg_lang != "auto":
            detected = cfg_lang
    if detected:
        update_key("whisper.detected_language", detected)
        rprint(f"[green]🌍 detected_language={detected}[/green]")

    # ---- adapt Soniox -> whisper style segments, with time offset ----
    parsed_result = soniox_to_whisper(api_resp, word_level_timestamp=True, time_offset=float(start))
    rprint(
        f"[green]✓ Transcription done, {len(parsed_result.get('segments', []))} segments, "
        f"{len(tokens)} tokens[/green]"
    )

    # ---- cache ----
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed_result, f, indent=4, ensure_ascii=False)

    return parsed_result


if __name__ == "__main__":
    file_path = input("Enter local audio file path (mp3): ").strip()
    result = transcribe_audio_soniox(file_path, file_path, start=0, end=None)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:1000])
