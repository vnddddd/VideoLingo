from pathlib import Path
import io
import json
import requests
from pydub import AudioSegment
from core.utils import load_key, except_handler, load_timeout

# Qwen3-TTS (Alibaba Cloud Bailian / DashScope) text-to-speech backend.
# Docs: https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2879134
#
# Non-streaming flow:
#   1. POST text -> DashScope returns JSON with output.audio.url (valid ~24h)
#   2. GET that url to download the audio bytes
#   3. Normalize to a standard PCM WAV via pydub. The downstream VideoLingo
#      pipeline scans every clip with the `wave` module and rejects non-WAV or
#      empty audio, so we always re-encode to WAV here.
#
# Region endpoints (Bailian DashScope):
#   beijing   -> https://dashscope.aliyuncs.com       (China mainland / Beijing)
#   singapore -> https://dashscope-intl.aliyuncs.com  (International / Singapore)
REGION_ENDPOINTS = {
    "beijing": "https://dashscope.aliyuncs.com",
    "singapore": "https://dashscope-intl.aliyuncs.com",
}
_API_PATH = "/api/v1/services/aigc/multimodal-generation/generation"

# Cherry / Ethan are confirmed timbres; the provider offers more (see docs).
# Not enforced, so newly added voices keep working.
KNOWN_VOICES = ["Cherry", "Ethan"]


@except_handler("Failed to generate audio using Qwen3 TTS", retry=3, delay=1)
def qwen3_tts(text, save_as, voice_cfg=None):
    """Alibaba Cloud Bailian Qwen3-TTS (DashScope, non-streaming).

    voice_cfg: optional dict from the C4 speaker router. When provided and
    voice_cfg["voice"] is truthy, it overrides the global qwen3_tts.voice config.
    """
    api_key = load_key("qwen3_tts.api_key")

    if voice_cfg and voice_cfg.get("voice"):
        voice = voice_cfg["voice"]
    else:
        voice = load_key("qwen3_tts.voice")

    model = load_key("qwen3_tts.model") or "qwen3-tts-flash"
    language_type = load_key("qwen3_tts.language_type")
    region = (load_key("qwen3_tts.region") or "beijing").strip().lower()

    base_url = REGION_ENDPOINTS.get(region)
    if base_url is None:
        raise ValueError(
            f"Invalid qwen3_tts.region: '{region}'. Choose from {list(REGION_ENDPOINTS)}"
        )
    url = base_url + _API_PATH

    input_obj = {"text": text, "voice": voice}
    if language_type:
        input_obj["language_type"] = language_type
    payload = {"model": model, "input": input_obj}
    headers = {"Authorization": f"Bearer {api_key}"}

    # 1. request synthesis -> obtain audio url
    resp = requests.post(url, headers=headers, json=payload,
                         timeout=load_timeout("tts", 60))
    if resp.status_code != 200:
        raise Exception(f"Qwen3-TTS API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    output = data.get("output") or {}
    audio = output.get("audio") or {}
    audio_url = audio.get("url")
    if not audio_url:
        raise Exception(
            f"Qwen3-TTS: no audio url in response: {json.dumps(data, ensure_ascii=False)[:500]}"
        )

    # 2. download audio bytes
    audio_resp = requests.get(audio_url, timeout=load_timeout("tts", 60))
    if audio_resp.status_code != 200:
        raise Exception(
            f"Qwen3-TTS: failed to download audio ({audio_resp.status_code}) from {audio_url}"
        )
    audio_bytes = audio_resp.content

    # 3. normalize to a standard PCM WAV
    speech_file_path = Path(save_as)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception:
        # Fallback: Qwen3-TTS raw audio spec is PCM 16-bit / 24000 Hz / mono
        seg = AudioSegment(data=audio_bytes, sample_width=2, frame_rate=24000, channels=1)
    seg.export(str(speech_file_path), format="wav")
    print(f"Audio saved to {speech_file_path}")


if __name__ == "__main__":
    qwen3_tts("你好，欢迎使用 VideoLingo！", "test_qwen3.wav")
