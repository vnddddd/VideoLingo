import requests
from core.utils import load_key

def azure_tts(text: str, save_path: str, voice_cfg: dict = None) -> None:
    """Azure (302.ai) TTS.

    voice_cfg: optional dict from the C4 speaker router. When provided and
    ``voice_cfg["voice"]`` is truthy, it overrides the global
    ``azure_tts.voice`` config (per-speaker multi-voice). Falsy / missing
    keeps the legacy single-voice behaviour.
    """
    url = "https://api.302.ai/cognitiveservices/v1"

    API_KEY = load_key("azure_tts.api_key")
    if voice_cfg and voice_cfg.get("voice"):
        voice = voice_cfg["voice"]
    else:
        voice = load_key("azure_tts.voice")
    
    payload = f"""<speak version='1.0' xml:lang='zh-CN'><voice name='{voice}'>{text}</voice></speak>"""
    headers = {
       'Authorization': f'Bearer {API_KEY}',
       'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
       'Content-Type': 'application/ssml+xml'
    }

    response = requests.request("POST", url, headers=headers, data=payload, timeout=load_timeout("tts", 60))

    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Audio saved to {save_path}")

if __name__ == "__main__":
    azure_tts("Hi! Welcome to VideoLingo!", "test.wav")