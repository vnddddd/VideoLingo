import requests
from core.utils import *
import json

@except_handler("Failed to generate audio using 302.ai Fish TTS", retry=3, delay=1)
def fish_tts(text: str, save_as: str, voice_cfg: dict = None) -> bool:
    """302.ai Fish TTS conversion.

    voice_cfg: optional dict from the C4 speaker router. When provided and
    ``voice_cfg["voice"]`` is truthy, it is treated as a Fish *character name*
    (i.e. the key into ``fish_tts.character_id_dict``); if the name is not in
    the dict we fall back to the global ``fish_tts.character`` config.
    """
    API_KEY = load_key("fish_tts.api_key")
    character_dict = load_key("fish_tts.character_id_dict")
    if voice_cfg and voice_cfg.get("voice"):
        character = voice_cfg["voice"]
        if character not in character_dict:
            print(f"[fish_tts] character '{character}' not in character_id_dict, falling back to global config")
            character = load_key("fish_tts.character")
    else:
        character = load_key("fish_tts.character")
    refer_id = character_dict[character]
    
    url = "https://api.302.ai/fish-audio/v1/tts"
    payload = json.dumps({
        "text": text,
        "reference_id": refer_id,
        "chunk_length": 200,
        "normalize": True,
        "format": "wav",
        "latency": "normal"
    })
    
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, data=payload, timeout=load_timeout("tts", 60))
    response.raise_for_status()
    response_data = response.json()
    
    if "url" in response_data:
        audio_response = requests.get(response_data["url"], timeout=load_timeout("tts", 60))
        audio_response.raise_for_status()
        
        with open(save_as, "wb") as f:
            f.write(audio_response.content)
        return True
    
    print("Request failed:", response_data)
    return False

if __name__ == '__main__':
    fish_tts("Hi! Welcome to VideoLingo!", "test.wav")
