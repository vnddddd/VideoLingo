from concurrent.futures import ThreadPoolExecutor, as_completed

from pydub.utils import mediainfo

from core.utils import *
from core.asr_backend.demucs_vl import demucs_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *


def _load_asr_max_workers(runtime: str, segments_count: int) -> int:
    """Return safe ASR worker count; keep local WhisperX serial to protect GPU/VRAM."""
    if runtime == "local":
        return 1
    try:
        max_workers = int(load_key("whisper.max_workers") or 1)
    except Exception:
        max_workers = 1
    max_workers = max(1, max_workers)
    return min(max_workers, max(1, segments_count))


@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Demucs vocal separation:
    if load_key("demucs"):
        demucs_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio segments
    # Multi-speaker mode (C2): ASR with diarization must see the whole audio in one
    # shot. Otherwise Soniox "speaker 1" on clip A and "speaker 1" on clip B are
    # NOT guaranteed to be the same person, so cross-clip merging would be wrong.
    multi_speaker = False
    try:
        multi_speaker = bool(load_key("multi_speaker_enabled"))
    except KeyError:
        multi_speaker = False
    asr_runtime_peek = load_key("whisper.runtime")
    if multi_speaker and asr_runtime_peek in ("soniox", "elevenlabs"):
        try:
            duration = float(mediainfo(_RAW_AUDIO_FILE)["duration"])
        except Exception:
            duration = 0.0
        segments = [(0.0, duration)]
        rprint(f"[cyan]🎤 Multi-speaker mode: sending whole audio ({duration:.1f}s) to {asr_runtime_peek} in one shot[/cyan]")
    else:
        segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    runtime = load_key("whisper.runtime")
    if runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local model...[/cyan]")
    elif runtime == "cloud":
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as ts
        rprint("[cyan]🎤 Transcribing audio with 302 API...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
    elif runtime == "soniox":
        from core.asr_backend.soniox_asr import transcribe_audio_soniox as ts
        rprint("[cyan]🎤 Transcribing audio with Soniox API...[/cyan]")
    else:
        raise ValueError(f"Unsupported whisper runtime: {runtime}")

    max_workers = _load_asr_max_workers(runtime, len(segments))
    rprint(f"[cyan]🎤 ASR clip concurrency: {max_workers}/{len(segments)}[/cyan]")

    if max_workers == 1:
        all_results = [ts(_RAW_AUDIO_FILE, vocal_audio, start, end) for start, end in segments]
    else:
        all_results = [None] * len(segments)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(ts, _RAW_AUDIO_FILE, vocal_audio, start, end): idx
                for idx, (start, end) in enumerate(segments)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                start, end = segments[idx]
                all_results[idx] = future.result()
                rprint(f"[green]✓ ASR segment {idx + 1}/{len(segments)} completed ({start:.2f}s-{end:.2f}s)[/green]")
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()