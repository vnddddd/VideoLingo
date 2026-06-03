import threading
import wave
from functools import lru_cache

import numpy as np
from pydub import AudioSegment
from rich import print as rprint


MIN_SEGMENT_DURATION_MS = 10
# VAD-based TTS compaction settings.  silero-vad is the sole post-trimmer; there
# is no dBFS fallback.  Any import/runtime failure must halt the pipeline so it
# never silently produces un-trimmed TTS audio.
TTS_VAD_ENABLED = True
TTS_VAD_SAMPLE_RATE = 16000
TTS_VAD_PAUSE_THRESHOLD_MS = 200
TTS_VAD_PAUSE_KEEP_MS = 100
TTS_VAD_HEAD_PAD_MS = 50
TTS_VAD_TAIL_PAD_MS = 100
TTS_VAD_MIN_SPEECH_MS = 80
TTS_VAD_MIN_TOTAL_SAVED_MS = 40
TTS_VAD_ALL_SILENCE_KEEP_MS = 50

_vad_lock = threading.Lock()

def wav_has_audio_frames(audio_file: str) -> bool:
    """Return True only when a WAV file contains at least one audio frame."""
    try:
        with wave.open(audio_file, 'rb') as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except Exception:
        return False

def ensure_non_empty_wav(audio_file: str) -> None:
    """Replace empty/zero-frame WAV output with a tiny silence segment."""
    if not wav_has_audio_frames(audio_file):
        AudioSegment.silent(duration=MIN_SEGMENT_DURATION_MS).set_frame_rate(16000).set_channels(1).export(audio_file, format="wav")
        rprint(f"[yellow]Empty audio segment replaced with {MIN_SEGMENT_DURATION_MS}ms silence: {audio_file}[/yellow]")



@lru_cache(maxsize=1)
def _get_tts_vad_model():
    """Load silero-vad eagerly on first use (no dBFS fallback exists)."""
    try:
        from silero_vad import load_silero_vad
    except ImportError as e:
        raise ImportError(
            "silero-vad is required for TTS post-trimming but is not installed. "
            "Install it with: pip install -r requirements.txt  (or: pip install silero-vad torch)."
        ) from e
    return load_silero_vad()


def _audiosegment_to_float32_mono(audio: AudioSegment) -> np.ndarray:
    """Convert a mono AudioSegment to float32 samples in [-1, 1]."""
    samples = np.array(audio.get_array_of_samples())
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)

    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    max_abs = float(1 << (8 * audio.sample_width - 1))
    if max_abs <= 0:
        max_abs = float(np.max(np.abs(samples)) or 1.0)
    return (samples.astype(np.float32) / max_abs).clip(-1.0, 1.0)


def _audiosegment_to_vad_samples(audio: AudioSegment) -> np.ndarray:
    """Prepare 16 kHz mono float32 samples for silero-vad."""
    vad_audio = audio.set_channels(1).set_frame_rate(TTS_VAD_SAMPLE_RATE).set_sample_width(2)
    return _audiosegment_to_float32_mono(vad_audio)


def _replace_audiosegment_wav(audio_file: str, audio: AudioSegment) -> None:
    audio.export(audio_file, format="wav")
    ensure_non_empty_wav(audio_file)




def vad_compact_tts_audio(audio_file: str) -> bool:
    """Compact generated TTS WAV in place using silero-vad.

    It trims leading/trailing non-speech and compresses long internal pauses.
    Returns True only when the WAV was rewritten.
    """
    if not TTS_VAD_ENABLED:
        return False

    audio = AudioSegment.from_wav(audio_file)
    if len(audio) <= MIN_SEGMENT_DURATION_MS:
        rprint(f"[dim]VAD skipped TTS audio: {audio_file} too short ({len(audio)}ms)[/dim]")
        return False

    vad_samples = _audiosegment_to_vad_samples(audio)
    if vad_samples.size == 0:
        placeholder = AudioSegment.silent(duration=TTS_VAD_ALL_SILENCE_KEEP_MS).set_frame_rate(16000).set_channels(1)
        _replace_audiosegment_wav(audio_file, placeholder)
        return True

    import torch
    from silero_vad import get_speech_timestamps

    # Silero's torch/onnx session is loaded once and protected during inference;
    # TTS generation can run in parallel threads.
    with _vad_lock:
        model = _get_tts_vad_model()
        segs = get_speech_timestamps(
            torch.from_numpy(vad_samples).float(),
            model,
            sampling_rate=TTS_VAD_SAMPLE_RATE,
            min_speech_duration_ms=TTS_VAD_MIN_SPEECH_MS,
            return_seconds=True,
        )

    orig_len_ms = len(audio)
    if not segs:
        placeholder = AudioSegment.silent(duration=TTS_VAD_ALL_SILENCE_KEEP_MS).set_frame_rate(audio.frame_rate).set_channels(audio.channels)
        _replace_audiosegment_wav(audio_file, placeholder)
        rprint(f"[yellow]VAD compacted TTS audio: {audio_file} all-silence -> {TTS_VAD_ALL_SILENCE_KEEP_MS}ms[/yellow]")
        return True

    segs = [dict(s) for s in segs]

    chunks: list[AudioSegment] = []
    pauses_compressed = 0
    pause_saved_ms = 0
    for i, seg in enumerate(segs):
        start_ms = int(round(float(seg["start"]) * 1000))
        end_ms = int(round(float(seg["end"]) * 1000))
        if i == 0:
            start_ms = max(0, start_ms - TTS_VAD_HEAD_PAD_MS)
        if i == len(segs) - 1:
            end_ms = min(orig_len_ms, end_ms + TTS_VAD_TAIL_PAD_MS)
        if end_ms > start_ms:
            chunks.append(audio[start_ms:end_ms])

        if i < len(segs) - 1:
            pause_start_ms = int(round(float(seg["end"]) * 1000))
            pause_end_ms = int(round(float(segs[i + 1]["start"]) * 1000))
            pause_len_ms = max(0, pause_end_ms - pause_start_ms)
            if pause_len_ms > TTS_VAD_PAUSE_THRESHOLD_MS:
                chunks.append(AudioSegment.silent(duration=TTS_VAD_PAUSE_KEEP_MS, frame_rate=audio.frame_rate))
                pauses_compressed += 1
                pause_saved_ms += pause_len_ms - TTS_VAD_PAUSE_KEEP_MS
            elif pause_len_ms > 0:
                chunks.append(audio[pause_start_ms:pause_end_ms])

    compacted = sum(chunks, AudioSegment.empty()) if chunks else AudioSegment.silent(duration=TTS_VAD_ALL_SILENCE_KEEP_MS, frame_rate=audio.frame_rate)
    if len(compacted) <= 0:
        compacted = AudioSegment.silent(duration=TTS_VAD_ALL_SILENCE_KEEP_MS, frame_rate=audio.frame_rate)

    saved_ms = orig_len_ms - len(compacted)
    if saved_ms < TTS_VAD_MIN_TOTAL_SAVED_MS and pauses_compressed == 0:
        rprint(
            f"[dim]VAD checked TTS audio: {audio_file} no trim "
            f"(saved={max(0, saved_ms)}ms < {TTS_VAD_MIN_TOTAL_SAVED_MS}ms threshold, "
            f"speech_segs={len(segs)})[/dim]"
        )
        return False

    _replace_audiosegment_wav(audio_file, compacted)
    rprint(
        f"[dim]VAD compacted TTS audio: {audio_file} -{max(0, saved_ms)}ms "
        f"(speech_segs={len(segs)}, pauses={pauses_compressed}, pause_saved={pause_saved_ms}ms)[/dim]"
    )
    return True

