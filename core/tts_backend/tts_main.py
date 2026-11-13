import json
import os
import re
import threading
import time
from pathlib import Path

from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
from core.tts_backend.sf_fishtts import siliconflow_fish_tts_for_videolingo
from core.tts_backend.openai_tts import openai_tts
from core.tts_backend.qwen3_tts import qwen3_tts
from core.tts_backend.soniox_tts import soniox_tts
from core.tts_backend.indextts2_tts import indextts2_tts_for_videolingo
from core.tts_backend.fish_tts import fish_tts
from core.tts_backend.fish_audio_tts import fish_audio_tts
from core.tts_backend.azure_tts import azure_tts
from core.tts_backend.edge_tts import edge_tts
from core.tts_backend.sf_cosyvoice2 import cosyvoice_tts_for_videolingo
from core.tts_backend.custom_tts import custom_tts
from core.prompts import get_correct_text_prompt
from core.tts_backend._302_f5tts import f5_tts_for_videolingo
from core.tts_backend.mimo_tts import mimo_tts_for_videolingo
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.tts_backend.tts_vad import ensure_non_empty_wav, vad_compact_tts_audio
from core.utils import *
from core.utils.speaker_router import UnsupportedCloneBackendError

# --- Bad TTS quality detection ---
# Catches mid-utterance held vowels / slowdown that blow up the audio duration.
# If a TTS attempt produces audio longer than expected_dur * TTS_BAD_DUR_RATIO,
# the attempt is rejected and retried. After all retries fail, the shortest
# bad output is kept as a fallback so the pipeline never hard-crashes.
TTS_BAD_DUR_RATIO = 2.0
TTS_BAD_DUR_MIN_EXPECTED = 0.5  # Floor for expected duration (seconds)
# Re-check existing wavs on resume: drop & regenerate ones whose duration > bad_threshold.
# Wavs left by an earlier run have already been VAD-trimmed by _10_gen_audio.py, so this
# only catches the mid-utterance held-vowel / slowdown failures that VAD cannot fix.
TTS_RESCAN_BAD_ON_RESUME = True

# Lazy-initialized singleton (loading g2p_en is slow)
_TTS_ESTIMATOR = None


class BadTTSAudioError(Exception):
    """The backend kept returning audio far longer than the line can hold.

    Raised only after every retry (and the LLM text rewrite) has failed AND the
    leading speech could not be salvaged. Callers must not silently substitute
    silence: a lost line is a lost translation, which is worse than a clear
    failure the user can act on.
    """


# Every line that could not be voiced cleanly, written to output/log so the run
# can be audited afterwards instead of relying on the user reading a long log.
_BAD_TTS_REPORT = "output/log/tts_failures.jsonl"
_report_lock = threading.Lock()


def report_bad_tts(save_as: str, text: str, kept: float, produced: float, action: str) -> None:
    """Append one unvoiced/degraded line to the run's failure report."""
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": str(save_as),
        "text": text,
        "expected_vs_produced_s": [round(kept, 2), round(produced, 2)],
        "action": action,
    }
    try:
        path = Path(_BAD_TTS_REPORT)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _report_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _salvage_leading_speech(save_as: str, limit_s: float) -> float | None:
    """Keep only the first run of speech from a looping TTS output.

    Duration checks alone cannot tell a hallucination from a genuinely long
    line, but a loop leaves a telltale shape: several separate speech segments
    where the first one already covers the requested words. Cut after that
    first utterance and return its length, or None when even the first segment
    is longer than the line can hold.
    """
    try:
        import torch
        from silero_vad import get_speech_timestamps

        from core.tts_backend.tts_vad import _audiosegment_to_vad_samples, _get_tts_vad_model

        audio = AudioSegment.from_wav(save_as)
        samples = _audiosegment_to_vad_samples(audio)
        if samples.size == 0:
            return None
        with _vad_lock_tts():
            model = _get_tts_vad_model()
            segs = get_speech_timestamps(
                torch.from_numpy(samples).float(),
                model,
                sampling_rate=16000,
                min_speech_duration_ms=80,
                return_seconds=True,
            )
        if len(segs) < 2:
            return None
        first_end_ms = int(round(float(segs[0]["end"]) * 1000))
        first_end_ms = min(len(audio), first_end_ms + 100)
        clipped = audio[:first_end_ms]
        kept = len(clipped) / 1000.0
        if kept <= 0 or kept > limit_s:
            return None
        clipped.export(save_as, format="wav")
        return kept
    except Exception:
        return None


def _vad_lock_tts():
    """The shared silero model must not be run concurrently."""
    from core.tts_backend import tts_vad

    return tts_vad._vad_lock


def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()

def tts_main(text, save_as, number, task_df, speaker_id=None, speed=None):
    """Generate a single audio clip via the configured TTS backend.

    Multi-speaker routing (added in C4):
        When `speaker_id` is provided AND `multi_speaker_enabled=true` AND the
        speaker_picker has produced a `speaker_voice_map`, the per-speaker
        config (method / voice / ref_wav) overrides the global tts_method for
        this single call. When no override applies, behaviour is identical to
        the pre-C4 single-voice pipeline (backwards-compatible default).

    `speed` overrides the backend's configured speaking rate for this call.
        Only soniox_tts can honour it (native rate control); every other backend
        ignores it and the timeline fitter falls back to ffmpeg atempo.
    """
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return
    
    # Estimate expected speech duration for bad-quality detection
    # (used both for new generations and to rescan existing wavs on resume).
    # If the TTS output is much longer than the linguistic estimate it almost
    # always means the model held a vowel / slowed down / looped mid-utterance.
    global _TTS_ESTIMATOR
    if _TTS_ESTIMATOR is None:
        _TTS_ESTIMATOR = init_estimator()
    expected_dur = max(TTS_BAD_DUR_MIN_EXPECTED, estimate_duration(text, _TTS_ESTIMATOR))
    bad_threshold = expected_dur * TTS_BAD_DUR_RATIO

    # Resume: bad-quality detection must inspect the raw existing wav before
    # any VAD compaction. If resume rescanning is disabled, do not trim here:
    # returning the cached file unchanged is the only way to guarantee no VAD
    # happens before bad-quality detection.
    if os.path.exists(save_as):
        if not TTS_RESCAN_BAD_ON_RESUME:
            return
        try:
            existing_dur = get_audio_duration(save_as)
            if existing_dur <= bad_threshold:
                ensure_non_empty_wav(save_as)
                vad_compact_tts_audio(save_as)
                ensure_non_empty_wav(save_as)
                return
            rprint(f"[yellow][BadTTS-rescan] {save_as} dur={existing_dur:.2f}s > {bad_threshold:.2f}s (expected {expected_dur:.2f}s); regenerating[/yellow]")
            os.remove(save_as)
        except Exception as e:
            rprint(f"[yellow][BadTTS-rescan] cannot probe {save_as} ({e}); regenerating[/yellow]")
            try:
                os.remove(save_as)
            except Exception:
                pass

    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")

    # --- C4 multi-speaker routing ---
    # When the caller passes a speaker_id, consult the per-speaker voice map
    # populated by the Streamlit picker. The router returns None whenever the
    # feature is disabled / sid is missing / entry is "default" / config is
    # malformed, in which case we silently fall back to the global tts_method
    # (legacy single-voice behaviour, fully backwards-compatible).
    voice_cfg = None
    # Truthy check: treat both None and empty-string as "no speaker_id given"
    # so callers (e.g. ASR rows where the column is blank) don't trip routing.
    if speaker_id:
        try:
            from core.utils.speaker_router import resolve_voice_cfg
            voice_cfg = resolve_voice_cfg(speaker_id)
        except UnsupportedCloneBackendError:
            # An explicit clone request must not silently become a preset voice.
            raise
        except Exception as e:  # noqa: BLE001 - never let routing break TTS
            rprint(f"[yellow]🎤 tts_main: speaker_router lookup failed for "
                   f"'{speaker_id}': {e}; falling back to global voice.[/yellow]")
            voice_cfg = None
    if voice_cfg is not None:
        rprint(f"[cyan]🎤 tts_main: speaker '{speaker_id}' → "
               f"method={voice_cfg['method']} clone={voice_cfg['is_clone']}[/cyan]")
        TTS_METHOD = voice_cfg["method"]
    # voice_cfg is forwarded into every backend call below; each *_for_videolingo
    # signature accepts voice_cfg=None as the legacy default, so single-voice
    # callers (speaker_id == None) get identical behaviour to pre-C4.

    max_retries = 3
    # Keep the shortest bad output as a fallback so we never hard-crash the pipeline
    fallback_blob = None
    fallback_dur = float('inf')

    for attempt in range(max_retries):
        try:
            if attempt >= max_retries - 1:
                print("Asking GPT to correct text...")
                try:
                    correct_text = ask_gpt(get_correct_text_prompt(text), resp_type="json", log_title='tts_correct_text')
                    text = correct_text['text']
                    # Recompute expected duration after GPT rewrite (length may differ)
                    expected_dur = max(TTS_BAD_DUR_MIN_EXPECTED, estimate_duration(text, _TTS_ESTIMATOR))
                    bad_threshold = expected_dur * TTS_BAD_DUR_RATIO
                except Exception as ge:
                    print(f"GPT correction failed: {ge}; using original text for last attempt")
            if TTS_METHOD == 'openai_tts':
                openai_tts(text, save_as, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'qwen3_tts':
                qwen3_tts(text, save_as, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'soniox_tts':
                soniox_tts(text, save_as, voice_cfg=voice_cfg, speed=speed)
            elif TTS_METHOD == 'gpt_sovits':
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'fish_tts':
                fish_tts(text, save_as, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'fish_audio_tts':
                fish_audio_tts(text, save_as, voice_cfg=voice_cfg, number=number)
            elif TTS_METHOD == 'azure_tts':
                azure_tts(text, save_as, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'sf_fish_tts':
                siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'edge_tts':
                edge_tts(text, save_as, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'custom_tts':
                custom_tts(text, save_as, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'sf_cosyvoice2':
                cosyvoice_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'f5tts':
                f5_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'mimo_tts':
                mimo_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'indextts2':
                indextts2_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
                
            # Check generated audio duration on the raw backend output.
            # Bad-quality detection must run before any VAD compaction; otherwise
            # long held vowels / slowdowns could be hidden by trimming silence.
            ensure_non_empty_wav(save_as)
            duration = get_audio_duration(save_as)
            if duration <= 0:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed (empty audio: duration=0), retrying...")
                continue

            # Bad-quality detection: catches held vowels / slowdown blowing up duration
            if duration > bad_threshold:
                # Remember the shortest bad output so far in case every retry is bad
                if duration < fallback_dur:
                    fallback_dur = duration
                    try:
                        with open(save_as, 'rb') as fb:
                            fallback_blob = fb.read()
                    except Exception:
                        fallback_blob = None
                rprint(f"[yellow][BadTTS] attempt {attempt + 1}/{max_retries}: dur={duration:.2f}s > {bad_threshold:.2f}s (expected {expected_dur:.2f}s x {TTS_BAD_DUR_RATIO}); retrying for: {text[:60]}[/yellow]")
                if attempt < max_retries - 1:
                    if os.path.exists(save_as):
                        os.remove(save_as)
                    continue
                # Exhausted all retries: write back the shortest bad output as fallback
                if fallback_blob is not None:
                    with open(save_as, 'wb') as fb:
                        fb.write(fallback_blob)
                ensure_non_empty_wav(save_as)
                vad_compact_tts_audio(save_as)
                ensure_non_empty_wav(save_as)
                # A hallucinated 20-40 s clip is usually a loop: the FIRST
                # utterance is still the requested words and the rest is the
                # model repeating itself. Keep the leading speech, drop the
                # loop, and say so loudly, instead of losing the line or
                # poisoning the chunk timeline.
                fallback_after_vad = get_audio_duration(save_as)
                if fallback_after_vad > bad_threshold:
                    salvaged = _salvage_leading_speech(save_as, bad_threshold)
                    if salvaged is not None:
                        rprint(
                            f"[yellow][BadTTS-SALVAGED] kept {salvaged:.2f}s of the leading "
                            f"speech from a {fallback_after_vad:.2f}s repeated output for "
                            f"{save_as} -- the tail was the model looping; check this line "
                            f"by ear: {text[:60]}[/yellow]"
                        )
                        report_bad_tts(save_as, text, salvaged, fallback_after_vad, "salvaged")
                        return
                    report_bad_tts(save_as, text, 0.0, fallback_after_vad, "dropped")
                    raise BadTTSAudioError(
                        f"TTS produced {fallback_after_vad:.2f}s for a line expected to "
                        f"take {expected_dur:.2f}s after {max_retries} attempts and the "
                        f"leading speech could not be salvaged: {text[:60]}"
                    )
                rprint(f"[red][BadTTS] gave up after {max_retries} attempts; using shortest fallback ({fallback_dur:.2f}s vs expected {expected_dur:.2f}s) for {save_as}[/red]")
                return

            # Bad-quality detection passed on raw audio. Now apply the final
            # silero-vad post-trim shared by qwen3_tts, mimo_tts, and all other
            # backends.
            ensure_non_empty_wav(save_as)
            vad_compact_tts_audio(save_as)
            ensure_non_empty_wav(save_as)
            return
        except BadTTSAudioError:
            # Already retried and text-rewritten inside the loop; retrying again
            # would only burn more API calls on a line the model cannot speak.
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:300]}, retrying...")
