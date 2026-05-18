import os
import re
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
from core.tts_backend.sf_fishtts import siliconflow_fish_tts_for_videolingo
from core.tts_backend.openai_tts import openai_tts
from core.tts_backend.fish_tts import fish_tts
from core.tts_backend.azure_tts import azure_tts
from core.tts_backend.edge_tts import edge_tts
from core.tts_backend.sf_cosyvoice2 import cosyvoice_tts_for_videolingo
from core.tts_backend.custom_tts import custom_tts
from core.prompts import get_correct_text_prompt
from core.tts_backend._302_f5tts import f5_tts_for_videolingo
from core.tts_backend.mimo_tts import mimo_tts_for_videolingo
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import *

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


def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()

def tts_main(text, save_as, number, task_df, speaker_id=None):
    """Generate a single audio clip via the configured TTS backend.

    Multi-speaker routing (added in C4):
        When `speaker_id` is provided AND `multi_speaker_enabled=true` AND the
        speaker_picker has produced a `speaker_voice_map`, the per-speaker
        config (method / voice / ref_wav) overrides the global tts_method for
        this single call. When no override applies, behaviour is identical to
        the pre-C4 single-voice pipeline (backwards-compatible default).
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

    # Resume: skip iff the existing wav looks OK; drop bad leftovers so the
    # normal retry loop below regenerates them.
    if os.path.exists(save_as):
        if not TTS_RESCAN_BAD_ON_RESUME:
            return
        try:
            existing_dur = get_audio_duration(save_as)
            if existing_dur <= bad_threshold:
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
            elif TTS_METHOD == 'gpt_sovits':
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df, voice_cfg=voice_cfg)
            elif TTS_METHOD == 'fish_tts':
                fish_tts(text, save_as, voice_cfg=voice_cfg)
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
                
            # Check generated audio duration
            duration = get_audio_duration(save_as)
            if duration <= 0:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed (empty), retrying...")
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
                rprint(f"[red][BadTTS] gave up after {max_retries} attempts; using shortest fallback ({fallback_dur:.2f}s vs expected {expected_dur:.2f}s) for {save_as}[/red]")
                return

            # All good
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")