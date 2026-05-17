import os
import time
import shutil
import subprocess
import wave
import threading
from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.tts_main import tts_main

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
MIN_SEGMENT_DURATION_MS = 10
# VAD-based TTS compaction settings.  The dBFS trimmer below is kept only as
# a defensive fallback when silero-vad/torch cannot be imported or the VAD pass
# fails on a malformed file.
TTS_VAD_ENABLED = True
TTS_VAD_SAMPLE_RATE = 16000
TTS_VAD_PAUSE_THRESHOLD_MS = 200
TTS_VAD_PAUSE_KEEP_MS = 100
TTS_VAD_HEAD_PAD_MS = 50
TTS_VAD_TAIL_PAD_MS = 100
TTS_VAD_MIN_SPEECH_MS = 80
TTS_VAD_MIN_TOTAL_SAVED_MS = 40
TTS_VAD_ALL_SILENCE_KEEP_MS = 50
TTS_VAD_TAIL_ELONG_DETECT = True
TTS_VAD_TAIL_ENERGY_RATIO = 0.35
TTS_VAD_TAIL_SAFETY_MS = 80
TTS_VAD_TAIL_MIN_CUT_MS = 500
TTS_VAD_TAIL_WIN_MS = 50
TTS_VAD_PLATEAU_DETECT = True
TTS_VAD_PLATEAU_LOOKBACK_MS = 1100
TTS_VAD_PLATEAU_KEEP_MS = 250
TTS_VAD_PLATEAU_MIN_MS = 700
TTS_VAD_PLATEAU_MIN_RATIO = 0.30
TTS_VAD_PLATEAU_MAX_RATIO = 0.75
TTS_VAD_PLATEAU_CV_RATIO = 0.18
TTS_VAD_PLATEAU_REL_DROP_RATIO = 0.22
TTS_VAD_REPETITION_DETECT = True
TTS_VAD_REPETITION_MIN_SEG_MS = 900
TTS_VAD_REPETITION_GAP_MS = 350
TTS_VAD_REPETITION_LEN_TOL = 0.35

TTS_SILENCE_TRIM_SEEK_STEP_MS = 10
TTS_SILENCE_TRIM_MIN_SILENCE_MS = 80
TTS_SILENCE_TRIM_LEADING_KEEP_MS = 80
TTS_SILENCE_TRIM_TRAILING_KEEP_MS = 120
TTS_SILENCE_TRIM_MIN_TOTAL_MS = 80
TTS_SILENCE_TRIM_MAX_LEADING_MS = 800
TTS_SILENCE_TRIM_MAX_TRAILING_MS = 1000
TTS_SILENCE_TRIM_NOISE_MARGIN_DB = 12.0
TTS_SILENCE_TRIM_MIN_THRESHOLD_DBFS = -55.0
TTS_SILENCE_TRIM_MAX_THRESHOLD_DBFS = -35.0

_vad_lock = threading.Lock()

def _wav_has_audio_frames(audio_file: str) -> bool:
    """Return True only when a WAV file contains at least one audio frame."""
    try:
        with wave.open(audio_file, 'rb') as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except Exception:
        return False

def _ensure_non_empty_wav(audio_file: str) -> None:
    """Replace empty/zero-frame WAV output with a tiny silence segment."""
    if not _wav_has_audio_frames(audio_file):
        AudioSegment.silent(duration=MIN_SEGMENT_DURATION_MS).set_frame_rate(16000).set_channels(1).export(audio_file, format="wav")
        rprint(f"[yellow]Empty audio segment replaced with {MIN_SEGMENT_DURATION_MS}ms silence: {audio_file}[/yellow]")



@lru_cache(maxsize=1)
def _get_tts_vad_model():
    """Load silero-vad lazily so importing this module does not require torch."""
    from silero_vad import load_silero_vad
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
    _ensure_non_empty_wav(audio_file)


def _trim_tts_tail_elongation(audio: AudioSegment, segs: list[dict]) -> Optional[dict]:
    """Shorten obvious held-vowel/repetition tails that VAD still marks as speech."""
    if not segs:
        return None

    meta = None

    if TTS_VAD_REPETITION_DETECT and len(segs) >= 2:
        seg_lens = [float(s["end"] - s["start"]) for s in segs]
        first_len = seg_lens[0]
        if first_len * 1000 >= TTS_VAD_REPETITION_MIN_SEG_MS:
            keep_mask = [True] * len(segs)
            for j in range(1, len(segs)):
                if seg_lens[j] * 1000 < TTS_VAD_REPETITION_MIN_SEG_MS:
                    continue
                gap_before = float(segs[j]["start"] - segs[j - 1]["end"])
                len_diff = abs(seg_lens[j] - first_len) / max(seg_lens[j], first_len)
                if (
                    gap_before * 1000 <= TTS_VAD_REPETITION_GAP_MS
                    and len_diff <= TTS_VAD_REPETITION_LEN_TOL
                ):
                    keep_mask[j] = False
            if not all(keep_mask):
                dropped = len(segs) - sum(keep_mask)
                segs[:] = [s for s, keep in zip(segs, keep_mask) if keep]
                meta = {"mode": "repetition", "dropped_segs": dropped}
                if not segs:
                    return meta

    if not TTS_VAD_TAIL_ELONG_DETECT:
        return meta

    last = segs[-1]
    sr = audio.frame_rate
    mono = audio.set_channels(1)
    samples = _audiosegment_to_float32_mono(mono)
    start = max(0, int(float(last["start"]) * sr))
    end = min(len(samples), int(float(last["end"]) * sr))
    tail = samples[start:end]
    if len(tail) <= int(0.3 * sr):
        return meta

    win = max(1, int(TTS_VAD_TAIL_WIN_MS / 1000 * sr))
    n_win = len(tail) // win
    if n_win <= 3:
        return meta

    energies = np.array([
        float(np.sqrt(np.mean(tail[i * win:(i + 1) * win] ** 2)) + 1e-12)
        for i in range(n_win)
    ], dtype=np.float64)
    peak = float(np.max(energies))
    thresh = peak * TTS_VAD_TAIL_ENERGY_RATIO
    loud = energies >= thresh
    loud_idxs = np.where(loud)[0]
    last_loud_idx = int(loud_idxs[-1]) if len(loud_idxs) else -1

    if last_loud_idx >= 0:
        cut_sample = start + (last_loud_idx + 1) * win + int(TTS_VAD_TAIL_SAFETY_MS / 1000 * sr)
        removed_ms = (end - cut_sample) / sr * 1000
        if removed_ms >= TTS_VAD_TAIL_MIN_CUT_MS:
            last["end"] = max(float(last["start"]), cut_sample / sr)
            meta = {
                "mode": "fade",
                "removed_ms": int(round(removed_ms)),
                "peak_rms": round(peak, 4),
                "thresh_rms": round(thresh, 4),
            }

    if TTS_VAD_PLATEAU_DETECT and peak > 1e-6 and last_loud_idx >= 0:
        look_wins = max(1, int(round(TTS_VAD_PLATEAU_LOOKBACK_MS / TTS_VAD_TAIL_WIN_MS)))
        keep_wins = max(1, int(round(TTS_VAD_PLATEAU_KEEP_MS / TTS_VAD_TAIL_WIN_MS)))
        min_plateau_wins = max(1, int(round(TTS_VAD_PLATEAU_MIN_MS / TTS_VAD_TAIL_WIN_MS)))
        plateau_end_idx = last_loud_idx + 1
        start_min = max(0, plateau_end_idx - look_wins)
        start_max = plateau_end_idx - min_plateau_wins
        for start_idx in range(start_min, start_max + 1):
            plateau = energies[start_idx:plateau_end_idx]
            if len(plateau) < min_plateau_wins:
                continue
            ratios = plateau / peak
            mean_ratio = float(np.mean(ratios))
            cv = float(np.std(plateau) / (np.mean(plateau) + 1e-12))
            rel_drop = float((plateau[0] - plateau[-1]) / (plateau[0] + 1e-12))
            if (
                TTS_VAD_PLATEAU_MIN_RATIO <= mean_ratio <= TTS_VAD_PLATEAU_MAX_RATIO
                and cv <= TTS_VAD_PLATEAU_CV_RATIO
                and abs(rel_drop) <= TTS_VAD_PLATEAU_REL_DROP_RATIO
                and float(np.min(ratios)) >= TTS_VAD_PLATEAU_MIN_RATIO * 0.75
                and float(np.max(ratios)) <= TTS_VAD_PLATEAU_MAX_RATIO * 1.15
            ):
                cut_win_idx = min(plateau_end_idx, start_idx + keep_wins)
                cut_sample = start + cut_win_idx * win + int(TTS_VAD_TAIL_SAFETY_MS / 1000 * sr)
                removed_ms = (end - cut_sample) / sr * 1000
                if removed_ms >= TTS_VAD_TAIL_MIN_CUT_MS:
                    last["end"] = max(float(last["start"]), cut_sample / sr)
                    meta = {
                        "mode": "plateau",
                        "removed_ms": int(round(removed_ms)),
                        "mean_ratio": round(mean_ratio, 3),
                        "cv": round(cv, 3),
                    }
                    break

    return meta


def vad_compact_tts_audio(audio_file: str) -> bool:
    """Compact generated TTS WAV in place using silero-vad.

    It trims leading/trailing non-speech and compresses long internal pauses.
    Returns True only when the WAV was rewritten.
    """
    if not TTS_VAD_ENABLED:
        return False

    audio = AudioSegment.from_wav(audio_file)
    if len(audio) <= MIN_SEGMENT_DURATION_MS:
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
    tail_meta = _trim_tts_tail_elongation(audio, segs)

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
    if saved_ms < TTS_VAD_MIN_TOTAL_SAVED_MS and not tail_meta and pauses_compressed == 0:
        return False

    _replace_audiosegment_wav(audio_file, compacted)
    msg = (
        f"VAD compacted TTS audio: {audio_file} -{max(0, saved_ms)}ms "
        f"(speech_segs={len(segs)}, pauses={pauses_compressed}, pause_saved={pause_saved_ms}ms"
    )
    if tail_meta:
        msg += f", tail={tail_meta}"
    msg += ")"
    rprint(f"[dim]{msg}[/dim]")
    return True


def _estimate_tts_silence_threshold(audio: AudioSegment) -> float:
    """Estimate a conservative silence threshold from short-frame loudness."""
    if len(audio) <= 0:
        return TTS_SILENCE_TRIM_MIN_THRESHOLD_DBFS

    frame_dbfs = []
    for start_ms in range(0, len(audio), TTS_SILENCE_TRIM_SEEK_STEP_MS):
        frame = audio[start_ms:start_ms + TTS_SILENCE_TRIM_SEEK_STEP_MS]
        dbfs = frame.dBFS
        frame_dbfs.append(-100.0 if dbfs == float("-inf") else dbfs)

    if not frame_dbfs:
        return TTS_SILENCE_TRIM_MIN_THRESHOLD_DBFS

    sorted_dbfs = sorted(frame_dbfs)
    quiet_count = max(1, int(len(sorted_dbfs) * 0.1))
    noise_floor = sum(sorted_dbfs[:quiet_count]) / quiet_count
    threshold = noise_floor + TTS_SILENCE_TRIM_NOISE_MARGIN_DB
    return max(
        TTS_SILENCE_TRIM_MIN_THRESHOLD_DBFS,
        min(TTS_SILENCE_TRIM_MAX_THRESHOLD_DBFS, threshold),
    )

def trim_tts_leading_trailing_silence(audio_file: str) -> bool:
    """Trim obvious leading/trailing silence from a generated TTS WAV in place.

    The trimming is intentionally conservative: it uses a dynamic dBFS
    threshold, requires a short non-silent run, keeps padding around speech,
    caps how much can be removed at either edge, skips tiny changes, and never
    rewrites all-silent audio. Returns True only when the file was rewritten.
    """
    audio = AudioSegment.from_wav(audio_file)
    if len(audio) <= MIN_SEGMENT_DURATION_MS:
        return False

    silence_thresh = _estimate_tts_silence_threshold(audio)
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=TTS_SILENCE_TRIM_MIN_SILENCE_MS,
        silence_thresh=silence_thresh,
        seek_step=TTS_SILENCE_TRIM_SEEK_STEP_MS,
    )
    if not nonsilent_ranges:
        return False

    first_voice_ms = nonsilent_ranges[0][0]
    last_voice_ms = nonsilent_ranges[-1][1]

    start_ms = max(0, first_voice_ms - TTS_SILENCE_TRIM_LEADING_KEEP_MS)
    end_ms = min(len(audio), last_voice_ms + TTS_SILENCE_TRIM_TRAILING_KEEP_MS)

    # Guard against cutting too much if a very soft syllable was misclassified.
    start_ms = min(start_ms, TTS_SILENCE_TRIM_MAX_LEADING_MS)
    end_ms = max(end_ms, len(audio) - TTS_SILENCE_TRIM_MAX_TRAILING_MS)

    if end_ms <= start_ms:
        return False

    trimmed_total_ms = start_ms + (len(audio) - end_ms)
    if trimmed_total_ms < TTS_SILENCE_TRIM_MIN_TOTAL_MS:
        return False

    trimmed_audio = audio[start_ms:end_ms]
    if len(trimmed_audio) <= 0:
        return False

    trimmed_audio.export(audio_file, format="wav")
    _ensure_non_empty_wav(audio_file)
    rprint(
        f"[dim]Trimmed TTS silence: {audio_file} "
        f"-{trimmed_total_ms}ms (threshold {silence_thresh:.1f} dBFS)[/dim]"
    )
    return True

def parse_df_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    hours, minutes, seconds = time_str.strip().split(':')
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def adjust_audio_speed(input_file: str, output_file: str, speed_factor: float) -> None:
    """Adjust audio speed and handle edge cases"""
    # If the speed factor is close to 1, directly copy the file
    if abs(speed_factor - 1.0) < 0.001:
        shutil.copy2(input_file, output_file)
        _ensure_non_empty_wav(output_file)
        return
        
    atempo = speed_factor
    cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={atempo}', '-y', output_file]
    input_duration = get_audio_duration(input_file)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            _ensure_non_empty_wav(output_file)
            output_duration = get_audio_duration(output_file)
            expected_duration = input_duration / speed_factor
            diff = output_duration - expected_duration
            # If the output duration exceeds the expected duration, but the input audio is less than 3 seconds, and the error is within 0.1 seconds, truncate to the expected length
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                audio = AudioSegment.from_wav(output_file)
                trimmed_audio = audio[:(expected_duration * 1000)]  # pydub uses milliseconds
                trimmed_audio.export(output_file, format="wav")
                _ensure_non_empty_wav(output_file)
                print(f"Trimmed to expected duration: {expected_duration:.2f} seconds")
                return
            elif output_duration >= expected_duration * 1.02:
                raise Exception(f"Audio duration abnormal: input file={input_file}, output file={output_file}, speed factor={speed_factor}, input duration={input_duration:.2f}s, output duration={output_duration:.2f}s")
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                rprint(f"[yellow]Warning: Audio speed adjustment failed, retrying in 1s ({attempt + 1}/{max_retries})[/yellow]")
                time.sleep(1)
            else:
                rprint(f"[red]Error: Audio speed adjustment failed, max retries reached ({max_retries})[/red]")
                raise e

def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
    real_dur = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        tts_main(line, temp_file, number, tasks_df)
        _ensure_non_empty_wav(temp_file)
        try:
            vad_compact_tts_audio(temp_file)
        except Exception as e:
            rprint(f"[yellow]Warning: VAD TTS compaction failed for {temp_file}, falling back to dBFS trim: {e}[/yellow]")
            trim_tts_leading_trailing_silence(temp_file)
        _ensure_non_empty_wav(temp_file)
        real_dur += get_audio_duration(temp_file)
    return number, real_dur

def generate_tts_audio(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate TTS audio sequentially and calculate actual duration"""
    tasks_df['real_dur'] = 0
    rprint("[bold green]Starting TTS audio generation...[/bold green]")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Generating TTS audio...", total=len(tasks_df))
        
        # warm up for first 5 rows
        warmup_size = min(WARMUP_SIZE, len(tasks_df))
        for _, row in tasks_df.head(warmup_size).iterrows():
            try:
                number, real_dur = process_row(row, tasks_df)
                tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                progress.advance(task)
            except Exception as e:
                rprint(f"[red]Error: Error in warmup: {str(e)}[/red]")
                raise e
        
        # for gpt_sovits, do not use parallel to avoid mistakes
        max_workers = load_positive_int("tts_max_workers", fallback_key="max_workers", default=1) if load_key("tts_method") != "gpt_sovits" else 1
        # parallel processing for remaining tasks
        if len(tasks_df) > warmup_size:
            remaining_tasks = tasks_df.iloc[warmup_size:].copy()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_row, row, tasks_df.copy())
                    for _, row in remaining_tasks.iterrows()
                ]
                
                for future in as_completed(futures):
                    try:
                        number, real_dur = future.result()
                        tasks_df.loc[tasks_df['number'] == number, 'real_dur'] = real_dur
                        progress.advance(task)
                    except Exception as e:
                        rprint(f"[red]Error: Error: {str(e)}[/red]")
                        raise e

    rprint("[bold green]TTS audio generation completed![/bold green]")
    return tasks_df

def process_chunk(chunk_df: pd.DataFrame, accept: float, min_speed: float) -> tuple[float, bool]:
    """Process audio chunk and calculate speed factor"""
    chunk_durs = chunk_df['real_dur'].sum()
    tol_durs = chunk_df['tol_dur'].sum()
    durations = tol_durs - chunk_df.iloc[-1]['tolerance']
    all_gaps = chunk_df['gap'].sum() - chunk_df.iloc[-1]['gap']
    
    keep_gaps = True
    speed_var_error = 0.1

    if (chunk_durs + all_gaps) / accept < durations:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (durations-speed_var_error))
    elif chunk_durs / accept < durations:
        speed_factor = max(min_speed, chunk_durs / (durations-speed_var_error))
        keep_gaps = False
    elif (chunk_durs + all_gaps) / accept < tol_durs:
        speed_factor = max(min_speed, (chunk_durs + all_gaps) / (tol_durs-speed_var_error))
    else:
        speed_factor = chunk_durs / (tol_durs-speed_var_error)
        keep_gaps = False
        
    return round(speed_factor, 3), keep_gaps

def merge_chunks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Merge audio chunks and adjust timeline"""
    rprint("[bold blue]Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")
    chunk_start = 0
    
    tasks_df['new_sub_times'] = None
    
    for index, row in tasks_df.iterrows():
        if row['cut_off'] == 1:
            chunk_df = tasks_df.iloc[chunk_start:index+1].reset_index(drop=True)
            speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
            
            # Step1: Start processing new timeline
            chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
            chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance'] # 加上tolerance才是这一块的结束
            cur_time = chunk_start_time
            for i, row in chunk_df.iterrows():
                # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
                if i != 0 and keep_gaps:
                    cur_time += chunk_df.iloc[i-1]['gap']/speed_factor
                new_sub_times = []
                number = row['number']
                lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
                for line_index, line in enumerate(lines):
                    # Step2: Start speed change and save as OUTPUT_FILE_TEMPLATE
                    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
                    adjust_audio_speed(temp_file, output_file, speed_factor)
                    ad_dur = get_audio_duration(output_file)
                    new_sub_times.append([cur_time, cur_time+ad_dur])
                    cur_time += ad_dur
                # Step3: Find corresponding main DataFrame index and update new_sub_times
                main_df_idx = tasks_df[tasks_df['number'] == row['number']].index[0]
                tasks_df.at[main_df_idx, 'new_sub_times'] = new_sub_times
                # Step4: Choose emoji based on speed_factor and accept comparison
                emoji = "FAST" if speed_factor <= accept else "Warning:"
                rprint(f"[cyan]{emoji} Processed chunk {chunk_start} to {index} with speed factor {speed_factor}[/cyan]")
            # Step5: Check if the last row exceeds the range
            if cur_time > chunk_end_time:
                time_diff = cur_time - chunk_end_time
                if time_diff <= 0.6:  # If exceeding time is within 0.6 seconds, truncate the last audio
                    rprint(f"[yellow]Warning: Chunk {chunk_start} to {index} exceeds by {time_diff:.3f}s, truncating last audio[/yellow]")
                    # Get the last audio file
                    last_number = tasks_df.iloc[index]['number']
                    last_lines = eval(tasks_df.iloc[index]['lines']) if isinstance(tasks_df.iloc[index]['lines'], str) else tasks_df.iloc[index]['lines']
                    last_line_index = len(last_lines) - 1
                    last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{last_line_index}")
                    
                    # Calculate the duration to keep
                    audio = AudioSegment.from_wav(last_file)
                    original_duration = len(audio) / 1000  # Convert to seconds
                    new_duration = original_duration - time_diff
                    trimmed_audio = audio[:(new_duration * 1000)]  # pydub uses milliseconds
                    trimmed_audio.export(last_file, format="wav")
                    _ensure_non_empty_wav(last_file)
                    
                    # Update the last timestamp
                    last_times = tasks_df.at[index, 'new_sub_times']
                    last_times[-1][1] = chunk_end_time
                    tasks_df.at[index, 'new_sub_times'] = last_times
                else:
                    raise Exception(f"Chunk {chunk_start} to {index} exceeds the chunk end time {chunk_end_time:.2f} seconds with current time {cur_time:.2f} seconds")
            chunk_start = index+1
    
    rprint("[bold green]OK: Audio chunks processing completed![/bold green]")
    return tasks_df

def gen_audio() -> None:
    """Main function: Generate audio and process timeline"""
    rprint("[bold magenta]Starting audio generation process...[/bold magenta]")
    
    # Step1: Create necessary directories
    os.makedirs(_AUDIO_TMP_DIR, exist_ok=True)
    os.makedirs(_AUDIO_SEGS_DIR, exist_ok=True)
    
    # Step2: Load task file
    tasks_df = pd.read_excel(_8_1_AUDIO_TASK)
    rprint("[green]Loaded task file successfully[/green]")
    
    # Step3: Generate TTS audio
    tasks_df = generate_tts_audio(tasks_df)
    
    # Step4: Merge audio chunks
    tasks_df = merge_chunks(tasks_df)
    
    # Step4b: Normalize numpy scalars to builtin Python types before writing xlsx.
    # numpy 2.x repr of np.float64(x) is "np.float64(x)" (not plain "x"), which
    # breaks eval(cell) in _11_merge_audio.py if numpy is not imported there.
    # Cleaning at the source keeps xlsx cells portable (e.g. "[[0.24, 1.41]]").
    def _to_builtin(v):
        if isinstance(v, list):
            return [_to_builtin(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_to_builtin(x) for x in v)
        if isinstance(v, np.generic):
            return v.item()
        return v
    for _col in tasks_df.columns:
        if tasks_df[_col].dtype == object:
            tasks_df[_col] = tasks_df[_col].apply(_to_builtin)
    
    # Step5: Save results
    tasks_df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint("[bold green]Audio generation completed successfully![/bold green]")

if __name__ == "__main__":
    gen_audio()
