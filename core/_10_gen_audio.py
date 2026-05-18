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
        rprint(
            f"[dim]VAD checked TTS audio: {audio_file} no trim "
            f"(saved={max(0, saved_ms)}ms < {TTS_VAD_MIN_TOTAL_SAVED_MS}ms threshold, "
            f"speech_segs={len(segs)})[/dim]"
        )
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
        # silero-vad is the only TTS post-trimmer; any ImportError or model
        # failure propagates so the pipeline halts loudly instead of silently
        # producing un-trimmed audio.
        vad_compact_tts_audio(temp_file)
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
    """Merge audio chunks and adjust timeline.

    Two-pass design (ffmpeg/ffprobe used to be the wall-clock bottleneck because
    every chunk waited for the previous one to finish):
      Pass 0 (plan): walk tasks_df once, decide chunk boundaries / speed_factor
                     and collect every (temp -> output) ffmpeg job.
      Pass 1 (parallel I/O): run all ``adjust_audio_speed`` +
                     ``get_audio_duration`` calls through a ThreadPoolExecutor
                     and cache durations keyed by (number, line_index).
      Pass 2 (serial timeline): replay the original chunk loop but, instead of
                     spawning ffmpeg, look up cached durations and accumulate
                     ``cur_time``.  This step is pure arithmetic so a single
                     thread is plenty and the resulting timeline is bit-exact
                     identical to the legacy serial implementation.
    """
    rprint("[bold blue]Starting audio chunks processing...[/bold blue]")
    accept = load_key("speed_factor.accept")
    min_speed = load_key("speed_factor.min")

    tasks_df['new_sub_times'] = None

    # ── Pass 0: plan chunks and enumerate every ffmpeg job up front ─────────
    chunk_plans = []   # [{start_idx, end_idx, chunk_df, speed_factor, keep_gaps}, ...]
    jobs = []          # [(number, line_index, temp_file, output_file, speed_factor), ...]
    chunk_start = 0
    for index, row in tasks_df.iterrows():
        if row['cut_off'] != 1:
            continue
        chunk_df = tasks_df.iloc[chunk_start:index + 1].reset_index(drop=True)
        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)
        chunk_plans.append({
            'start_idx': chunk_start,
            'end_idx': index,
            'chunk_df': chunk_df,
            'speed_factor': speed_factor,
            'keep_gaps': keep_gaps,
        })
        for _, r in chunk_df.iterrows():
            number = r['number']
            lines = eval(r['lines']) if isinstance(r['lines'], str) else r['lines']
            for line_index, _line in enumerate(lines):
                jobs.append((
                    number,
                    line_index,
                    TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}"),
                    OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}"),
                    speed_factor,
                ))
        chunk_start = index + 1

    # ── Pass 1: parallel ffmpeg atempo + ffprobe duration probe ─────────────
    # ffmpeg_max_workers controls just this stage; falls back to the shared
    # ``max_workers`` (default 4) so existing configs keep working unchanged.
    max_workers = load_positive_int(
        "ffmpeg_max_workers", fallback_key="max_workers", default=4
    )
    durations: dict = {}

    def _do_one(job):
        number, line_index, temp_file, output_file, sf = job
        adjust_audio_speed(temp_file, output_file, sf)
        ad_dur = get_audio_duration(output_file)
        return (number, line_index), ad_dur

    if jobs:
        rprint(f"[blue]Adjusting {len(jobs)} audio segment(s) with {max_workers} parallel worker(s)...[/blue]")
        with Progress() as progress:
            task = progress.add_task("[cyan]Adjusting audio speed...", total=len(jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_do_one, j) for j in jobs]
                for future in as_completed(futures):
                    key, ad_dur = future.result()  # let exceptions surface
                    durations[key] = ad_dur
                    progress.advance(task)

    # ── Pass 2: serial timeline accumulation (dict lookups + arithmetic) ────
    for plan in chunk_plans:
        chunk_start = plan['start_idx']
        index = plan['end_idx']
        chunk_df = plan['chunk_df']
        speed_factor = plan['speed_factor']
        keep_gaps = plan['keep_gaps']

        # Step1: Start processing new timeline
        chunk_start_time = parse_df_srt_time(chunk_df.iloc[0]['start_time'])
        chunk_end_time = parse_df_srt_time(chunk_df.iloc[-1]['end_time']) + chunk_df.iloc[-1]['tolerance']  # 加上tolerance才是这一块的结束
        cur_time = chunk_start_time
        for i, row in chunk_df.iterrows():
            # If i is not 0, which is not the first row of the chunk, cur_time needs to be added with the gap of the previous row, remember to divide by speed_factor
            if i != 0 and keep_gaps:
                cur_time += chunk_df.iloc[i - 1]['gap'] / speed_factor
            new_sub_times = []
            number = row['number']
            lines = eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
            for line_index, _line in enumerate(lines):
                # Step2: Look up the duration produced by Pass 1 (ffmpeg already ran)
                ad_dur = durations[(number, line_index)]
                new_sub_times.append([cur_time, cur_time + ad_dur])
                cur_time += ad_dur
            # Step3: Find corresponding main DataFrame index and update new_sub_times
            main_df_idx = tasks_df[tasks_df['number'] == number].index[0]
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
