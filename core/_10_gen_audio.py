import os
import re
import time
import shutil
import subprocess
import threading
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pydub import AudioSegment
from rich.console import Console
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils import *
from core.utils.models import *
from core.asr_backend.audio_preprocess import get_audio_duration
from core.prompts import get_subtitle_trim_prompt
from core.tts_backend.tts_main import tts_main

console = Console()

TEMP_FILE_TEMPLATE = f"{_AUDIO_TMP_DIR}/{{}}_temp.wav"
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"
WARMUP_SIZE = 5
SAFE_TIMELINE_OVERRUN_SECONDS = 1.0
from core.tts_backend.tts_vad import (
    MIN_SEGMENT_DURATION_MS,
    ensure_non_empty_wav as _ensure_non_empty_wav,
)
from core.tts_backend.soniox_tts import (
    clamp_speed as _clamp_soniox_speed,
    configured_speed as _soniox_configured_speed,
)

# Backend whose TTS can render a line at a chosen speaking rate. ffmpeg atempo
# stretches an already-rendered wav and degrades as the factor grows, so when a
# line does not fit we would rather re-render it faster than stretch it harder.
NATIVE_SPEED_TTS_METHOD = 'soniox_tts'

# Soniox allows only 3 concurrent REST requests, while the chunk-adjust stage
# runs with ffmpeg_max_workers (12 by default). Gate the TTS calls alone so
# ffmpeg keeps its parallelism and we never trip the provider's limit.
NATIVE_SPEED_MAX_CONCURRENCY = 3
_native_speed_semaphore = threading.Semaphore(NATIVE_SPEED_MAX_CONCURRENCY)


def native_speed_enabled() -> bool:
    """True when the configured backend can render at a chosen speaking rate."""
    return load_key("tts_method") == NATIVE_SPEED_TTS_METHOD


def render_line_at_native_speed(
    text: str,
    temp_file: str,
    output_file: str,
    speed_factor: float,
    number,
    tasks_df: pd.DataFrame,
    speaker_id,
) -> bool:
    """Re-render a line at the target speaking rate instead of atempo-stretching it.

    ffmpeg atempo resamples already-rendered audio and its artefacts grow with
    the factor; asking the model to speak at that rate instead yields natural
    speech. When the rate needed is beyond what the backend accepts, the backend
    takes what it can and ffmpeg covers the small remainder.

    Costs one extra TTS call per adjusted line, so callers should only reach for
    it when the backend actually supports it.

    Returns False when there is nothing the backend can do, leaving the caller
    to fall back to ffmpeg.
    """
    if abs(speed_factor - 1.0) < 0.001:
        return False

    base_speed = _soniox_configured_speed()
    native_speed = _clamp_soniox_speed(base_speed * speed_factor)
    # Clamped straight back to the baseline: nothing gained by re-rendering.
    if native_speed is None or abs(native_speed - base_speed) < 0.001:
        return False

    # What plain ffmpeg would have produced from the take we already have.
    target_duration = get_audio_duration(temp_file) / speed_factor
    if target_duration <= 0:
        return False

    # tts_main treats an existing file as a finished resume artefact and returns
    # without regenerating, so clear it before asking for a new rate.
    if os.path.exists(output_file):
        os.remove(output_file)
    with _native_speed_semaphore:
        tts_main(text, output_file, number, tasks_df, speaker_id=speaker_id, speed=native_speed)
    _ensure_non_empty_wav(output_file)

    # Rate control tracks the ratio closely (measured 1.325x for speed 1.3), but
    # each render varies by a few percent, and anything past the backend's cap
    # is only partly covered by the rate anyway. So measure what came back
    # rather than trusting the ratio, and let ffmpeg close the gap. Coming up
    # short is fine -- the timeline is built from measured durations either way.
    actual = get_audio_duration(output_file)
    if actual > target_duration:
        residual_file = f"{output_file}.residual.tmp.wav"
        try:
            adjust_audio_speed(output_file, residual_file, actual / target_duration)
            os.replace(residual_file, output_file)
        finally:
            if os.path.exists(residual_file):
                os.remove(residual_file)
    return True


class AudioFitTooFastError(Exception):
    """Raised when fitting audio would exceed speed_factor.max."""

    def __init__(self, input_file: str, needed_speed: float, max_speed: float):
        super().__init__(
            f"Cannot fit audio segment {input_file}: needs speed factor "
            f"{needed_speed:.3f}, configured max is {max_speed:.3f}"
        )
        self.input_file = input_file
        self.needed_speed = needed_speed
        self.max_speed = max_speed

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
            # If ffmpeg leaves a short clip slightly long, retry with a refined
            # speed factor. Do not hard-trim the tail; that can cut speech.
            if output_duration >= expected_duration * 1.02 and input_duration < 3 and diff <= 0.1:
                refined_speed_factor = speed_factor * (output_duration / expected_duration)
                refined_cmd = ['ffmpeg', '-i', input_file, '-filter:a', f'atempo={refined_speed_factor}', '-y', output_file]
                subprocess.run(refined_cmd, check=True, stderr=subprocess.PIPE)
                _ensure_non_empty_wav(output_file)
                print(f"Refined speed adjustment to expected duration: {expected_duration:.2f} seconds")
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

def fit_audio_to_duration(input_file: str, target_duration: float, current_speed_factor: float) -> float:
    """Speed up one generated segment so the chunk can fit its target timeline."""
    min_duration = MIN_SEGMENT_DURATION_MS / 1000
    if target_duration <= min_duration:
        raise Exception(
            f"Cannot fit audio segment {input_file}: target duration "
            f"{target_duration:.3f}s is too short"
        )

    original_duration = get_audio_duration(input_file)
    if original_duration <= target_duration:
        return original_duration

    extra_speed_factor = original_duration / target_duration
    max_speed_factor = float(load_key("speed_factor.max"))
    effective_speed_factor = current_speed_factor * extra_speed_factor
    if effective_speed_factor > max_speed_factor:
        raise AudioFitTooFastError(input_file, effective_speed_factor, max_speed_factor)

    temp_file = f"{input_file}.fit.tmp.wav"
    try:
        adjust_audio_speed(input_file, temp_file, extra_speed_factor)
        os.replace(temp_file, input_file)
        _ensure_non_empty_wav(input_file)
        return get_audio_duration(input_file)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def parse_lines_value(lines):
    return eval(lines) if isinstance(lines, str) else list(lines)


def get_row_speaker_id(row: pd.Series):
    if 'speaker_id' not in row.index:
        return None
    speaker_id = row['speaker_id']
    if speaker_id is None or (isinstance(speaker_id, float) and pd.isna(speaker_id)):
        return None
    return speaker_id


def shorten_text_for_audio_fit(text: str, target_duration: float) -> str:
    rprint(
        f"[yellow]Audio still needs more than speed_factor.max; asking LLM to "
        f"shorten subtitle for {target_duration:.2f}s raw TTS[/yellow]"
    )
    prompt = get_subtitle_trim_prompt(text, target_duration)

    def valid_trim(response):
        if 'result' not in response or not str(response['result']).strip():
            return {'status': 'error', 'message': 'No result in response'}
        return {'status': 'success', 'message': ''}

    try:
        response = ask_gpt(prompt, resp_type='json', log_title='audio_fit_sub_trim', valid_def=valid_trim)
        shortened_text = str(response['result']).strip()
    except Exception:
        rprint("[bold red]LLM shortening failed; falling back to punctuation cleanup[/bold red]")
        shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()

    rprint(f"[green]Subtitle shortened for audio fit:[/green] {text} -> {shortened_text}")
    return shortened_text or text


def is_short_unshrinkable_text(text: str) -> bool:
    compact = re.sub(r'[^\w\u4e00-\u9fff]', '', str(text))
    return len(compact) <= 6


def regenerate_adjusted_line(tasks_df: pd.DataFrame, row_index: int, line_index: int, text: str, speed_factor: float, speed: Optional[float] = None) -> str:
    row = tasks_df.iloc[row_index]
    number = row['number']
    temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
    output_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
    for path in (temp_file, output_file):
        if os.path.exists(path):
            os.remove(path)

    tts_main(text, temp_file, number, tasks_df, speaker_id=get_row_speaker_id(row), speed=speed)
    _ensure_non_empty_wav(temp_file)
    adjust_audio_speed(temp_file, output_file, speed_factor)

    lines = parse_lines_value(tasks_df.at[row_index, 'lines'])
    real_dur = 0
    for idx in range(len(lines)):
        line_temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{idx}")
        if os.path.exists(line_temp_file):
            real_dur += get_audio_duration(line_temp_file)
    tasks_df.at[row_index, 'real_dur'] = real_dur
    return output_file


def try_native_speed_refit(
    tasks_df: pd.DataFrame,
    row_index: int,
    line_index: int,
    output_file: str,
    target_duration: float,
    speed_factor: float,
) -> Optional[str]:
    """Re-render one line at a faster native TTS rate instead of stretching it.

    Only worth doing when ffmpeg alone cannot reach the target within
    speed_factor.max: the alternative there is an LLM rewrite that drops words,
    whereas re-rendering keeps the line intact.

    The native rate is deliberately NOT counted against speed_factor.max. That
    budget exists to bound ffmpeg atempo, which resamples an already-rendered
    wav and adds artefacts as it grows; asking the model to simply speak faster
    produces natural speech with none of that. Folding the two together would
    keep the end-to-end ratio constant and make this refit a no-op.

    Returns the re-rendered output file, or None when the backend has no native
    rate control or has no headroom left.
    """
    if load_key("tts_method") != NATIVE_SPEED_TTS_METHOD:
        return None

    duration = get_audio_duration(output_file)
    if duration <= target_duration:
        return None

    base_speed = _soniox_configured_speed()
    native_speed = _clamp_soniox_speed(base_speed * (duration / target_duration))
    # Already at the backend's cap, or the configured baseline sits above it.
    if native_speed is None or native_speed <= base_speed + 0.001:
        return None

    lines = parse_lines_value(tasks_df.at[row_index, 'lines'])
    rprint(
        f"[cyan]Re-rendering line at native TTS speed {native_speed:.2f} "
        f"(baseline {base_speed:.2f}) to fit {target_duration:.2f}s[/cyan]"
    )
    return regenerate_adjusted_line(
        tasks_df,
        row_index,
        line_index,
        str(lines[line_index]),
        speed_factor,
        speed=native_speed,
    )


def fit_or_shorten_line(
    tasks_df: pd.DataFrame,
    row_index: int,
    line_index: int,
    output_file: str,
    target_duration: float,
    speed_factor: float,
    max_rewrites: int = 2,
) -> float:
    max_speed_factor = float(load_key("speed_factor.max"))
    min_duration = MIN_SEGMENT_DURATION_MS / 1000
    lines = parse_lines_value(tasks_df.at[row_index, 'lines'])
    original_text = str(lines[line_index])
    duration = get_audio_duration(output_file)
    overrun = duration - target_duration

    if target_duration <= min_duration:
        if overrun <= SAFE_TIMELINE_OVERRUN_SECONDS:
            rprint(
                f"[yellow]Audio target is too short ({target_duration:.3f}s); keeping current audio "
                f"and allowing {overrun:.3f}s overrun: {original_text}[/yellow]"
            )
            return duration
        raise Exception(
            f"Cannot fit audio segment {output_file}: target duration "
            f"{target_duration:.3f}s is too short and overrun {overrun:.3f}s exceeds "
            f"{SAFE_TIMELINE_OVERRUN_SECONDS:.3f}s"
        )

    raw_target_duration = target_duration * max_speed_factor * 0.95

    # Give native rate control first refusal, but only once ffmpeg alone is out
    # of reach: it costs an extra TTS call, so lines that already fit must not
    # pay for it. On success the line keeps every word that an LLM rewrite would
    # otherwise have dropped.
    if duration > target_duration and speed_factor * (duration / target_duration) > max_speed_factor:
        refit_file = try_native_speed_refit(
            tasks_df, row_index, line_index, output_file, target_duration, speed_factor
        )
        if refit_file is not None:
            output_file = refit_file

    for rewrite_attempt in range(max_rewrites + 1):
        try:
            return fit_audio_to_duration(output_file, target_duration, speed_factor)
        except AudioFitTooFastError:
            duration = get_audio_duration(output_file)
            overrun = duration - target_duration
            if overrun <= SAFE_TIMELINE_OVERRUN_SECONDS:
                rprint(
                    f"[yellow]Audio exceeds target by {overrun:.3f}s; keeping current audio "
                    f"and allowing timeline overrun: {original_text}[/yellow]"
                )
                return duration

            if rewrite_attempt >= max_rewrites:
                raise

            shortened_text = shorten_text_for_audio_fit(original_text, raw_target_duration)
            if shortened_text == original_text:
                if is_short_unshrinkable_text(original_text):
                    rprint(
                        f"[yellow]Short subtitle cannot be shortened safely; keeping current audio "
                        f"and allowing {overrun:.3f}s overrun: {original_text}[/yellow]"
                    )
                    return duration
                raise

            lines[line_index] = shortened_text
            tasks_df.at[row_index, 'lines'] = lines
            tasks_df.at[row_index, 'text'] = ' '.join(str(line) for line in lines)
            output_file = regenerate_adjusted_line(
                tasks_df,
                row_index,
                line_index,
                shortened_text,
                speed_factor,
            )

    return get_audio_duration(output_file)


def process_row(row: pd.Series, tasks_df: pd.DataFrame) -> Tuple[int, float]:
    """Helper function for processing single row data"""
    number = row['number']
    lines = parse_lines_value(row['lines'])
    # 🎙️ multi-speaker (plan_multispeaker C4-S3): forward per-row speaker_id to
    # tts_main so the router can pick the right voice/method. Column is created
    # by _8_1_audio_task.py from the sidecar; absent or NaN ⇒ legacy single-voice.
    speaker_id = get_row_speaker_id(row)
    real_dur = 0
    for line_index, line in enumerate(lines):
        temp_file = TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}")
        tts_main(line, temp_file, number, tasks_df, speaker_id=speaker_id)
        # tts_main applies the mandatory silero-vad post-trim only after
        # raw-output BadTTS duration/retry/fallback decisions have completed.
        # The duration recorded here is the post-VAD duration used by merge_chunks
        # to calculate ffmpeg speed adjustment later.
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

    max_speed = float(load_key("speed_factor.max"))
    speed_factor = min(speed_factor, max_speed)
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
            lines = parse_lines_value(r['lines'])
            speaker_id = get_row_speaker_id(r)
            for line_index, line in enumerate(lines):
                jobs.append((
                    number,
                    line_index,
                    TEMP_FILE_TEMPLATE.format(f"{number}_{line_index}"),
                    OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}"),
                    speed_factor,
                    str(line),
                    speaker_id,
                ))
        chunk_start = index + 1

    # ── Pass 1: parallel ffmpeg atempo + ffprobe duration probe ─────────────
    # ffmpeg_max_workers controls just this stage; falls back to the shared
    # ``max_workers`` (default 4) so existing configs keep working unchanged.
    max_workers = load_positive_int(
        "ffmpeg_max_workers", fallback_key="max_workers", default=4
    )
    durations: dict = {}
    # Resolved once: every job asks the same question and load_key re-reads config.
    use_native_speed = native_speed_enabled()

    def _do_one(job):
        number, line_index, temp_file, output_file, sf, text, speaker_id = job
        handled = False
        if use_native_speed:
            try:
                handled = render_line_at_native_speed(
                    text, temp_file, output_file, sf, number, tasks_df, speaker_id
                )
            except Exception as e:  # noqa: BLE001 - never let this break the pipeline
                rprint(
                    f"[yellow]Native-speed render failed for {number}_{line_index} "
                    f"({type(e).__name__}: {str(e)[:120]}); falling back to ffmpeg[/yellow]"
                )
                handled = False
        if not handled:
            adjust_audio_speed(temp_file, output_file, sf)
        ad_dur = get_audio_duration(output_file)
        return (number, line_index), ad_dur

    if jobs:
        how = "native TTS speed" if use_native_speed else "ffmpeg atempo"
        rprint(f"[blue]Adjusting {len(jobs)} audio segment(s) via {how} with {max_workers} parallel worker(s)...[/blue]")
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
            lines = parse_lines_value(row['lines'])
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
            last_number = tasks_df.iloc[index]['number']
            last_lines = parse_lines_value(tasks_df.iloc[index]['lines'])
            last_line_index = len(last_lines) - 1
            last_file = OUTPUT_FILE_TEMPLATE.format(f"{last_number}_{last_line_index}")

            audio = AudioSegment.from_wav(last_file)
            original_duration = len(audio) / 1000
            target_duration = original_duration - time_diff
            rprint(
                f"[yellow]Warning: Chunk {chunk_start} to {index} exceeds by "
                f"{time_diff:.3f}s, fitting last audio with speed-up[/yellow]"
            )
            final_duration = fit_or_shorten_line(
                tasks_df,
                index,
                last_line_index,
                last_file,
                target_duration,
                speed_factor,
            )

            # Update the last timestamp
            last_times = tasks_df.at[index, 'new_sub_times']
            last_times[-1][1] = last_times[-1][0] + final_duration
            tasks_df.at[index, 'new_sub_times'] = last_times

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
