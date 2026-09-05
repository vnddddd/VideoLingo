import streamlit as st
import os, sys, time
from core.st_utils.imports_and_utils import *
from core.st_utils.task_runner import StopTask, TaskRunner
from core.st_utils.speaker_picker import render_speaker_picker_if_pending
from core import *
from core import _3_speaker_preview as _speaker_preview

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
DUB_RAW_AUDIO = "output/dub.mp3"
DUB_AUDIO = "output/dub_loudnorm.mp3"
DUB_SUBTITLE = "output/dub.srt"
TRANS_SUBTITLE = "output/trans.srt"
RAW_AUDIO = "output/audio/raw.mp3"


def _has_source_video() -> bool:
    try:
        return os.path.exists(_1_ytdlp.find_video_files())
    except Exception:
        return False


def _is_audio_only_source() -> bool:
    return os.path.exists(RAW_AUDIO) and not _has_source_video()


def _render_final_video_enabled() -> bool:
    """Return the persisted output mode, keeping old configs backward compatible."""
    try:
        value = load_key("render_final_video")
    except Exception:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _should_render_video() -> bool:
    """Whether the current run should create a subtitle/dubbed video file."""
    return not _is_audio_only_source() and _render_final_video_enabled()


def _subtitle_outputs_complete() -> bool:
    if not _should_render_video():
        return os.path.exists(TRANS_SUBTITLE)
    return os.path.exists(SUB_VIDEO)


def _dubbing_outputs_complete() -> bool:
    standalone_outputs_ready = os.path.exists(DUB_AUDIO) and os.path.exists(DUB_SUBTITLE)
    if not _should_render_video():
        return standalone_outputs_ready
    return standalone_outputs_ready and os.path.exists(DUB_VIDEO)


def _steps_markdown(steps) -> str:
    """Render the actual pipeline steps so skipped video stages are not shown."""
    return "<br>".join(
        f"{index}. {label}" for index, (label, _func) in enumerate(steps, start=1)
    )


def _download_output_file(path: str, label: str, mime: str, key: str) -> bool:
    """Add a download button for an output file when it exists."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as file:
        data = file.read()
    st.download_button(
        label=label,
        data=data,
        file_name=os.path.basename(path),
        mime=mime,
        key=key,
        use_container_width=True,
    )
    return True


# ─── Kickoff memo: remember last pipeline start so the speaker picker can auto-resume ───
def _kickoff(runner_key: str, steps_provider):
    """Start a TaskRunner and remember how it was started.

    If the pipeline halts at the multi-speaker picker, ``confirm_picks`` in the
    picker UI sets ``_resume_after_picker``; on the next rerun, ``main()`` reads
    this memo and replays the same start call so the user doesn't have to click
    the start button a second time.
    """
    runner = TaskRunner.get(st.session_state, runner_key)
    steps = steps_provider()
    st.session_state["_last_kickoff"] = {
        "runner_key": runner_key,
        "steps_provider": steps_provider,
    }
    runner.start(steps)
    st.rerun()


def _resume_after_picker_if_needed() -> None:
    """Replay the previous _kickoff after the speaker picker is confirmed.

    Triggered when:
      - picker is no longer pending (already cleared by confirm_picks)
      - _resume_after_picker flag is set in session_state
      - a previous _last_kickoff memo is available
    """
    if not st.session_state.pop("_resume_after_picker", False):
        return
    info = st.session_state.get("_last_kickoff")
    if not info:
        return
    try:
        runner = TaskRunner.get(st.session_state, info["runner_key"])
        steps = info["steps_provider"]()
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"{t('Voice picks saved, but auto-resume failed; please click the start button again.')}"
            f" ({exc})"
        )
        return
    runner.start(steps)
    st.rerun()


def _speaker_preview_inproc_step():
    """In-process speaker preview step for the text-pipeline runner.

    Runs only when multi-speaker is enabled in config. Writes the .pending
    flag plus per-speaker wav/txt previews under output/preview/, then
    raises StopTask so the runner halts cleanly and main() can render the
    picker UI on the next Streamlit rerun.
    """
    if not _speaker_preview.is_required():
        return
    _speaker_preview.generate_previews()
    if _speaker_preview.is_pending():
        raise StopTask(
            "Speaker preview pending: pick voices in the UI before continuing."
        )


# ─── Task control UI (auto-refreshes every 1s while task is active) ───


@st.fragment(run_every=1)
def _task_control_panel(runner_key: str):
    """Renders progress bar + pause/stop buttons. Auto-refreshes every 1s."""
    runner = TaskRunner.get(st.session_state, runner_key)

    if runner.state == "idle":
        return

    # Progress
    step_text = (
        f"({runner.current_step + 1}/{runner.total_steps}) {runner.current_label}"
        if runner.current_step >= 0
        else ""
    )

    if runner.is_active:
        if runner.state == "paused":
            st.warning(f"⏸️ {t('Paused')} {step_text}")
        else:
            st.info(f"⏳ {t('Running...')} {step_text}")
        st.progress(runner.progress)

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if runner.state == "paused":
                if st.button(
                    f"▶️ {t('Resume')}",
                    key=f"{runner_key}_resume",
                    use_container_width=True,
                ):
                    runner.resume()
                    st.rerun()
            else:
                if st.button(
                    f"⏸️ {t('Pause')}",
                    key=f"{runner_key}_pause",
                    use_container_width=True,
                ):
                    runner.pause()
                    st.rerun()
        with col2:
            if st.button(
                f"⏹️ {t('Stop')}",
                key=f"{runner_key}_stop",
                use_container_width=True,
                type="primary",
            ):
                runner.stop()
                st.rerun()

    elif runner.state == "completed":
        st.success(t("Task completed!"))
        st.progress(1.0)
        runner.reset()
        time.sleep(0.5)
        st.rerun(scope="app")

    elif runner.state == "stopped":
        st.warning(f"⏹️ {t('Task stopped')} {step_text}")
        if st.button(t("OK"), key=f"{runner_key}_ack_stop", use_container_width=True):
            runner.reset()
            st.rerun(scope="app")

    elif runner.state == "error":
        st.error(f"❌ {t('Task error')}: {runner.error_msg}")
        if st.button(t("OK"), key=f"{runner_key}_ack_error", use_container_width=True):
            runner.reset()
            st.rerun(scope="app")


# ─── Translation and dubbing ───


def _get_text_steps():
    """Return subtitle translation steps as (label, callable) pairs."""
    steps = [
        (t("WhisperX word-level transcription"), _2_asr.transcribe),
        (t("Speaker preview for multi-speaker picker"), _speaker_preview_inproc_step),
        (
            t("Sentence segmentation using NLP and LLM"),
            lambda: (
                _3_1_split_nlp.split_by_spacy(),
                _3_2_split_meaning.split_sentences_by_meaning(),
            ),
        ),
        (
            t("Summarization and multi-step translation"),
            lambda: (_4_1_summarize.get_summary(), _4_2_translate.translate_all()),
        ),
        (
            t("Cutting and aligning long subtitles"),
            lambda: (
                _5_split_sub.split_for_sub_main(),
                _6_gen_sub.align_timestamp_main(),
            ),
        ),
    ]
    if _should_render_video():
        steps.append(
            (
                t("Merging subtitles into the video"),
                _7_sub_into_vid.merge_subtitles_to_video,
            )
        )
    return steps


def _get_audio_steps():
    """Return dubbing steps as (label, callable) pairs."""
    steps = [
        (
            t("Generate audio tasks and chunks"),
            lambda: (
                _8_1_audio_task.gen_audio_task_main(),
                _8_2_dub_chunks.gen_dub_chunks(),
            ),
        ),
        (t("Extract reference audio"), _9_refer_audio.extract_refer_audio_main),
        (t("Generate and merge audio files"), _10_gen_audio.gen_audio),
        (t("Merge full audio"), _11_merge_audio.merge_full_audio),
        (t("Normalize final dubbed audio"), _11_merge_audio.normalize_dub_audio),
    ]
    if _should_render_video():
        steps.append((t("Merge final audio into video"), _12_dub_to_vid.merge_video_audio))
    return steps


def _get_translation_dubbing_steps():
    """Build one continuous workflow, skipping stages whose outputs already exist."""
    steps = []
    subtitle_ready = _subtitle_outputs_complete()
    if not subtitle_ready:
        steps.extend(_get_text_steps())
    if not _dubbing_outputs_complete():
        # A run from an older version may already have the raw dub and subtitle.
        # In separate-output mode, only add the new loudness pass instead of
        # repeating all TTS work.
        raw_dub_ready = os.path.exists(DUB_RAW_AUDIO) and os.path.exists(DUB_SUBTITLE)
        if subtitle_ready and raw_dub_ready:
            if not os.path.exists(DUB_AUDIO):
                steps.append(
                    (t("Normalize final dubbed audio"), _11_merge_audio.normalize_dub_audio)
                )
            if _should_render_video() and not os.path.exists(DUB_VIDEO):
                steps.append(
                    (t("Merge final audio into video"), _12_dub_to_vid.merge_video_audio)
                )
        else:
            steps.extend(_get_audio_steps())
    return steps


def _render_subtitle_outputs() -> bool:
    """Render/download subtitle results when the subtitle stage is complete."""
    if not _subtitle_outputs_complete():
        return False

    if _should_render_video():
        if load_key("burn_subtitles") and os.path.exists(SUB_VIDEO):
            st.video(SUB_VIDEO)
    else:
        st.success(t("Subtitle processing is complete!"))
        _download_output_file(
            TRANS_SUBTITLE,
            t("Download translated subtitle"),
            "application/x-subrip",
            "download_translated_subtitle",
        )
    return True


def _render_dubbing_outputs() -> bool:
    """Render/download dubbing results when the audio stage is complete."""
    if not _dubbing_outputs_complete():
        return False

    if _should_render_video():
        st.success(
            t(
                "Audio processing is complete! You can check the audio files in the `output` folder."
            )
        )
        if load_key("burn_subtitles") and os.path.exists(DUB_VIDEO):
            st.video(DUB_VIDEO)
    else:
        st.success(t("Standalone subtitles and audio are ready in the output folder."))

    st.audio(DUB_AUDIO)
    audio_col, subtitle_col = st.columns(2)
    with audio_col:
        _download_output_file(
            DUB_AUDIO,
            t("Download dubbed audio"),
            "audio/mpeg",
            "download_dubbed_audio",
        )
    with subtitle_col:
        _download_output_file(
            DUB_SUBTITLE,
            t("Download dubbed subtitle"),
            "application/x-subrip",
            "download_dubbed_subtitle",
        )
    return True


def translation_dubbing_section():
    """Show one button and one task runner for translation, subtitles, and dubbing."""
    st.header(t("b. Translate, Generate Subtitles and Dub"))
    runner = TaskRunner.get(st.session_state, "_translation_runner")
    steps = _get_translation_dubbing_steps()

    with st.container(border=True):
        if steps:
            st.markdown(
                f"""
            <p style='font-size: 20px;'>
            {t("This stage includes the following steps:")}
            <p style='font-size: 20px;'>
                {_steps_markdown(steps)}
            """,
                unsafe_allow_html=True,
            )
        else:
            st.success(t("Translation and dubbing are complete!"))

        if runner.is_active or runner.is_done:
            _task_control_panel("_translation_runner")
        elif steps:
            if st.button(
                t("Start Translation and Dubbing"),
                key="translation_dubbing_button",
            ):
                _kickoff("_translation_runner", _get_translation_dubbing_steps)

        subtitle_ready = _render_subtitle_outputs()
        audio_ready = _render_dubbing_outputs()

        if subtitle_ready or audio_ready:
            download_subtitle_zip_button(text=t("Download All Srt Files"))

        if audio_ready and st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
            delete_dubbing_files()
            st.rerun()
        if subtitle_ready and audio_ready and st.button(
            t("Archive to 'history'"), key="cleanup_in_translation_dubbing"
        ):
            cleanup()
            st.rerun()


# ─── Main ───


def main():
    logo_col, _ = st.columns([1, 1])
    with logo_col:
        st.image("docs/logo.png", width="stretch")
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t(
        'Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href="https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh" target="_blank">here</a>! You can also try out our SaaS website at <a href="https://videolingo.io" target="_blank">videolingo.io</a> for free!'
    )
    st.markdown(
        f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>",
        unsafe_allow_html=True,
    )
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    download_video_section()
    # Multi-speaker picker: if a speaker preview is pending, render the picker
    # in place of the pipeline section so the user can audition each speaker
    # and pick a voice / clone / default.
    # confirm_picks() clears the pending flag; users then press the start
    # button again to resume the pipeline.
    if render_speaker_picker_if_pending():
        return
    _resume_after_picker_if_needed()
    translation_dubbing_section()


if __name__ == "__main__":
    main()
