import streamlit as st
import os, sys, time, subprocess
from core.st_utils.imports_and_utils import *
from core.st_utils.task_runner import TaskRunner, get_current_runner
from core import *

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
SPLIT_RENDER_ZIP = "output/render_inputs.zip"


def _run_split_pipeline_command(*args: str) -> str:
    """Run tools/split_pipeline.py from the project root and let logs stream to the server terminal."""
    command = [sys.executable, "tools/split_pipeline.py", *args]
    print("$ " + " ".join(command), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        command,
        cwd=current_dir,
        env=env,
    )
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"split_pipeline failed with exit code {returncode}")
    return ""


def _run_split_pipeline_status_command() -> str:
    """Run split pipeline status and return captured output for the UI."""
    command = [sys.executable, "tools/split_pipeline.py", "status"]
    print("$ " + " ".join(command), flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    env.setdefault("PYTHONUNBUFFERED", "1")

    result = subprocess.run(
        command,
        cwd=current_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        raise RuntimeError(f"split_pipeline status failed with exit code {result.returncode}\n{output}")
    return output


def _split_step(label: str, *args: str):
    """Build a TaskRunner step for a split pipeline CLI command."""
    return (label, lambda: _run_split_pipeline_command(*args))


def _split_local_resume_steps():
    """Expose local-stop-before-video as guarded sub-steps for pause/resume UI."""
    return [
        _split_step(t("WhisperX word-level transcription"), "local-step", "asr"),
        _split_step(t("Sentence segmentation using NLP and LLM"), "local-step", "split"),
        _split_step(t("Summarization and multi-step translation"), "local-step", "translate"),
        _split_step(t("Cut and align long subtitles"), "local-step", "subtitles"),
        _split_step(t("Generate timeline and subtitles"), "local-step", "timeline"),
        _split_step(t("Generate audio tasks and chunks"), "local-step", "audio-tasks"),
        _split_step(t("Extract reference audio"), "local-step", "reference-audio"),
        _split_step(t("Generate audio and merge into dub.mp3/dub.srt"), "local-step", "tts-merge"),
    ]


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


# ─── Text processing ───


def _get_text_steps():
    """Return the subtitle processing steps as (label, callable) list."""
    steps = [
        (t("WhisperX word-level transcription"), _2_asr.transcribe),
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
        (
            t("Merging subtitles into the video"),
            _7_sub_into_vid.merge_subtitles_to_video,
        ),
    ]
    return steps


def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    runner = TaskRunner.get(st.session_state, "_text_runner")

    with st.container(border=True):
        st.markdown(
            f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("WhisperX word-level transcription")}<br>
            2. {t("Sentence segmentation using NLP and LLM")}<br>
            3. {t("Summarization and multi-step translation")}<br>
            4. {t("Cutting and aligning long subtitles")}<br>
            5. {t("Generating timeline and subtitles")}<br>
            6. {t("Merging subtitles into the video")}
        """,
            unsafe_allow_html=True,
        )

        if not os.path.exists(SUB_VIDEO):
            if runner.is_active:
                _task_control_panel("_text_runner")
            elif runner.is_done:
                _task_control_panel("_text_runner")
            else:
                if st.button(
                    t("Start Processing Subtitles"), key="text_processing_button"
                ):
                    steps = _get_text_steps()
                    runner.start(steps)
                    st.rerun()
        else:
            if load_key("burn_subtitles"):
                st.video(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))

            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True


# ─── Audio processing ───


def _get_audio_steps():
    """Return the audio/dubbing processing steps as (label, callable) list."""
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
        (t("Merge final audio into video"), _12_dub_to_vid.merge_video_audio),
    ]
    return steps


def split_pipeline_section():
    st.header(t("d. Split Local/Remote Pipeline"))
    runner = TaskRunner.get(st.session_state, "_split_pipeline_runner")

    with st.container(border=True):
        st.markdown(
            f"""
        <p style='font-size: 20px;'>
        {t("Use these advanced actions when audio separation/final rendering must run on another machine.")}
        <p style='font-size: 20px;'>
            1. {t("Remote/GPU machine: prepare raw, vocal and background audio")}<br>
            2. {t("Local machine: run subtitles and dubbing, stopping before final video render")}<br>
            3. {t("Create a render input zip for the remote/GPU machine")}<br>
            4. {t("Remote/GPU machine: render the final dubbed video")}
        """,
            unsafe_allow_html=True,
        )

        status_col, zip_col = st.columns(2)
        with status_col:
            if st.button(t("Check Split Pipeline Status"), key="split_pipeline_status", use_container_width=True):
                st.code(_run_split_pipeline_status_command() or t("No status output."))
        with zip_col:
            st.caption(t("Default render package path: output/render_inputs.zip"))

        if runner.is_active or runner.is_done:
            _task_control_panel("_split_pipeline_runner")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("1. Prepare Audio on Remote/GPU"), key="split_pipeline_prep_audio", use_container_width=True):
                    runner.start([_split_step(t("Prepare split pipeline audio"), "prep-audio")])
                    st.rerun()
                if st.button(t("3. Package Render Inputs"), key="split_pipeline_pack", use_container_width=True):
                    runner.start([_split_step(t("Package render inputs"), "pack-render-inputs", "--zip", SPLIT_RENDER_ZIP)])
                    st.rerun()
            with col2:
                if st.button(t("2. Run Local Steps Until Audio"), key="split_pipeline_local_until_audio", use_container_width=True):
                    runner.start(_split_local_resume_steps())
                    st.rerun()
                if st.button(t("4. Render Final Video on Remote/GPU"), key="split_pipeline_remote_render", use_container_width=True):
                    runner.start([_split_step(t("Render final dubbed video"), "remote-render")])
                    st.rerun()


def audio_processing_section():
    st.header(t("c. Dubbing"))
    runner = TaskRunner.get(st.session_state, "_audio_runner")

    with st.container(border=True):
        st.markdown(
            f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """,
            unsafe_allow_html=True,
        )

        if not os.path.exists(DUB_VIDEO):
            if runner.is_active:
                _task_control_panel("_audio_runner")
            elif runner.is_done:
                _task_control_panel("_audio_runner")
            else:
                if st.button(
                    t("Start Audio Processing"), key="audio_processing_button"
                ):
                    steps = _get_audio_steps()
                    runner.start(steps)
                    st.rerun()
        else:
            st.success(
                t(
                    "Audio processing is complete! You can check the audio files in the `output` folder."
                )
            )
            if load_key("burn_subtitles"):
                st.video(DUB_VIDEO)
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
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
    text_processing_section()
    audio_processing_section()
    split_pipeline_section()


if __name__ == "__main__":
    main()
