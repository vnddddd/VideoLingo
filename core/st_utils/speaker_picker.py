"""Streamlit UI: multi-speaker voice picker.

Rendered between ASR (`speaker-preview` stage) and the rest of the dubbing
pipeline whenever `core._3_speaker_preview` produced a manifest and is
awaiting user voice picks.

Public API
----------
    render_speaker_picker_if_pending() -> bool
        True  -> picker has been rendered; caller should NOT proceed with the
                 rest of the pipeline UI on this rerun.
        False -> no pending picks; caller resumes its normal flow.

On submit, persists `speaker_voice_map` into config.yaml via
`_sp.confirm_picks(...)` and triggers `st.rerun()` so the picker disappears
and the caller falls through to the next pipeline stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from core import _3_speaker_preview as _sp
from translations.translations import translate as t


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_NS = "_sp_picker_"                       # session_state key namespace
_K_MODE = _NS + "mode__"                  # per-speaker selected mode key prefix
_K_VOICE = _NS + "voice__"                # per-speaker fixed-voice input prefix

_MODE_FIXED = "fixed"
_MODE_CLONE = "clone"
_MODE_DEFAULT = "default"

_MODE_LABELS: Dict[str, str] = {
    _MODE_FIXED: "🎙️ Fixed voice (specify a TTS voice name)",
    _MODE_CLONE: "👥 Clone this speaker (use the preview clip as reference)",
    _MODE_DEFAULT: "🔇 Use default voice (no per-speaker switching)",
}
_MODE_ORDER: List[str] = [_MODE_FIXED, _MODE_CLONE, _MODE_DEFAULT]


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def render_speaker_picker_if_pending() -> bool:
    """Render the picker UI iff `_3_speaker_preview.is_pending()`.

    Returns
    -------
    bool
        True  -> picker rendered, caller should early-return.
        False -> nothing rendered, caller proceeds normally.
    """
    if not _sp.is_pending():
        return False

    manifest: List[Dict] = _sp.read_manifest()
    if not manifest:
        # Pending flag exists but manifest is unreadable -> offer escape hatch.
        st.error(
            t("Speaker preview is pending but manifest is missing or unreadable.")
        )
        if st.button(
            t("Discard preview state"),
            key=_NS + "discard_corrupt",
        ):
            _sp.reset()
            st.rerun()
        return True

    _render_header()
    picks, incomplete = _render_speaker_rows(manifest)
    _render_action_buttons(picks, incomplete)
    return True


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _render_header() -> None:
    st.subheader("🎤 " + t("Multi-speaker voice assignment"))
    st.info(
        t(
            "Pipeline is paused. Pick a voice strategy for each detected "
            "speaker, then click 'Continue dubbing' to resume."
        )
    )


def _render_speaker_rows(manifest: List[Dict]) -> tuple[Dict[str, Dict], bool]:
    """Render one row per speaker; return (picks_dict, incomplete_flag)."""
    picks: Dict[str, Dict] = {}
    incomplete = False

    for entry in manifest:
        speaker_id = str(entry.get("speaker_id", "")).strip()
        if not speaker_id:
            continue

        wav_path = entry.get("wav") or ""
        sample_text = entry.get("text") or ""
        duration = entry.get("duration")
        num_words = entry.get("num_words")

        with st.container(border=True):
            # Header line ------------------------------------------------------
            header_bits = [f"**Speaker `{speaker_id}`**"]
            if isinstance(duration, (int, float)):
                header_bits.append(f"{duration:.1f}s")
            if isinstance(num_words, int):
                header_bits.append(f"{num_words} " + t("words"))
            st.markdown(" · ".join(header_bits))

            # Audio preview ----------------------------------------------------
            if wav_path and Path(wav_path).exists():
                st.audio(wav_path)
            else:
                st.warning(t("Preview audio file missing: ") + wav_path)

            # Collapsible ASR sample ------------------------------------------
            if sample_text:
                with st.expander(t("Show recognized sample text")):
                    st.write(sample_text)

            # Mode picker ------------------------------------------------------
            mode_key = _K_MODE + speaker_id
            mode = st.radio(
                t("Voice strategy"),
                options=_MODE_ORDER,
                format_func=lambda m: t(_MODE_LABELS[m]),
                key=mode_key,
                horizontal=False,
            )

            entry_pick: Dict = {"mode": mode}

            if mode == _MODE_FIXED:
                voice_key = _K_VOICE + speaker_id
                voice = st.text_input(
                    t("TTS voice name"),
                    key=voice_key,
                    placeholder=t(
                        "e.g. zh-CN-XiaoxiaoNeural / alloy / default"
                    ),
                    help=t(
                        "Enter the voice identifier accepted by your current "
                        "TTS backend (see sidebar 'TTS Method')."
                    ),
                ).strip()
                if not voice:
                    incomplete = True
                    st.caption(
                        "⚠️ " + t("Voice name required for Fixed mode.")
                    )
                entry_pick["voice"] = voice

            elif mode == _MODE_CLONE:
                # Reference wav = this very preview clip.
                entry_pick["ref_wav"] = wav_path
                st.caption(
                    "ℹ️ "
                    + t(
                        "This preview clip will be used as the cloning "
                        "reference."
                    )
                )
            # _MODE_DEFAULT: no extra fields.

            picks[speaker_id] = entry_pick

    return picks, incomplete


def _render_action_buttons(picks: Dict[str, Dict], incomplete: bool) -> None:
    st.divider()
    col_submit, col_discard = st.columns([3, 1])

    with col_submit:
        if st.button(
            "✅ " + t("Continue dubbing"),
            key=_NS + "submit",
            type="primary",
            use_container_width=True,
            disabled=incomplete,
        ):
            try:
                _sp.confirm_picks(picks)
            except Exception as exc:  # noqa: BLE001 - surface any save error in UI
                st.error(f"{t('Failed to save picks')}: {exc}")
                return
            st.success(t("Voice picks saved. Resuming pipeline..."))
            st.rerun()

    with col_discard:
        if st.button(
            "🗑 " + t("Discard preview"),
            key=_NS + "discard",
            use_container_width=True,
            help=t(
                "Wipe the preview directory. Next pipeline run will re-detect "
                "speakers and regenerate previews."
            ),
        ):
            _sp.reset()
            st.rerun()
