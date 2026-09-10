"""Live and persisted stage timings for the Streamlit pipeline."""

import json

import streamlit as st

from core.utils.stage_timer import format_duration
from translations.translations import translate as t


def render_timing_panel(runner, key: str):
    data = runner.timing_snapshot()
    stages = data.get("stages", {})

    with st.expander(t("Step timings"), expanded=True):
        if not stages:
            st.caption(t("Timings will appear here when a new task starts. Earlier work cannot be timed retroactively."))
            return

        total = sum(float(entry.get("seconds", 0)) for entry in stages.values())
        media_seconds = float(data.get("media", {}).get("duration_seconds") or 0)
        columns = st.columns(3 if media_seconds > 0 else 1)
        columns[0].metric(t("Total processing time"), format_duration(total))
        if media_seconds > 0:
            columns[1].metric(t("Source duration"), format_duration(media_seconds))
            columns[2].metric(t("Processing time / source duration"), f"{total / media_seconds:.2f}x")

        status_labels = {
            "running": t("Running..."),
            "stopping": t("Stopping..."),
            "completed": t("Completed"),
            "stopped": t("Task stopped"),
            "error": t("Task error"),
        }
        recorded_label = t("Recorded")
        headers = {name: t(name) for name in ("Step", "Status", "Elapsed", "Share", "Runs")}
        rows = []
        for alias, entry in stages.items():
            seconds = float(entry.get("seconds", 0))
            rows.append({
                headers["Step"]: entry.get("label") or alias,
                headers["Status"]: status_labels.get(entry.get("status"), recorded_label),
                headers["Elapsed"]: format_duration(seconds),
                headers["Share"]: f"{seconds / total:.1%}" if total > 0 else "—",
                headers["Runs"]: int(entry.get("runs", 1)),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(t("Step times include retries; waiting between steps is excluded. Times use m:ss or h:mm:ss."))
        st.download_button(
            t("Download timing log"),
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="stage_timings.json",
            mime="application/json",
            key=f"{key}_timings_download",
            on_click="ignore",
        )
