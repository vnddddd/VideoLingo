const startButton = document.getElementById("startButton");
const overlayButton = document.getElementById("overlayButton");
const statusEl = document.getElementById("status");
const bridgeStateEl = document.getElementById("bridgeState");
const videoTitleEl = document.getElementById("videoTitle");
const videoUrlEl = document.getElementById("videoUrl");

let activeTabId = null;
let refreshTimer = null;

function send(message) {
  return chrome.runtime.sendMessage(message);
}

function isYouTubeUrl(url) {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === "youtu.be" || host === "youtube.com" || host.endsWith(".youtube.com");
  } catch {
    return false;
  }
}

function videoIdFromUrl(url) {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();
    if (!(host === "youtu.be" || host === "youtube.com" || host.endsWith(".youtube.com"))) {
      return "";
    }

    const parts = parsed.pathname.split("/").filter(Boolean);
    if (host === "youtu.be") {
      return parts[0] || "";
    }
    if (parsed.pathname === "/watch") {
      return parsed.searchParams.get("v") || "";
    }
    if (parts.length >= 2 && ["shorts", "embed", "live"].includes(parts[0])) {
      return parts[1] || "";
    }
  } catch {
    return "";
  }
  return "";
}

function isYouTubeVideoUrl(url) {
  return Boolean(videoIdFromUrl(url));
}

function setStatus(text) {
  statusEl.textContent = text;
}

function phaseLabel(job, playbackMode = "none") {
  if (!job) {
    return "No cached dub for this video.";
  }
  if (job.status === "queued") {
    return job.queue_position ? `Queued. Position ${job.queue_position}.` : "Queued.";
  }
  if (job.status === "running") {
    if (job.phase === "audio_ready") {
      return "Running: reusing downloaded audio.";
    }
    const phase = job.phase ? job.phase.replaceAll("_", " ") : "running";
    return `Running: ${phase}`;
  }
  if (job.status === "done") {
    if (playbackMode === "original") {
      return "Original audio is on for this tab.";
    }
    return "Dub is on for this video.";
  }
  if (job.status === "error") {
    return `Error: ${job.error || "unknown error"}`;
  }
  return job.status || "Unknown";
}

function renderState(tab, job, playbackMode = "none") {
  activeTabId = tab?.id || null;
  const isYoutube = Boolean(tab?.url && isYouTubeUrl(tab.url));
  const isVideo = Boolean(tab?.url && isYouTubeVideoUrl(tab.url));
  videoTitleEl.textContent = isYoutube ? (tab.title || "YouTube video") : "No YouTube video detected";
  videoUrlEl.textContent = isYoutube ? tab.url : "";

  startButton.disabled = !isVideo || (job && ["queued", "running"].includes(job.status));
  overlayButton.disabled = !(job && job.status === "done" && isVideo);
  overlayButton.textContent = playbackMode === "original" ? "Use Dubbed Audio" : "Use Original Audio";
  setStatus(isVideo ? phaseLabel(job, playbackMode) : "Open a YouTube video first.");
}

async function checkBridge() {
  try {
    const response = await fetch("http://127.0.0.1:8765/health", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    bridgeStateEl.textContent = "Bridge online";
    return true;
  } catch {
    bridgeStateEl.textContent = "Bridge offline";
    return false;
  }
}

async function refresh() {
  await checkBridge();
  const state = await send({ type: "VIDEOLINGO_GET_STATE" });
  if (!state.ok) {
    setStatus(state.error || "Failed to read extension state");
    return;
  }

  let job = state.job || null;
  if (job && ["queued", "running"].includes(job.status)) {
    const polled = await send({ type: "VIDEOLINGO_POLL" });
    if (polled.ok) {
      job = polled.job || job;
    }
  }
  renderState(state.tab, job, state.playbackMode || "none");
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  setStatus("Sending YouTube URL and cookies to local bridge...");
  try {
    const result = await send({
      type: "VIDEOLINGO_START_CURRENT_TAB",
      tabId: activeTabId
    });
    if (!result.ok) {
      throw new Error(result.error || "Failed to start job");
    }
    renderState({ id: activeTabId, url: videoUrlEl.textContent, title: videoTitleEl.textContent }, result.job, "dub");
    await refresh();
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    await refresh();
  }
});

overlayButton.addEventListener("click", async () => {
  overlayButton.disabled = true;
  setStatus("Switching audio...");
  try {
    const result = await send({
      type: "VIDEOLINGO_TOGGLE_CURRENT_DUB",
      tabId: activeTabId
    });
    if (!result.ok) {
      throw new Error(result.error || "Failed to switch audio");
    }
    renderState({ id: activeTabId, url: videoUrlEl.textContent, title: videoTitleEl.textContent }, result.job, result.playbackMode || "none");
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    await refresh();
  }
});

refresh();
refreshTimer = setInterval(refresh, 5000);
window.addEventListener("unload", () => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
});
