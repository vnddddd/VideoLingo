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

function phaseName(phase) {
  const labels = {
    queued: "排队中",
    preparing: "准备中",
    audio_ready: "复用已下载音频",
    downloading_audio: "下载音频",
    done: "已完成",
    error: "出错"
  };
  return labels[phase] || (phase ? phase.replaceAll("_", " ") : "处理中");
}

function phaseLabel(job, playbackMode = "none") {
  if (!job) {
    return "这个视频还没有缓存的配音。";
  }
  if (job.status === "queued") {
    return job.queue_position ? `排队中，当前位置：${job.queue_position}。` : "排队中。";
  }
  if (job.status === "running") {
    if (job.phase === "audio_ready") {
      return "处理中：正在复用已下载的音频。";
    }
    return `处理中：${phaseName(job.phase)}`;
  }
  if (job.status === "done") {
    if (playbackMode === "original") {
      return "当前标签页正在使用原声。";
    }
    return "当前视频正在使用配音。";
  }
  if (job.status === "error") {
    return `出错：${job.error || "未知错误"}`;
  }
  return job.status || "未知状态";
}

function renderState(tab, job, playbackMode = "none") {
  activeTabId = tab?.id || null;
  const isYoutube = Boolean(tab?.url && isYouTubeUrl(tab.url));
  const isVideo = Boolean(tab?.url && isYouTubeVideoUrl(tab.url));
  videoTitleEl.textContent = isYoutube ? (tab.title || "YouTube 视频") : "未检测到 YouTube 视频";
  videoUrlEl.textContent = isYoutube ? tab.url : "";

  startButton.disabled = !isVideo || (job && ["queued", "running"].includes(job.status));
  overlayButton.disabled = !(job && job.status === "done" && isVideo);
  overlayButton.textContent = playbackMode === "original" ? "使用配音" : "使用原声";
  setStatus(isVideo ? phaseLabel(job, playbackMode) : "请先打开一个 YouTube 视频。");
}

async function checkBridge() {
  try {
    const response = await fetch("http://127.0.0.1:8765/health", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    bridgeStateEl.textContent = "服务在线";
    return true;
  } catch {
    bridgeStateEl.textContent = "服务离线";
    return false;
  }
}

async function refresh() {
  await checkBridge();
  const state = await send({ type: "VIDEOLINGO_GET_STATE" });
  if (!state.ok) {
    setStatus(state.error || "读取插件状态失败");
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
  setStatus("正在把 YouTube 链接和 Cookie 发送给本地服务...");
  try {
    const result = await send({
      type: "VIDEOLINGO_START_CURRENT_TAB",
      tabId: activeTabId
    });
    if (!result.ok) {
      throw new Error(result.error || "启动任务失败");
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
  setStatus("正在切换音频...");
  try {
    const result = await send({
      type: "VIDEOLINGO_TOGGLE_CURRENT_DUB",
      tabId: activeTabId
    });
    if (!result.ok) {
      throw new Error(result.error || "切换音频失败");
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
