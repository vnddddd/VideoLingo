const BRIDGE_ORIGIN = "http://127.0.0.1:8765";
const POLL_ALARM = "videolingo-poll";
const tabDubPrefs = new Map();

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

function clearTabPreference(tabId) {
  tabDubPrefs.delete(tabId);
}

function isDubDisabledForTab(tabId, videoId) {
  const pref = tabDubPrefs.get(tabId);
  if (!pref) {
    return false;
  }
  if (pref.videoId !== videoId) {
    tabDubPrefs.delete(tabId);
    return false;
  }
  return pref.disabled === true;
}

function setDubDisabledForTab(tabId, videoId, disabled) {
  if (disabled) {
    tabDubPrefs.set(tabId, { videoId, disabled: true });
    return;
  }
  const pref = tabDubPrefs.get(tabId);
  if (!pref || pref.videoId === videoId) {
    tabDubPrefs.delete(tabId);
  }
}

async function activeTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0] || null;
}

async function getYouTubeCookies() {
  const urls = [
    "https://www.youtube.com/",
    "https://youtube.com/",
    "https://m.youtube.com/",
    "https://accounts.google.com/",
    "https://www.google.com/"
  ];
  const byKey = new Map();
  for (const url of urls) {
    const cookies = await chrome.cookies.getAll({ url });
    for (const cookie of cookies) {
      const key = `${cookie.domain}\n${cookie.path}\n${cookie.name}`;
      byKey.set(key, cookie);
    }
  }
  return Array.from(byKey.values());
}

async function lookupUrl(url) {
  const response = await fetch(`${BRIDGE_ORIGIN}/lookup?url=${encodeURIComponent(url)}`, {
    cache: "no-store"
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Bridge returned HTTP ${response.status}`);
  }
  return body;
}

async function fetchJob(jobId) {
  const response = await fetch(`${BRIDGE_ORIGIN}/jobs/${jobId}`, { cache: "no-store" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Bridge returned HTTP ${response.status}`);
  }
  return body;
}

function absoluteOutputUrls(job) {
  if (!job || !job.outputs) {
    return null;
  }
  return {
    audioUrl: new URL(job.outputs.audio, BRIDGE_ORIGIN).href,
    subtitleUrl: new URL(job.outputs.subtitle, BRIDGE_ORIGIN).href,
    logUrl: new URL(job.outputs.log, BRIDGE_ORIGIN).href
  };
}

async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "PING" });
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content.js"]
    });
  }
}

async function stopOverlayToTab(tabId) {
  await ensureContentScript(tabId);
  await chrome.tabs.sendMessage(tabId, { type: "VIDEOLINGO_STOP_OVERLAY" });
}

async function applyOverlayToTab(tabId, job) {
  const urls = absoluteOutputUrls(job);
  if (!urls) {
    throw new Error("Job outputs are not ready");
  }
  await ensureContentScript(tabId);
  await chrome.tabs.sendMessage(tabId, {
    type: "VIDEOLINGO_APPLY_OVERLAY",
    jobId: job.id,
    ...urls
  });
}

async function syncTabWithCache(tabId, url) {
  const videoId = url ? videoIdFromUrl(url) : "";
  if (!tabId || !url || !videoId) {
    if (tabId) {
      clearTabPreference(tabId);
      await stopOverlayToTab(tabId).catch(() => {});
    }
    return null;
  }

  let lookup;
  try {
    lookup = await lookupUrl(url);
  } catch {
    return null;
  }

  const job = lookup.job || null;
  if (job && job.status === "done") {
    if (isDubDisabledForTab(tabId, videoId)) {
      await stopOverlayToTab(tabId).catch(() => {});
    } else {
      await applyOverlayToTab(tabId, job).catch(() => {});
    }
  } else {
    await stopOverlayToTab(tabId).catch(() => {});
  }
  return job;
}

async function syncActiveYouTubeTabs() {
  const tabs = await chrome.tabs.query({ active: true });
  await Promise.allSettled(
    tabs
      .filter((tab) => tab.id && tab.url && isYouTubeUrl(tab.url))
      .map((tab) => syncTabWithCache(tab.id, tab.url))
  );
}

async function jobForTab(tab) {
  if (!tab || !tab.url || !isYouTubeVideoUrl(tab.url)) {
    return null;
  }
  const lookup = await lookupUrl(tab.url);
  return lookup.job || null;
}

async function startCurrentTabJob(tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : await activeTab();
  if (!tab || !tab.url || !isYouTubeVideoUrl(tab.url)) {
    throw new Error("Open a YouTube video tab first");
  }

  const cookies = await getYouTubeCookies();
  const response = await fetch(`${BRIDGE_ORIGIN}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url: tab.url,
      title: tab.title || "",
      cookies
    })
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Bridge returned HTTP ${response.status}`);
  }

  const job = { ...body, tabId: tab.id, pageUrl: tab.url, pageTitle: tab.title || "" };
  await chrome.alarms.create(POLL_ALARM, { periodInMinutes: 1 });
  if (job.status === "done") {
    setDubDisabledForTab(tab.id, videoIdFromUrl(tab.url), false);
    await applyOverlayToTab(tab.id, job).catch(() => {});
  } else {
    await stopOverlayToTab(tab.id).catch(() => {});
  }
  return job;
}

async function pollCurrentTab(tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : await activeTab();
  if (!tab || !tab.url || !isYouTubeVideoUrl(tab.url)) {
    return null;
  }
  let job = await jobForTab(tab);
  if (job && ["queued", "running"].includes(job.status)) {
    job = await fetchJob(job.id);
  }
  if (job && job.status === "done" && tab.id) {
    const videoId = videoIdFromUrl(tab.url);
    if (isDubDisabledForTab(tab.id, videoId)) {
      await stopOverlayToTab(tab.id).catch(() => {});
    } else {
      await applyOverlayToTab(tab.id, job).catch(() => {});
    }
  }
  return job;
}

async function toggleCurrentDub(tabId) {
  const tab = tabId ? await chrome.tabs.get(tabId) : await activeTab();
  if (!tab || !tab.id || !tab.url || !isYouTubeVideoUrl(tab.url)) {
    throw new Error("Open a YouTube video tab first");
  }
  const job = await jobForTab(tab);
  if (!job || job.status !== "done") {
    throw new Error("This video has no completed VideoLingo dub yet");
  }
  const videoId = videoIdFromUrl(tab.url);
  const disabled = isDubDisabledForTab(tab.id, videoId);
  if (disabled) {
    setDubDisabledForTab(tab.id, videoId, false);
    await applyOverlayToTab(tab.id, job);
    return { job, playbackMode: "dub" };
  }
  setDubDisabledForTab(tab.id, videoId, true);
  await stopOverlayToTab(tab.id);
  return { job, playbackMode: "original" };
}

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === POLL_ALARM) {
    syncActiveYouTubeTabs().catch(() => {});
  }
});

chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId).then((tab) => {
    if (tab.url && isYouTubeUrl(tab.url)) {
      syncTabWithCache(tabId, tab.url).catch(() => {});
    }
  }).catch(() => {});
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const url = changeInfo.url || tab.url || "";
  if (changeInfo.status === "loading") {
    clearTabPreference(tabId);
  }
  if ((changeInfo.url || changeInfo.status === "complete") && isYouTubeUrl(url)) {
    syncTabWithCache(tabId, url).catch(() => {});
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  clearTabPreference(tabId);
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    if (message.type === "VIDEOLINGO_URL_CHANGED") {
      const tabId = sender.tab?.id;
      if (tabId && message.url) {
        if (message.initial) {
          clearTabPreference(tabId);
        }
        const job = await syncTabWithCache(tabId, message.url);
        sendResponse({ ok: true, job });
        return;
      }
      sendResponse({ ok: true, job: null });
      return;
    }
    if (message.type === "VIDEOLINGO_GET_STATE") {
      const tab = await activeTab();
      let job = null;
      let playbackMode = "none";
      if (tab && tab.url && isYouTubeVideoUrl(tab.url)) {
        job = await jobForTab(tab).catch(() => null);
        if (job && job.status === "done") {
          playbackMode = isDubDisabledForTab(tab.id, videoIdFromUrl(tab.url)) ? "original" : "dub";
        }
      }
      sendResponse({ ok: true, tab, job, playbackMode });
      return;
    }
    if (message.type === "VIDEOLINGO_START_CURRENT_TAB") {
      const job = await startCurrentTabJob(message.tabId);
      sendResponse({ ok: true, job });
      return;
    }
    if (message.type === "VIDEOLINGO_POLL") {
      const job = await pollCurrentTab(message.tabId);
      sendResponse({ ok: true, job });
      return;
    }
    if (message.type === "VIDEOLINGO_TOGGLE_CURRENT_DUB") {
      const result = await toggleCurrentDub(message.tabId);
      sendResponse({ ok: true, ...result });
      return;
    }
    if (message.type === "VIDEOLINGO_CLEAR_JOB") {
      const tab = await activeTab();
      if (tab?.id) {
        await stopOverlayToTab(tab.id).catch(() => {});
      }
      sendResponse({ ok: true });
      return;
    }
    sendResponse({ ok: false, error: "Unknown message type" });
  })().catch((error) => {
    sendResponse({ ok: false, error: error.message || String(error) });
  });
  return true;
});

chrome.alarms.create(POLL_ALARM, { periodInMinutes: 1 }).catch(() => {});
