(() => {
  if (window.__videolingoContentLoaded) {
    return;
  }
  window.__videolingoContentLoaded = true;

  const INSTANCE_ID = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const CONTENT_PROTOCOL_VERSION = 2;
  const SUPERSEDE_EVENT = "videolingo:supersede-content";
  const DUB_AUDIO_SELECTOR = 'audio[data-videolingo-dub-audio="1"]';

  const STATE = {
    active: true,
    jobId: null,
    audio: null,
    audioObjectUrl: null,
    cues: [],
    subtitleEl: null,
    timer: null,
    video: null,
    listeners: [],
    dubAudios: new Set(),
    overlayVersion: 0,
    originalVideoStates: new Map(),
    videoWaiting: false,
    lastVideoTime: null,
    lastVideoCheckMs: 0,
    videoStalledSinceMs: null,
    urlWatchTimer: null,
    lastUrl: location.href
  };

  document.addEventListener(SUPERSEDE_EVENT, (event) => {
    if (event.detail?.instanceId && event.detail.instanceId !== INSTANCE_ID) {
      deactivateInstance();
    }
  });
  document.dispatchEvent(new CustomEvent(SUPERSEDE_EVENT, {
    detail: { instanceId: INSTANCE_ID }
  }));

  function findVideo() {
    return document.querySelector("video.html5-main-video") || document.querySelector("video");
  }

  function parseTime(value) {
    const match = String(value).trim().match(/^(\d+):(\d{2}):(\d{2})[,.](\d{1,3})$/);
    if (!match) {
      return 0;
    }
    const [, hh, mm, ss, ms] = match;
    return (
      Number(hh) * 3600 +
      Number(mm) * 60 +
      Number(ss) +
      Number(ms.padEnd(3, "0")) / 1000
    );
  }

  function parseSrt(text) {
    return text
      .replace(/\r/g, "")
      .trim()
      .split(/\n{2,}/)
      .map((block) => {
        const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
        const timeLineIndex = lines.findIndex((line) => line.includes("-->"));
        if (timeLineIndex < 0) {
          return null;
        }
        const [start, end] = lines[timeLineIndex].split("-->").map((part) => part.trim());
        return {
          start: parseTime(start),
          end: parseTime(end),
          text: lines.slice(timeLineIndex + 1).join("\n")
        };
      })
      .filter((cue) => cue && cue.text && cue.end > cue.start);
  }

  function ensureStyles() {
    if (document.getElementById("videolingo-overlay-style")) {
      return;
    }
    const style = document.createElement("style");
    style.id = "videolingo-overlay-style";
    style.textContent = `
      #videolingo-subtitle {
        position: fixed;
        left: 50%;
        bottom: 84px;
        transform: translateX(-50%);
        max-width: min(86vw, 980px);
        padding: 0 10px;
        color: #fff;
        font: 800 56px/1.18 Arial, "Microsoft YaHei", "PingFang SC", sans-serif;
        letter-spacing: 0;
        text-align: center;
        white-space: pre-line;
        -webkit-text-stroke: 2.6px rgba(0, 0, 0, 0.96);
        paint-order: stroke fill;
        text-shadow:
          0 2px 2px rgba(0, 0, 0, 0.95),
          2px 0 2px rgba(0, 0, 0, 0.95),
          -2px 0 2px rgba(0, 0, 0, 0.95),
          0 -2px 2px rgba(0, 0, 0, 0.95),
          0 4px 8px rgba(0, 0, 0, 0.8);
        pointer-events: none;
        z-index: 2147483646;
        display: none;
      }
      @media (max-width: 680px) {
        #videolingo-subtitle {
          max-width: 94vw;
          font-size: 34px;
          -webkit-text-stroke-width: 1.8px;
          bottom: 68px;
        }
      }
    `;
    document.documentElement.appendChild(style);
  }

  function ensureOverlay() {
    ensureStyles();
    if (!STATE.subtitleEl) {
      STATE.subtitleEl = document.createElement("div");
      STATE.subtitleEl.id = "videolingo-subtitle";
      document.body.appendChild(STATE.subtitleEl);
    }
  }

  function removeForeignDubAudios(currentAudio = null) {
    for (const audio of document.querySelectorAll(DUB_AUDIO_SELECTOR)) {
      if (audio === currentAudio) {
        continue;
      }
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      audio.remove();
    }
  }

  function removeListeners() {
    for (const [target, event, handler, options] of STATE.listeners) {
      target.removeEventListener(event, handler, options);
    }
    STATE.listeners = [];
  }

  function addListener(target, event, handler, options) {
    target.addEventListener(event, handler, options);
    STATE.listeners.push([target, event, handler, options]);
  }

  function cueAt(time) {
    let low = 0;
    let high = STATE.cues.length - 1;
    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const cue = STATE.cues[mid];
      if (time < cue.start) {
        high = mid - 1;
      } else if (time > cue.end) {
        low = mid + 1;
      } else {
        return cue;
      }
    }
    return null;
  }

  function updateSubtitle() {
    if (!STATE.video || !STATE.subtitleEl) {
      return;
    }
    const cue = cueAt(STATE.video.currentTime || 0);
    if (cue) {
      STATE.subtitleEl.textContent = cue.text;
      STATE.subtitleEl.style.display = "block";
    } else {
      STATE.subtitleEl.textContent = "";
      STATE.subtitleEl.style.display = "none";
    }
  }

  function updatePosition() {
    if (!STATE.video || !STATE.subtitleEl) {
      return;
    }
    const rect = STATE.video.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    STATE.subtitleEl.style.left = `${rect.left + rect.width / 2}px`;
    STATE.subtitleEl.style.maxWidth = `${Math.max(220, rect.width * 0.9)}px`;
    STATE.subtitleEl.style.bottom = `${Math.max(18, window.innerHeight - rect.bottom + 58)}px`;
  }

  function allVideoElements() {
    const videos = new Set(document.querySelectorAll("video"));
    if (STATE.video) {
      videos.add(STATE.video);
    }
    return Array.from(videos);
  }

  function resetVideoProgressWatch(video = STATE.video) {
    STATE.lastVideoTime = video ? video.currentTime || 0 : null;
    STATE.lastVideoCheckMs = performance.now();
    STATE.videoStalledSinceMs = null;
  }

  function isVideoTimeStalled(video) {
    const now = performance.now();
    const currentTime = video.currentTime || 0;
    if (STATE.lastVideoTime === null || !STATE.lastVideoCheckMs) {
      STATE.lastVideoTime = currentTime;
      STATE.lastVideoCheckMs = now;
      STATE.videoStalledSinceMs = null;
      return false;
    }

    if (Math.abs(currentTime - STATE.lastVideoTime) > 0.03) {
      STATE.lastVideoTime = currentTime;
      STATE.lastVideoCheckMs = now;
      STATE.videoStalledSinceMs = null;
      return false;
    }

    if (now - STATE.lastVideoCheckMs < 300) {
      return false;
    }

    if (STATE.videoStalledSinceMs === null) {
      STATE.videoStalledSinceMs = STATE.lastVideoCheckMs;
    }
    return now - STATE.videoStalledSinceMs >= 300;
  }

  function shouldPauseDubForVideo(video) {
    if (video.paused || video.ended || video.seeking) {
      return true;
    }
    if (STATE.videoWaiting) {
      return true;
    }
    if (video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
      return true;
    }
    return isVideoTimeStalled(video);
  }

  function markVideoBuffering() {
    STATE.videoWaiting = true;
    pauseAllDubAudios();
  }

  function markVideoReady() {
    STATE.videoWaiting = false;
    resetVideoProgressWatch();
    syncAudio(false);
  }

  async function syncAudio(forcePlay = false) {
    if (!STATE.active) {
      return;
    }
    const video = STATE.video;
    const audio = STATE.audio;
    if (!video || !audio) {
      return;
    }

    removeForeignDubAudios(audio);
    enforceOriginalMuted();
    pauseOtherDubAudios(audio);
    audio.playbackRate = video.playbackRate || 1;

    if (shouldPauseDubForVideo(video)) {
      audio.currentTime = video.currentTime || 0;
      pauseAllDubAudios();
      return;
    }

    const drift = Math.abs((audio.currentTime || 0) - (video.currentTime || 0));
    if (drift > 0.18 || forcePlay) {
      audio.currentTime = video.currentTime || 0;
    }

    try {
      await audio.play();
      if (shouldPauseDubForVideo(video)) {
        audio.currentTime = video.currentTime || 0;
        pauseAllDubAudios();
      }
    } catch {
      // The next play/sync tick will retry after a user gesture.
    }
  }

  function enforceOriginalMuted() {
    for (const video of allVideoElements()) {
      if (!STATE.originalVideoStates.has(video)) {
        STATE.originalVideoStates.set(video, {
          muted: video.muted,
          volume: video.volume,
          defaultMuted: video.defaultMuted,
          hadMutedAttribute: video.hasAttribute("muted")
        });
      }
      if (!video.muted) {
        video.muted = true;
      }
      if (video.volume !== 0) {
        video.volume = 0;
      }
      video.defaultMuted = true;
      video.setAttribute("muted", "");
    }
  }

  function restoreOriginalVideos() {
    for (const [video, state] of STATE.originalVideoStates.entries()) {
      try {
        video.muted = state.muted;
        video.volume = state.volume;
        video.defaultMuted = state.defaultMuted;
        if (state.hadMutedAttribute) {
          video.setAttribute("muted", "");
        } else {
          video.removeAttribute("muted");
        }
      } catch {
      }
    }
    STATE.originalVideoStates.clear();
  }

  function enforceOriginalMutedOnMediaEvent(event) {
    if (event.target && event.target.tagName === "VIDEO") {
      enforceOriginalMuted();
    }
  }

  function blockYoutubeVolumeKeys(event) {
    if (!STATE.audio) {
      return;
    }
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      enforceOriginalMuted();
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }

  function pauseOtherDubAudios(currentAudio) {
    for (const audio of STATE.dubAudios) {
      if (audio !== currentAudio) {
        audio.pause();
      }
    }
  }

  function pauseAllDubAudios() {
    for (const audio of STATE.dubAudios) {
      audio.pause();
    }
  }

  function destroyAudio(audio) {
    if (!audio) {
      return;
    }
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
    audio.remove();
    STATE.dubAudios.delete(audio);
  }

  function stopCurrentOverlay({ restoreOriginal = true } = {}) {
    STATE.overlayVersion += 1;
    removeListeners();
    if (STATE.timer) {
      clearInterval(STATE.timer);
      STATE.timer = null;
    }
    if (STATE.audio) {
      destroyAudio(STATE.audio);
      STATE.audio = null;
    }
    for (const audio of Array.from(STATE.dubAudios)) {
      destroyAudio(audio);
    }
    if (STATE.audioObjectUrl) {
      URL.revokeObjectURL(STATE.audioObjectUrl);
      STATE.audioObjectUrl = null;
    }
    if (restoreOriginal) {
      restoreOriginalVideos();
    }
    if (STATE.subtitleEl) {
      STATE.subtitleEl.textContent = "";
      STATE.subtitleEl.style.display = "none";
    }
    STATE.jobId = null;
    STATE.cues = [];
    STATE.video = null;
    STATE.videoWaiting = false;
    resetVideoProgressWatch(null);
  }

  function deactivateInstance() {
    if (!STATE.active) {
      return;
    }
    STATE.active = false;
    if (STATE.urlWatchTimer) {
      clearInterval(STATE.urlWatchTimer);
      STATE.urlWatchTimer = null;
    }
    stopCurrentOverlay({ restoreOriginal: true });
  }

  async function fetchBlobUrl(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to fetch audio: HTTP ${response.status}`);
    }
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  }

  async function fetchText(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to fetch subtitles: HTTP ${response.status}`);
    }
    return response.text();
  }

  async function applyOverlay(message) {
    if (!STATE.active) {
      return;
    }
    const video = findVideo();
    if (!video) {
      throw new Error("No YouTube video element found");
    }

    if (STATE.jobId === message.jobId && STATE.audio && STATE.video === video) {
      updatePosition();
      updateSubtitle();
      await syncAudio(!video.paused);
      return;
    }

    stopCurrentOverlay({ restoreOriginal: false });
    const overlayVersion = STATE.overlayVersion;
    ensureOverlay();
    removeForeignDubAudios();

    STATE.jobId = message.jobId;
    STATE.video = video;
    const subtitleText = await fetchText(message.subtitleUrl);
    if (overlayVersion !== STATE.overlayVersion) {
      return;
    }
    const audioObjectUrl = await fetchBlobUrl(message.audioUrl);
    if (overlayVersion !== STATE.overlayVersion) {
      URL.revokeObjectURL(audioObjectUrl);
      return;
    }

    STATE.cues = parseSrt(subtitleText);
    STATE.audioObjectUrl = audioObjectUrl;
    const audio = document.createElement("audio");
    audio.dataset.videolingoDubAudio = "1";
    audio.dataset.videolingoInstance = INSTANCE_ID;
    audio.src = audioObjectUrl;
    audio.style.display = "none";
    document.documentElement.appendChild(audio);
    STATE.dubAudios.add(audio);
    STATE.audio = audio;
    STATE.audio.preload = "auto";
    STATE.audio.volume = 1;
    STATE.audio.playbackRate = video.playbackRate || 1;
    resetVideoProgressWatch(video);

    enforceOriginalMuted();

    addListener(video, "play", markVideoReady);
    addListener(video, "pause", () => syncAudio(false));
    addListener(video, "seeking", () => {
      resetVideoProgressWatch();
      syncAudio(false);
    });
    addListener(video, "seeked", markVideoReady);
    addListener(video, "waiting", markVideoBuffering);
    addListener(video, "stalled", markVideoBuffering);
    addListener(video, "emptied", markVideoBuffering);
    addListener(video, "error", markVideoBuffering);
    addListener(video, "playing", markVideoReady);
    addListener(video, "canplay", markVideoReady);
    addListener(video, "canplaythrough", markVideoReady);
    addListener(video, "timeupdate", markVideoReady);
    addListener(video, "ratechange", () => syncAudio(false));
    addListener(video, "volumechange", enforceOriginalMuted);
    addListener(document, "play", enforceOriginalMutedOnMediaEvent, true);
    addListener(document, "playing", enforceOriginalMutedOnMediaEvent, true);
    addListener(document, "volumechange", enforceOriginalMutedOnMediaEvent, true);
    addListener(window, "resize", updatePosition);
    addListener(window, "scroll", updatePosition);
    addListener(window, "keydown", blockYoutubeVolumeKeys, true);

    STATE.timer = setInterval(() => {
      enforceOriginalMuted();
      updatePosition();
      updateSubtitle();
      syncAudio(false);
    }, 150);

    updatePosition();
    updateSubtitle();
    await syncAudio(!video.paused);
  }

  function currentOverlayState() {
    const video = findVideo();
    const hasCurrentAudio = Boolean(STATE.audio && STATE.dubAudios.has(STATE.audio));
    if (STATE.active && hasCurrentAudio) {
      removeForeignDubAudios(STATE.audio);
      enforceOriginalMuted();
      updatePosition();
      updateSubtitle();
    }
    return {
      ok: STATE.active,
      protocolVersion: CONTENT_PROTOCOL_VERSION,
      instanceId: INSTANCE_ID,
      jobId: STATE.jobId,
      hasAudio: hasCurrentAudio,
      sameVideo: Boolean(video && STATE.video === video)
    };
  }

  function notifyUrlChanged() {
    if (!STATE.active) {
      return;
    }
    if (STATE.lastUrl === location.href) {
      return;
    }
    STATE.lastUrl = location.href;
    stopCurrentOverlay();
    chrome.runtime.sendMessage({
      type: "VIDEOLINGO_URL_CHANGED",
      url: location.href
    }).catch(() => {});
  }

  function watchUrlChanges() {
    STATE.urlWatchTimer = setInterval(notifyUrlChanged, 1000);
    window.addEventListener("yt-navigate-finish", notifyUrlChanged);
    window.addEventListener("popstate", notifyUrlChanged);

    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;
    history.pushState = function pushState(...args) {
      const result = originalPushState.apply(this, args);
      setTimeout(notifyUrlChanged, 0);
      return result;
    };
    history.replaceState = function replaceState(...args) {
      const result = originalReplaceState.apply(this, args);
      setTimeout(notifyUrlChanged, 0);
      return result;
    };
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    (async () => {
      if (message.type === "PING") {
        sendResponse({ ok: STATE.active, protocolVersion: CONTENT_PROTOCOL_VERSION, instanceId: INSTANCE_ID });
        return;
      }
      if (message.type === "VIDEOLINGO_APPLY_OVERLAY") {
        await applyOverlay(message);
        sendResponse({ ok: true });
        return;
      }
      if (message.type === "VIDEOLINGO_STOP_OVERLAY") {
        stopCurrentOverlay();
        sendResponse({ ok: true });
        return;
      }
      if (message.type === "VIDEOLINGO_GET_OVERLAY_STATE") {
        sendResponse(currentOverlayState());
        return;
      }
      sendResponse({ ok: false, error: "Unknown message type" });
    })().catch((error) => {
      sendResponse({ ok: false, error: error.message || String(error) });
    });
    return true;
  });

  watchUrlChanges();
  chrome.runtime.sendMessage({
    type: "VIDEOLINGO_URL_CHANGED",
    url: location.href,
    initial: true
  }).catch(() => {});
})();
