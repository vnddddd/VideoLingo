(() => {
  if (window.__videolingoContentLoaded) {
    return;
  }
  window.__videolingoContentLoaded = true;

  const STATE = {
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
    originalVideoMuted: null,
    originalVideoVolume: null,
    lastUrl: location.href
  };

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

  async function syncAudio(forcePlay = false) {
    const video = STATE.video;
    const audio = STATE.audio;
    if (!video || !audio) {
      return;
    }

    enforceOriginalMuted();
    pauseOtherDubAudios(audio);
    audio.playbackRate = video.playbackRate || 1;
    const drift = Math.abs((audio.currentTime || 0) - (video.currentTime || 0));
    if (drift > 0.18 || forcePlay) {
      audio.currentTime = video.currentTime || 0;
    }

    if (video.paused) {
      pauseAllDubAudios();
      return;
    }

    try {
      await audio.play();
    } catch {
      // The next play/sync tick will retry after a user gesture.
    }
  }

  function enforceOriginalMuted() {
    const video = STATE.video;
    if (!video) {
      return;
    }
    if (!video.muted) {
      video.muted = true;
    }
    if (video.volume !== 0) {
      video.volume = 0;
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

  function stopCurrentOverlay() {
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
    if (STATE.video) {
      if (STATE.originalVideoMuted !== null) {
        STATE.video.muted = STATE.originalVideoMuted;
      }
      if (STATE.originalVideoVolume !== null) {
        STATE.video.volume = STATE.originalVideoVolume;
      }
    }
    if (STATE.subtitleEl) {
      STATE.subtitleEl.textContent = "";
      STATE.subtitleEl.style.display = "none";
    }
    STATE.jobId = null;
    STATE.cues = [];
    STATE.video = null;
    STATE.originalVideoMuted = null;
    STATE.originalVideoVolume = null;
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

    stopCurrentOverlay();
    const overlayVersion = STATE.overlayVersion;
    ensureOverlay();

    STATE.jobId = message.jobId;
    STATE.video = video;
    STATE.originalVideoMuted = video.muted;
    STATE.originalVideoVolume = video.volume;
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
    const audio = new Audio(audioObjectUrl);
    STATE.dubAudios.add(audio);
    STATE.audio = audio;
    STATE.audio.preload = "auto";
    STATE.audio.volume = 1;
    STATE.audio.playbackRate = video.playbackRate || 1;

    enforceOriginalMuted();

    addListener(video, "play", () => syncAudio(false));
    addListener(video, "pause", () => syncAudio(false));
    addListener(video, "seeking", () => syncAudio(false));
    addListener(video, "seeked", () => syncAudio(false));
    addListener(video, "playing", () => syncAudio(false));
    addListener(video, "ratechange", () => syncAudio(false));
    addListener(video, "volumechange", enforceOriginalMuted);
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

  function notifyUrlChanged() {
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
    setInterval(notifyUrlChanged, 1000);
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
        sendResponse({ ok: true });
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
