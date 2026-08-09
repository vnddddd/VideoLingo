const BRIDGE_ORIGIN = "http://127.0.0.1:8765";

const startButton = document.getElementById("startButton");
const overlayButton = document.getElementById("overlayButton");
const statusEl = document.getElementById("status");
const bridgeStateEl = document.getElementById("bridgeState");
const videoTitleEl = document.getElementById("videoTitle");
const videoUrlEl = document.getElementById("videoUrl");
const configStatusEl = document.getElementById("configStatus");
const speakerPickerEl = document.getElementById("speakerPicker");
const speakerPickerCountEl = document.getElementById("speakerPickerCount");
const speakerRowsEl = document.getElementById("speakerRows");
const speakerContinueButton = document.getElementById("speakerContinueButton");
const speakerPickerStatusEl = document.getElementById("speakerPickerStatus");

let activeTabId = null;
let refreshTimer = null;
let currentConfigValues = {};
let configLoaded = false;
let applyingConfig = false;
let saveTimer = null;
let saveInFlight = false;
let saveAgain = false;
let activeSpeakerJobId = null;
let speakerManifest = [];
let restoringDetailsState = false;

const DETAILS_STATE_STORAGE_KEY = "videolingoPopupDetailsState";

const INTEGER_KEYS = new Set([
  "api.max_workers",
  "whisper.max_workers",
  "tts_max_workers",
  "gpt_sovits.refer_mode"
]);
const FLOAT_KEYS = new Set(["indextts2.emo_weight", "soniox_tts.speed"]);
const LIST_TEXT_KEYS = new Set(["indextts2.base_url"]);

const OPTIONS = {
  "display_language": [
    ["en", "EN English"],
    ["zh-CN", "CN 简体中文"],
    ["zh-HK", "HK 繁体中文"],
    ["ja", "JA 日本語"],
    ["es", "ES Español"],
    ["ru", "RU Русский"],
    ["fr", "FR Français"]
  ],
  "whisper.language": [
    ["en", "us English"],
    ["zh", "简体中文"],
    ["es", "Español"],
    ["ru", "Русский"],
    ["fr", "Français"],
    ["de", "Deutsch"],
    ["it", "Italiano"],
    ["ja", "日本語"]
  ],
  "whisper.runtime": [
    ["local", "local"],
    ["cloud", "cloud"],
    ["elevenlabs", "elevenlabs"],
    ["soniox", "soniox"]
  ],
  "demucs_backend": [
    ["local", "local"],
    ["hf_space", "hf_space"]
  ],
  "tts_method": [
    ["azure_tts", "azure_tts"],
    ["openai_tts", "openai_tts"],
    ["qwen3_tts", "qwen3_tts"],
    ["soniox_tts", "soniox_tts"],
    ["fish_tts", "fish_tts"],
    ["sf_fish_tts", "sf_fish_tts"],
    ["edge_tts", "edge_tts"],
    ["gpt_sovits", "gpt_sovits"],
    ["custom_tts", "custom_tts"],
    ["sf_cosyvoice2", "sf_cosyvoice2"],
    ["f5tts", "f5tts"],
    ["mimo_tts", "mimo_tts"],
    ["indextts2", "indextts2"]
  ],
  "soniox_tts.model": [
    ["tts-rt-v1", "tts-rt-v1（官方正式版，28 音色）"],
    ["tts-rt-v2", "tts-rt-v2（71 音色，官方文档尚未公布）"]
  ],
  "soniox_tts.mode": [
    ["preset", "preset - 使用下方预置音色"],
    ["clone", "clone - 克隆原视频说话人"]
  ],
  "soniox_tts.voice": [
    ["Maya", "Maya - 沉稳清晰、自然亲和（女）※仅 v1"],
    ["Claire", "Claire - 干练清晰、精致亲切（女）※仅 v1"],
    ["Noah", "Noah - 年轻明快、友好现代（男）※仅 v1"],
    ["Jack", "Jack - 亲和自信、真诚上扬（男）※仅 v1"],
    ["Nina", "Nina - 明亮活泼、富有个性（女）"],
    ["Emma", "Emma - 顺滑自然、轻松从容（女）"],
    ["Grace", "Grace - 轻柔舒缓、温暖抚慰（女）"],
    ["Mina", "Mina - 柔和沉静、真诚耐听（女）"],
    ["Daniel", "Daniel - 浑厚沉稳、成熟可靠（男）"],
    ["Adrian", "Adrian - 低沉专注、权威专业（男）"],
    ["Owen", "Owen - 沉着平实、内敛自信（男）"],
    ["Kenji", "Kenji - 冷静精准、稳重可信（男）"]
  ],
  "qwen3_tts.model": [
    ["qwen3-tts-flash", "qwen3-tts-flash"],
    ["qwen3-tts-instruct-flash", "qwen3-tts-instruct-flash"],
    ["qwen3-tts-flash-2025-11-27", "qwen3-tts-flash-2025-11-27"],
    ["qwen3-tts-flash-2025-09-18", "qwen3-tts-flash-2025-09-18"],
    ["qwen3-tts-instruct-flash-2026-01-26", "qwen3-tts-instruct-flash-2026-01-26"],
    ["qwen-tts-latest", "qwen-tts-latest"],
    ["qwen-tts", "qwen-tts"]
  ],
  "qwen3_tts.voice": [
    ["Cherry", "Cherry - 芊悦｜阳光积极、亲切自然小姐姐（女）"],
    ["Serena", "Serena - 苏瑶｜温柔小姐姐（女）"],
    ["Ethan", "Ethan - 晨煦｜阳光、温暖、活力（男）"],
    ["Chelsie", "Chelsie - 千雪｜二次元虚拟女友（女）"],
    ["Momo", "Momo - 茉兔｜撒娇搞怪，逗你开心（女）"],
    ["Vivian", "Vivian - 十三｜拽拽的、可爱的小暴躁（女）"],
    ["Moon", "Moon - 月白｜率性帅气（男）"],
    ["Maia", "Maia - 四月｜知性与温柔（女）"],
    ["Kai", "Kai - 凯｜耳朵的一场 SPA（男）"],
    ["Nofish", "Nofish - 不吃鱼｜不会翘舌音的设计师（男）"],
    ["Bella", "Bella - 萌宝｜小萝莉（女）"],
    ["Jennifer", "Jennifer - 詹妮弗｜电影质感美语女声（女）"],
    ["Ryan", "Ryan - 甜茶｜节奏拉满、戏感炸裂（男）"],
    ["Katerina", "Katerina - 卡捷琳娜｜御姐音色（女）"],
    ["Aiden", "Aiden - 艾登｜美语大男孩（男）"],
    ["Eldric Sage", "Eldric Sage - 沧明子｜沉稳睿智老者（男）"],
    ["Mia", "Mia - 乖小妹｜温顺乖巧（女）"],
    ["Mochi", "Mochi - 沙小弥｜聪明伶俐小大人（男）"],
    ["Bellona", "Bellona - 燕铮莺｜洪亮清晰、江湖感（女）"],
    ["Vincent", "Vincent - 田叔｜沙哑烟嗓（男）"],
    ["Bunny", "Bunny - 萌小姬｜萌属性小萝莉（女）"],
    ["Neil", "Neil"],
    ["Elias", "Elias"],
    ["Arthur", "Arthur"],
    ["Nini", "Nini - 邻家妹妹｜软糯甜妹（女）"],
    ["Seren", "Seren - 小婉｜温和舒缓（女）"],
    ["Pip", "Pip - 顽屁小孩｜调皮童真（男）"],
    ["Stella", "Stella - 少女阿月｜甜美少女音（女）"],
    ["Bodega", "Bodega - 博德加｜热情西班牙大叔（男）"],
    ["Sonrisa", "Sonrisa - 索尼莎｜热情拉美大姐（女）"],
    ["Alek", "Alek - 阿列克｜战斗民族冷暖感（男）"],
    ["Dolce", "Dolce - 多尔切｜慵懒意大利大叔（男）"],
    ["Sohee", "Sohee - 素熙｜韩国欧尼（女）"],
    ["Ono Anna", "Ono Anna - 小野杏｜鬼灵精怪青梅竹马（女）"],
    ["Lenn", "Lenn - 莱恩｜德国青年（男）"],
    ["Emilien", "Emilien - 埃米尔安｜法国大哥哥（男）"],
    ["Andre", "Andre - 安德雷｜磁性沉稳男声（男）"],
    ["Radio Gol", "Radio Gol - 拉迪奥·戈尔｜足球诗人（男）"],
    ["Jada", "Jada - 上海-阿珍｜沪上阿姐（女）"],
    ["Dylan", "Dylan - 北京-晓东｜北京胡同少年（男）"],
    ["Li", "Li - 南京-老李｜耐心瑜伽老师（男）"],
    ["Marcus", "Marcus - 陕西-秦川｜老陕味道（男）"],
    ["Roy", "Roy - 闽南-阿杰｜台湾哥仔（男）"],
    ["Peter", "Peter - 天津-李彼得｜天津相声捧哏（男）"],
    ["Sunny", "Sunny - 四川-晴儿｜川妹子（女）"],
    ["Eric", "Eric - 四川-程川｜成都男子（男）"],
    ["Rocky", "Rocky - 粤语-阿强｜幽默风趣（男）"],
    ["Kiki", "Kiki - 粤语-阿清｜甜美港妹（女）"]
  ],
  "qwen3_tts.language_type": [
    ["Chinese", "Chinese"],
    ["English", "English"],
    ["Japanese", "Japanese"],
    ["Korean", "Korean"],
    ["German", "German"],
    ["French", "French"],
    ["Spanish", "Spanish"],
    ["Italian", "Italian"],
    ["Portuguese", "Portuguese"],
    ["Russian", "Russian"]
  ],
  "qwen3_tts.region": [
    ["beijing", "beijing"],
    ["singapore", "singapore"]
  ],
  "sf_fish_tts.mode": [
    ["preset", "Preset"],
    ["custom", "Refer_stable"],
    ["dynamic", "Refer_dynamic"]
  ],
  "gpt_sovits.refer_mode": [
    ["1", "Mode 1: Use provided reference audio only"],
    ["2", "Mode 2: Use first audio from video as reference"],
    ["3", "Mode 3: Use each audio from video as reference"]
  ],
  "mimo_tts.model": [
    ["mimo-v2.5-tts", "mimo-v2.5-tts"],
    ["mimo-v2.5-tts-voicedesign", "mimo-v2.5-tts-voicedesign"],
    ["mimo-v2.5-tts-voiceclone", "mimo-v2.5-tts-voiceclone"]
  ],
  "mimo_tts.voice": [
    ["mimo_default", "mimo_default"],
    ["冰糖", "冰糖"],
    ["茉莉", "茉莉"],
    ["苏打", "苏打"],
    ["白桦", "白桦"],
    ["Mia", "Mia"],
    ["Chloe", "Chloe"],
    ["Milo", "Milo"],
    ["Dean", "Dean"]
  ]
};

const KNOWN_MODELS = [
  "deepseek-v4-flash",
  "deepseek-v3.2",
  "qwen-plus",
  "qwen-max",
  "qwen-turbo",
  "gpt-4o-mini",
  "gpt-4o"
];

const SPEAKER_MODES = [
  ["fixed", "固定音色（填写当前 TTS 支持的 voice）"],
  ["qwen3_tts", "Qwen3 TTS（选择千问音色）"],
  ["mimo_voicedesign", "MiMo 声音设计（描述声音）"],
  ["clone", "克隆该说话人（用预览音频做参考）"],
  ["default", "使用全局默认音色"]
];

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

function setConfigStatus(text) {
  configStatusEl.textContent = text;
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
  if (job.status === "waiting_speaker") {
    return "已暂停：请在下方为每个说话人选择配音方式。";
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

  startButton.disabled = !isVideo || (job && ["queued", "running", "waiting_speaker"].includes(job.status));
  overlayButton.disabled = !(job && job.status === "done" && isVideo);
  overlayButton.textContent = playbackMode === "original" ? "使用配音" : "使用原声";
  setStatus(isVideo ? phaseLabel(job, playbackMode) : "请先打开一个 YouTube 视频。");
}

function configControls() {
  return Array.from(document.querySelectorAll("[data-config-key]"));
}

function persistedDetails() {
  return Array.from(document.querySelectorAll("details[data-details-key]"));
}

function getStoredDetailsState() {
  return new Promise((resolve) => {
    chrome.storage.local.get(DETAILS_STATE_STORAGE_KEY, (items) => {
      const value = items[DETAILS_STATE_STORAGE_KEY];
      resolve(value && typeof value === "object" ? value : {});
    });
  });
}

function setStoredDetailsState(state) {
  chrome.storage.local.set({ [DETAILS_STATE_STORAGE_KEY]: state });
}

function currentDetailsState() {
  const state = {};
  for (const details of persistedDetails()) {
    state[details.dataset.detailsKey] = details.open;
  }
  return state;
}

async function restoreDetailsState() {
  restoringDetailsState = true;
  try {
    const state = await getStoredDetailsState();
    for (const details of persistedDetails()) {
      const key = details.dataset.detailsKey;
      if (Object.prototype.hasOwnProperty.call(state, key)) {
        details.open = Boolean(state[key]);
      }
    }
  } finally {
    restoringDetailsState = false;
  }
}

function bindDetailsStatePersistence() {
  for (const details of persistedDetails()) {
    details.addEventListener("toggle", () => {
      if (restoringDetailsState) {
        return;
      }
      setStoredDetailsState(currentDetailsState());
    });
  }
}

function appendOption(select, value, label) {
  const option = document.createElement("option");
  option.value = String(value);
  option.textContent = String(label);
  select.appendChild(option);
}

function setSelectOptions(select, options) {
  select.textContent = "";
  for (const [value, label] of options) {
    appendOption(select, value, label);
  }
}

function ensureSelectValue(select, value) {
  const textValue = value == null ? "" : String(value);
  if (textValue && !Array.from(select.options).some((option) => option.value === textValue)) {
    appendOption(select, textValue, textValue);
  }
  select.value = textValue;
}

function initStaticOptions() {
  for (const control of configControls()) {
    if (control.tagName !== "SELECT") {
      continue;
    }
    const options = OPTIONS[control.dataset.configKey];
    if (options) {
      setSelectOptions(control, options);
    }
  }

  const knownModels = document.getElementById("knownModels");
  knownModels.textContent = "";
  for (const model of KNOWN_MODELS) {
    appendOption(knownModels, model, model);
  }
}

function applyDynamicOptions(dynamicOptions = {}) {
  for (const [key, values] of Object.entries(dynamicOptions)) {
    const select = configControls().find((control) => control.dataset.configKey === key && control.tagName === "SELECT");
    if (!select || !Array.isArray(values)) {
      continue;
    }
    const options = values.length ? values.map((value) => [value, value]) : [[currentConfigValues[key] || "", currentConfigValues[key] || ""]];
    setSelectOptions(select, options);
  }
}

function valueForKey(key) {
  const control = configControls().find((item) => item.dataset.configKey === key);
  if (!control) {
    return undefined;
  }
  if (control.type === "checkbox") {
    return control.checked;
  }
  return control.value;
}

function showConditionMatches(conditionText) {
  if (!conditionText) {
    return true;
  }
  return conditionText.split(";").every((part) => {
    const [key, expected] = part.split("=");
    const actual = valueForKey(key);
    return String(actual) === expected;
  });
}

function updateConditionalVisibility() {
  for (const element of document.querySelectorAll(".conditional")) {
    element.classList.toggle("hidden", !showConditionMatches(element.dataset.showWhen));
  }

  const ttsMethod = valueForKey("tts_method");
  for (const panel of document.querySelectorAll(".tts-settings")) {
    panel.classList.toggle("active", panel.dataset.ttsPanel === ttsMethod);
  }
}

function populateConfig(values) {
  applyingConfig = true;
  currentConfigValues = values || {};
  for (const control of configControls()) {
    const key = control.dataset.configKey;
    const value = currentConfigValues[key];
    if (control.type === "checkbox") {
      control.checked = Boolean(value);
    } else if (control.tagName === "SELECT") {
      ensureSelectValue(control, value);
    } else if (Array.isArray(value)) {
      const nextValue = value.join("\n");
      if (control.value !== nextValue) {
        control.value = nextValue;
      }
    } else {
      const nextValue = value == null ? "" : String(value);
      if (control.value !== nextValue) {
        control.value = nextValue;
      }
    }
  }
  updateConditionalVisibility();
  applyingConfig = false;
}

function readConfigForm() {
  const values = {};
  for (const control of configControls()) {
    const key = control.dataset.configKey;
    if (control.type === "checkbox") {
      values[key] = control.checked;
    } else if (INTEGER_KEYS.has(key)) {
      values[key] = Number.parseInt(control.value || "1", 10);
    } else if (FLOAT_KEYS.has(key)) {
      values[key] = Number.parseFloat(control.value || "0");
    } else if (LIST_TEXT_KEYS.has(key)) {
      const lines = control.value.split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean);
      values[key] = lines.length > 1 ? lines : (lines[0] || "");
    } else {
      values[key] = control.value;
    }
  }
  return values;
}

async function loadConfig() {
  setConfigStatus("正在读取配置...");
  try {
    const response = await fetch(`${BRIDGE_ORIGIN}/config`, { cache: "no-store" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.ok) {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
    applyDynamicOptions(body.dynamic_options || {});
    populateConfig(body.values || {});
    configLoaded = true;
    setConfigStatus("配置已加载。");
  } catch (error) {
    configLoaded = false;
    setConfigStatus(`读取配置失败：${error.message || String(error)}`);
  }
}

async function saveConfig() {
  if (!configLoaded) {
    return;
  }
  if (saveInFlight) {
    saveAgain = true;
    return;
  }
  saveInFlight = true;
  setConfigStatus("正在保存配置...");
  try {
    const values = readConfigForm();
    const response = await fetch(`${BRIDGE_ORIGIN}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.ok) {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
    if (!saveAgain) {
      applyDynamicOptions(body.dynamic_options || {});
      populateConfig(body.values || {});
    } else {
      currentConfigValues = body.values || currentConfigValues;
    }
    setConfigStatus("配置已自动保存。新任务会使用这些设置。");
  } catch (error) {
    setConfigStatus(`保存失败：${error.message || String(error)}`);
  } finally {
    saveInFlight = false;
    if (saveAgain) {
      saveAgain = false;
      scheduleConfigSave(120);
    }
  }
}

function scheduleConfigSave(delay = 500) {
  if (!configLoaded || applyingConfig) {
    return;
  }
  window.clearTimeout(saveTimer);
  setConfigStatus("配置有改动，准备自动保存...");
  saveTimer = window.setTimeout(saveConfig, delay);
}

function setSpeakerPickerStatus(text) {
  speakerPickerStatusEl.textContent = text;
}

function hideSpeakerPicker() {
  activeSpeakerJobId = null;
  speakerManifest = [];
  speakerPickerEl.classList.add("hidden");
  speakerRowsEl.textContent = "";
  speakerPickerCountEl.textContent = "";
  setSpeakerPickerStatus("");
}

function createSpeakerField(labelText, control) {
  const field = document.createElement("div");
  field.className = "field";
  const label = document.createElement("label");
  label.textContent = labelText;
  field.append(label, control);
  return field;
}

function updateSpeakerRowVisibility(row) {
  const mode = row.querySelector(".speaker-mode")?.value || "default";
  for (const extra of row.querySelectorAll(".speaker-extra")) {
    extra.classList.toggle("hidden", extra.dataset.mode !== mode);
  }
}

function renderSpeakerPicker(jobId, manifest) {
  activeSpeakerJobId = jobId;
  speakerManifest = manifest;
  speakerPickerEl.classList.remove("hidden");
  speakerPickerCountEl.textContent = manifest.length ? `${manifest.length} 人` : "";
  speakerRowsEl.textContent = "";

  if (!manifest.length) {
    setSpeakerPickerStatus("等待说话人预览生成，稍后会自动刷新。");
    return;
  }

  for (const entry of manifest) {
    const speakerId = String(entry.speaker_id || "").trim();
    if (!speakerId) {
      continue;
    }

    const row = document.createElement("div");
    row.className = "speaker-row";
    row.dataset.speakerId = speakerId;
    row.dataset.refWav = entry.wav || "";

    const title = document.createElement("div");
    title.className = "speaker-title";
    const name = document.createElement("span");
    name.textContent = `Speaker ${speakerId}`;
    const meta = document.createElement("span");
    meta.className = "speaker-meta";
    const duration = typeof entry.duration === "number" ? `${entry.duration.toFixed(1)}s` : "";
    const words = Number.isInteger(entry.num_words) ? `${entry.num_words} words` : "";
    meta.textContent = [duration, words].filter(Boolean).join(" · ");
    title.append(name, meta);
    row.append(title);

    if (entry.audio_url) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = new URL(entry.audio_url, BRIDGE_ORIGIN).href;
      row.append(audio);
    }

    if (entry.text) {
      const sample = document.createElement("div");
      sample.className = "speaker-sample";
      sample.textContent = entry.text;
      row.append(sample);
    }

    const modeSelect = document.createElement("select");
    modeSelect.className = "speaker-mode";
    for (const [value, label] of SPEAKER_MODES) {
      appendOption(modeSelect, value, label);
    }
    row.append(createSpeakerField("配音策略", modeSelect));

    const fixedInput = document.createElement("input");
    fixedInput.className = "speaker-voice";
    fixedInput.type = "text";
    fixedInput.placeholder = "例如 Andre / zh-CN-XiaoxiaoNeural / alloy";
    const fixedExtra = createSpeakerField("TTS voice 名称", fixedInput);
    fixedExtra.classList.add("speaker-extra");
    fixedExtra.dataset.mode = "fixed";
    row.append(fixedExtra);

    const qwen3Select = document.createElement("select");
    qwen3Select.className = "speaker-qwen3-voice";
    setSelectOptions(qwen3Select, OPTIONS["qwen3_tts.voice"]);
    const globalQwen3Voice = valueForKey("qwen3_tts.voice") || "Andre";
    ensureSelectValue(qwen3Select, globalQwen3Voice);
    const qwen3Extra = createSpeakerField("Qwen3 TTS Voice", qwen3Select);
    qwen3Extra.classList.add("speaker-extra");
    qwen3Extra.dataset.mode = "qwen3_tts";
    row.append(qwen3Extra);

    const mimoText = document.createElement("textarea");
    mimoText.className = "speaker-mimo-description";
    mimoText.rows = 4;
    mimoText.placeholder = "例如：沉稳、低沉、自然的中文男声，语速适中。";
    const mimoExtra = createSpeakerField("MiMo 声音描述", mimoText);
    mimoExtra.classList.add("speaker-extra");
    mimoExtra.dataset.mode = "mimo_voicedesign";
    row.append(mimoExtra);

    const cloneNote = document.createElement("div");
    cloneNote.className = "speaker-extra speaker-sample";
    cloneNote.dataset.mode = "clone";
    cloneNote.textContent = "将使用上面的预览音频作为该说话人的克隆参考。";
    row.append(cloneNote);

    modeSelect.addEventListener("change", () => updateSpeakerRowVisibility(row));
    updateSpeakerRowVisibility(row);
    speakerRowsEl.append(row);
  }

  setSpeakerPickerStatus("任务已暂停，提交后会自动继续配音。");
}

async function loadSpeakerPreview(jobId) {
  if (!jobId) {
    hideSpeakerPicker();
    return;
  }
  if (activeSpeakerJobId === jobId && speakerManifest.length) {
    return;
  }
  setSpeakerPickerStatus("正在读取说话人预览...");
  try {
    const response = await fetch(`${BRIDGE_ORIGIN}/jobs/${jobId}/speaker-preview`, { cache: "no-store" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.ok) {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
    renderSpeakerPicker(jobId, body.manifest || []);
  } catch (error) {
    speakerPickerEl.classList.remove("hidden");
    setSpeakerPickerStatus(`读取说话人预览失败：${error.message || String(error)}`);
  }
}

function collectSpeakerPicks() {
  const picks = {};
  const missing = [];
  for (const row of speakerRowsEl.querySelectorAll(".speaker-row")) {
    const speakerId = row.dataset.speakerId;
    const mode = row.querySelector(".speaker-mode")?.value || "default";
    const pick = { mode };
    if (mode === "fixed") {
      const voice = (row.querySelector(".speaker-voice")?.value || "").trim();
      if (!voice) {
        missing.push(`Speaker ${speakerId} 的 voice 名称`);
      }
      pick.voice = voice;
    } else if (mode === "qwen3_tts") {
      const voice = (row.querySelector(".speaker-qwen3-voice")?.value || "").trim();
      if (!voice) {
        missing.push(`Speaker ${speakerId} 的 Qwen3 TTS Voice`);
      }
      pick.voice = voice;
    } else if (mode === "mimo_voicedesign") {
      const description = (row.querySelector(".speaker-mimo-description")?.value || "").trim();
      if (!description) {
        missing.push(`Speaker ${speakerId} 的 MiMo 声音描述`);
      }
      pick.voice_description = description;
    } else if (mode === "clone") {
      pick.ref_wav = row.dataset.refWav || "";
    }
    picks[speakerId] = pick;
  }
  return { picks, missing };
}

async function submitSpeakerPicks() {
  if (!activeSpeakerJobId) {
    return;
  }
  const { picks, missing } = collectSpeakerPicks();
  if (missing.length) {
    setSpeakerPickerStatus(`请补全：${missing.join("，")}`);
    return;
  }

  speakerContinueButton.disabled = true;
  setSpeakerPickerStatus("正在提交选择并恢复任务...");
  try {
    const response = await fetch(`${BRIDGE_ORIGIN}/jobs/${activeSpeakerJobId}/speaker-preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ picks })
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.ok) {
      throw new Error(body.error || `HTTP ${response.status}`);
    }
    setSpeakerPickerStatus("已提交，任务正在继续。");
    await refresh();
  } catch (error) {
    setSpeakerPickerStatus(`提交失败：${error.message || String(error)}`);
  } finally {
    speakerContinueButton.disabled = false;
  }
}

async function checkBridge() {
  try {
    const response = await fetch(`${BRIDGE_ORIGIN}/health`, { cache: "no-store" });
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
  if (job && job.status === "waiting_speaker") {
    await loadSpeakerPreview(job.id);
  } else {
    hideSpeakerPicker();
  }
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

speakerContinueButton.addEventListener("click", submitSpeakerPicks);

for (const control of configControls()) {
  control.addEventListener("change", () => {
    updateConditionalVisibility();
    scheduleConfigSave(80);
  });
  control.addEventListener("input", () => {
    updateConditionalVisibility();
    const quickSave = control.type === "checkbox" || control.tagName === "SELECT";
    scheduleConfigSave(quickSave ? 80 : 350);
  });
  control.addEventListener("blur", () => {
    updateConditionalVisibility();
    scheduleConfigSave(0);
  });
}

initStaticOptions();
bindDetailsStatePersistence();
restoreDetailsState();
loadConfig();
refresh();
refreshTimer = setInterval(refresh, 5000);
window.addEventListener("unload", () => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
});
