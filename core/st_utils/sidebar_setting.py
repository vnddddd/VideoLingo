import streamlit as st
import requests
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *


def _ensure_demucs_keys():
    """Idempotently insert demucs_backend + hf_demucs defaults into config.yaml.

    Needed because users pulling new code with an older config.yaml would crash
    on the load_key('demucs_backend') / load_key('hf_demucs.hf_token') calls below.
    Best-effort: any IO/parse error is swallowed (UI must not be blocked).
    """
    try:
        from core.utils.config_utils import CONFIG_PATH, lock, yaml as _yaml
        with lock:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = _yaml.load(f)
            if data is None:
                return
            changed = False
            if 'demucs_backend' not in data:
                data['demucs_backend'] = 'local'
                changed = True
            if 'hf_demucs' not in data or not isinstance(data.get('hf_demucs'), dict):
                data['hf_demucs'] = {}
                changed = True
            hf_defaults = {
                'space_id': 'abidlabs/music-separation',
                'hf_token': '',
                'api_name': '/predict',
            }
            for k, v in hf_defaults.items():
                if k not in data['hf_demucs']:
                    data['hf_demucs'][k] = v
                    changed = True
            if changed:
                with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                    _yaml.dump(data, f)
    except Exception:
        pass


def _ensure_multi_speaker_key():
    """Idempotently insert `multi_speaker_enabled` default into config.yaml.

    Newer code reads this top-level flag in the sidebar (and in _2_asr) to decide
    whether the ASR backend should keep the audio intact for diarization. Older
    config.yaml files won't have the key, so we seed it as False on first render.
    """
    try:
        from core.utils.config_utils import CONFIG_PATH, lock, yaml as _yaml
        with lock:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = _yaml.load(f)
            if data is None:
                return
            if 'multi_speaker_enabled' not in data:
                data['multi_speaker_enabled'] = False
                with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                    _yaml.dump(data, f)
    except Exception:
        pass


def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


# Chinese blurbs for the voices most likely to be picked. Doubles as the
# offline fallback list when the models endpoint cannot be reached; any voice
# not listed here shows the English description the API returns.
SONIOX_VOICE_ZH = {
    "Maya": "沉稳清晰、自然亲和（女）",
    "Nina": "明亮活泼、富有个性（女）",
    "Emma": "顺滑自然、轻松从容（女）",
    "Claire": "干练清晰、精致亲切（女）",
    "Grace": "轻柔舒缓、温暖抚慰（女）",
    "Mina": "柔和沉静、真诚耐听（女）",
    "Daniel": "浑厚沉稳、成熟可靠（男）",
    "Noah": "年轻明快、友好现代（男）",
    "Jack": "亲和自信、真诚上扬（男）",
    "Adrian": "低沉专注、权威专业（男）",
    "Owen": "沉着平实、内敛自信（男）",
    "Kenji": "冷静精准、稳重可信（男）",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _soniox_model_voices(model):
    """Built-in voices for a Soniox model, read from the API.

    Cached because Streamlit re-runs the sidebar on every widget interaction.
    Returns None when the API cannot be reached (no key configured yet, or
    offline) so the caller can fall back to a static list.
    """
    try:
        from core.tts_backend.soniox_tts import list_model_voices
        return [
            (v["id"], v.get("description") or "", v.get("gender") or "")
            for v in list_model_voices(model)
        ]
    except Exception:
        return None


def _positive_int_config(key, fallback_key=None, default=1):
    """Read a positive integer config value with optional backward-compatible fallback."""
    try:
        value = load_key(key)
    except Exception:
        if fallback_key is None:
            value = default
        else:
            try:
                value = load_key(fallback_key)
            except Exception:
                value = default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _fetch_model_list(base_url, api_key):
    """Fetch available models from OpenAI-compatible /v1/models endpoint."""
    if not api_key or not base_url:
        return []
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted([m["id"] for m in data if "id" in m])
    except Exception:
        return []


def _search_models(search_term, **kwargs):
    """Search function for st_searchbox — returns models matching the search term."""
    models = st.session_state.get("_model_list", [])
    if not search_term:
        return models if models else []
    term = search_term.lower()
    matched = [m for m in models if term in m.lower()]
    # Always include the raw input as an option so users can type custom model names
    if search_term not in matched:
        matched.insert(0, search_term)
    return matched


def page_setting():
    # Make sure newly added demucs config keys exist so older config.yaml files
    # do not crash this UI on KeyError when load_key('demucs_backend') runs.
    _ensure_demucs_keys()
    _ensure_multi_speaker_key()

    # Widen the sidebar slightly to accommodate the model searchbox
    st.markdown(
        """<style>[data-testid="stSidebar"] {min-width: 420px; max-width: 420px;}</style>""",
        unsafe_allow_html=True,
    )

    display_language = st.selectbox(
        "Display Language 🌐",
        options=list(DISPLAY_LANGUAGES.keys()),
        index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")),
    )
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key", placeholder=t("Enter your API key"))
        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("Openai format, will add /v1/chat/completions automatically"),
        )

        # Try to use searchbox for model selection, fall back to text_input
        try:
            from streamlit_searchbox import st_searchbox
            from streamlit_searchbox import _list_to_options_js, _list_to_options_py

            if st.button(
                t("Fetch Model List"), key="fetch_models", use_container_width=True
            ):
                with st.spinner(t("Fetching models...")):
                    models = _fetch_model_list(
                        load_key("api.base_url"), load_key("api.key")
                    )
                    st.session_state["_model_list"] = models
                    if models:
                        # Update searchbox internal state directly so dropdown shows options
                        sb_key = "model_searchbox"
                        if sb_key in st.session_state:
                            st.session_state[sb_key]["options_js"] = (
                                _list_to_options_js(models)
                            )
                            st.session_state[sb_key]["options_py"] = (
                                _list_to_options_py(models)
                            )
                        st.toast(
                            t("Fetched {n} models").replace("{n}", str(len(models))),
                            icon="✅",
                        )
                    else:
                        st.toast(
                            t(
                                "Failed to fetch models, please check API Key and Base URL"
                            ),
                            icon="❌",
                        )

            current_model = load_key("api.model")
            model_list = st.session_state.get("_model_list", None)

            sb_key = "model_searchbox"
            selected = st_searchbox(
                _search_models,
                placeholder=t("Search or enter model name"),
                default=current_model if current_model else None,
                default_searchterm=current_model if current_model else "",
                default_use_searchterm=True,
                default_options=model_list if model_list else None,
                key=sb_key,
                clear_on_submit=False,
            )
            if selected and selected != load_key("api.model"):
                update_key("api.model", selected)

            if st.button("📡 " + t("Check API"), key="api", use_container_width=True):
                with st.spinner(t("Check API") + "..."):
                    is_valid = check_api()
                st.toast(
                    t("API Key is valid") if is_valid else t("API Key is invalid"),
                    icon="✅" if is_valid else "❌",
                )
        except ImportError:
            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(
                    t("MODEL"),
                    "api.model",
                    help=t("click to check API validity") + " 👉",
                    placeholder=t("Search or enter model name"),
                )
            with c2:
                if st.button("📡", key="api"):
                    is_valid = check_api()
                    st.toast(
                        t("API Key is valid") if is_valid else t("API Key is invalid"),
                        icon="✅" if is_valid else "❌",
                    )
        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()

        llm_max_workers = st.number_input(
            t("LLM Concurrency"),
            min_value=1,
            step=1,
            value=_positive_int_config("api.max_workers", fallback_key="max_workers", default=1),
            help=t("Maximum concurrent LLM requests for translation and subtitle splitting."),
        )
        if int(llm_max_workers) != _positive_int_config("api.max_workers", fallback_key="max_workers", default=1):
            update_key("api.max_workers", int(llm_max_workers))
            st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja",
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language")),
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtime = st.selectbox(
            t("WhisperX Runtime"),
            options=["local", "cloud", "elevenlabs", "soniox"],
            index=["local", "cloud", "elevenlabs", "soniox"].index(load_key("whisper.runtime")),
            help=t(
                "Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key, soniox runtime requires Soniox API key (stt-async-v4 model)"
            ),
        )
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")
        if runtime == "soniox":
            config_input(t("Soniox API"), "whisper.soniox_api_key")
            soniox_diarize = st.toggle(
                t("Soniox Speaker Diarization"),
                value=load_key("whisper.soniox_diarize"),
                help=t(
                    "Enable speaker diarization (adds speaker labels to transcript). Increases API cost."
                ),
            )
            if soniox_diarize != load_key("whisper.soniox_diarize"):
                update_key("whisper.soniox_diarize", soniox_diarize)
                st.rerun()

        # --- Multi-speaker (diarization) toggle ---
        # Routes each detected speaker to its own TTS voice. Requires an ASR
        # backend whose response carries speaker_id labels (Soniox / ElevenLabs).
        # For unsupported backends we hard-disable the toggle and show a red
        # helper, so users do not silently get an all-None speaker column.
        ms_supported = runtime in ("soniox", "elevenlabs")
        try:
            ms_current = bool(load_key("multi_speaker_enabled"))
        except Exception:
            ms_current = False
        multi_speaker_enabled = st.toggle(
            t("Multi-speaker diarization"),
            value=ms_current and ms_supported,
            disabled=not ms_supported,
            help=t(
                "Detect different speakers and assign a distinct TTS voice to each. "
                "Only works with Soniox or ElevenLabs (whose ASR returns speaker labels)."
            ),
        )
        if not ms_supported and ms_current:
            # Backend was switched to one that cannot diarize; auto-revert so
            # downstream stages do not crash on missing speaker_id columns.
            update_key("multi_speaker_enabled", False)
            st.rerun()
        elif ms_supported and multi_speaker_enabled != ms_current:
            update_key("multi_speaker_enabled", multi_speaker_enabled)
            st.rerun()
        if not ms_supported:
            st.markdown(
                "<span style='color:#d32f2f;font-size:0.85em'>⚠️ "
                + t("Current ASR backend does not support diarization; switch to Soniox or ElevenLabs.")
                + "</span>",
                unsafe_allow_html=True,
            )

        asr_max_workers = st.number_input(
            t("ASR Clip Concurrency"),
            min_value=1,
            max_value=16,
            value=int(load_key("whisper.max_workers")),
            step=1,
            help=t(
                "Number of ASR audio clips to transcribe concurrently. Local WhisperX is forced to 1 to protect GPU/VRAM."
            ),
        )
        if int(asr_max_workers) != int(load_key("whisper.max_workers")):
            update_key("whisper.max_workers", int(asr_max_workers))
            st.rerun()

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t(
                    "Input any language in natural language, as long as llm can understand"
                ),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            help=t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        if demucs:
            backend_options = ["local", "hf_space"]
            try:
                cur_backend = load_key("demucs_backend")
            except KeyError:
                cur_backend = "local"
            if cur_backend not in backend_options:
                cur_backend = "local"
            demucs_backend = st.radio(
                t("Demucs Backend"),
                options=backend_options,
                index=backend_options.index(cur_backend),
                horizontal=True,
                help=t(
                    "local: run htdemucs on this machine (needs CUDA GPU). "
                    "hf_space: offload to a HuggingFace Space (free T4, ~1 min per 15 min video, needs HF token)."
                ),
                key="demucs_backend_radio",
            )
            if demucs_backend != cur_backend:
                update_key("demucs_backend", demucs_backend)
                st.rerun()

            if demucs_backend == "hf_space":
                try:
                    cur_token = load_key("hf_demucs.hf_token") or ""
                except KeyError:
                    cur_token = ""
                hf_token = st.text_input(
                    t("HF Token"),
                    value=cur_token,
                    type="password",
                    help=t(
                        "Read-scope token from https://huggingface.co/settings/tokens. "
                        "Saved only to your local config.yaml."
                    ),
                    placeholder="hf_xxxxxxxxxxxxxxx",
                    key="hf_demucs_token_input",
                )
                if hf_token != cur_token:
                    update_key("hf_demucs.hf_token", hf_token)
                    st.rerun()

        burn_subtitles = st.toggle(
            t("Burn-in Subtitles"),
            value=load_key("burn_subtitles"),
            help=t(
                "Whether to burn subtitles into the video, will increase processing time"
            ),
        )
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        tts_methods = [
            "azure_tts",
            "openai_tts",
            "qwen3_tts",
            "soniox_tts",
            "fish_tts",
            "sf_fish_tts",
            "edge_tts",
            "gpt_sovits",
            "custom_tts",
            "sf_cosyvoice2",
            "f5tts",
            "mimo_tts",
            "indextts2",
        ]
        select_tts = st.selectbox(
            t("TTS Method"),
            options=tts_methods,
            index=tts_methods.index(load_key("tts_method")),
        )
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        tts_max_workers = st.number_input(
            t("TTS Concurrency"),
            min_value=1,
            step=1,
            value=_positive_int_config("tts_max_workers", fallback_key="max_workers", default=1),
            help=t("Maximum concurrent TTS generation requests. GPT-SoVITS is forced to 1 to avoid reference-audio/state conflicts."),
        )
        if int(tts_max_workers) != _positive_int_config("tts_max_workers", fallback_key="max_workers", default=1):
            update_key("tts_max_workers", int(tts_max_workers))
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "soniox_tts":
            config_input(
                t("Soniox API Key"),
                "soniox_tts.api_key",
                help=t("Leave empty to reuse the Soniox key configured for ASR."),
            )

            soniox_models = ["tts-rt-v1", "tts-rt-v2"]
            try:
                current_soniox_model = load_key("soniox_tts.model")
            except Exception:
                current_soniox_model = "tts-rt-v1"
            soniox_model = st.selectbox(
                t("Soniox Model"),
                options=soniox_models,
                index=soniox_models.index(current_soniox_model)
                if current_soniox_model in soniox_models
                else 0,
                help=t("v1 is the officially released model. v2 offers many more voices but is not yet announced in Soniox's docs."),
            )
            if soniox_model != current_soniox_model:
                update_key("soniox_tts.model", soniox_model)
                st.rerun()

            soniox_mode_options = {
                "preset": t("Preset"),
                "clone": t("Clone original speaker"),
            }
            try:
                current_soniox_mode = load_key("soniox_tts.mode")
            except Exception:
                current_soniox_mode = "preset"
            soniox_mode = st.selectbox(
                t("Mode Selection"),
                options=list(soniox_mode_options.keys()),
                format_func=lambda x: soniox_mode_options[x],
                index=list(soniox_mode_options.keys()).index(current_soniox_mode)
                if current_soniox_mode in soniox_mode_options
                else 0,
                help=t("Clone reuses the video's own reference audio; a cloned voice is created once and reused (max 20 per Soniox organisation)."),
            )
            if soniox_mode != current_soniox_mode:
                update_key("soniox_tts.mode", soniox_mode)
                st.rerun()

            if soniox_mode == "preset":
                # Pulled from the API rather than hardcoded: v1 offers 28 voices
                # and v2 offers 71, and six v1 names (Maya, Noah, Jack, Claire,
                # Sofia, Meera) are absent from v2 — a static list would leave a
                # silently invalid voice selected after switching models.
                api_voices = _soniox_model_voices(soniox_model)
                if api_voices:
                    soniox_voice_options = {
                        vid: SONIOX_VOICE_ZH.get(vid)
                        or (f"{desc[:56]}（{gender}）" if gender else desc[:56])
                        for vid, desc, gender in api_voices
                    }
                else:
                    soniox_voice_options = dict(SONIOX_VOICE_ZH)
                    st.caption(t("Could not reach Soniox; showing a built-in voice list."))

                current_voice = load_key("soniox_tts.voice")
                voice_names = list(soniox_voice_options.keys())
                if current_voice and current_voice not in voice_names:
                    # Either a cloned-voice UUID, or a built-in the selected
                    # model does not offer (e.g. Maya after switching to v2).
                    voice_names.insert(0, current_voice)
                    soniox_voice_options[current_voice] = t("cloned voice, or not offered by this model")
                    # Soniox resolves a UUID as a cloned voice, so only warn when
                    # the value looks like a built-in name the model lacks.
                    looks_like_uuid = len(current_voice) == 36 and current_voice.count("-") == 4
                    if api_voices and not looks_like_uuid:
                        st.warning(
                            t("Voice '{v}' is not available on {m}. Pick another one.").format(
                                v=current_voice, m=soniox_model
                            )
                        )
                soniox_voice = st.selectbox(
                    t("Soniox Voice"),
                    options=voice_names,
                    format_func=lambda x: f"{x} - {soniox_voice_options.get(x, x)}",
                    index=voice_names.index(current_voice) if current_voice in voice_names else 0,
                    help=t("Built-in voice, or paste a cloned-voice UUID into config.yaml."),
                )
                if soniox_voice != current_voice:
                    update_key("soniox_tts.voice", soniox_voice)
                    st.rerun()

            config_input(
                t("Language Code"),
                "soniox_tts.language",
                help=t("BCP-47 code such as zh / en / ja. Leave empty to auto-detect from the text."),
            )

            current_speed = float(load_key("soniox_tts.speed") or 1.0)
            soniox_speed = st.slider(
                t("Speaking Speed"),
                min_value=0.7,
                max_value=1.3,
                value=min(1.3, max(0.7, current_speed)),
                step=0.05,
                help=t("Baseline speaking rate. Lines that overrun their subtitle slot are re-rendered faster automatically."),
            )
            if abs(soniox_speed - current_speed) > 1e-6:
                update_key("soniox_tts.speed", round(float(soniox_speed), 2))
                st.rerun()

        elif select_tts == "qwen3_tts":
            config_input(t("DashScope API Key"), "qwen3_tts.api_key")

            qwen3_models = [
                "qwen3-tts-flash",
                "qwen3-tts-instruct-flash",
                "qwen3-tts-flash-2025-11-27",
                "qwen3-tts-flash-2025-09-18",
                "qwen3-tts-instruct-flash-2026-01-26",
                "qwen-tts-latest",
                "qwen-tts",
            ]
            current_model = load_key("qwen3_tts.model")
            qwen3_model = st.selectbox(
                t("Qwen3 TTS Model"),
                options=qwen3_models,
                index=qwen3_models.index(current_model) if current_model in qwen3_models else 0,
                help=t("Use qwen3-tts-flash by default; instruct model supports instructions in DashScope."),
            )
            if qwen3_model != current_model:
                update_key("qwen3_tts.model", qwen3_model)
                st.rerun()

            qwen3_voice_options = {
                "Cherry": "芊悦｜阳光积极、亲切自然小姐姐（女）",
                "Serena": "苏瑶｜温柔小姐姐（女）",
                "Ethan": "晨煦｜阳光、温暖、活力（男）",
                "Chelsie": "千雪｜二次元虚拟女友（女）",
                "Momo": "茉兔｜撒娇搞怪，逗你开心（女）",
                "Vivian": "十三｜拽拽的、可爱的小暴躁（女）",
                "Moon": "月白｜率性帅气（男）",
                "Maia": "四月｜知性与温柔（女）",
                "Kai": "凯｜耳朵的一场 SPA（男）",
                "Nofish": "不吃鱼｜不会翘舌音的设计师（男）",
                "Bella": "萌宝｜小萝莉（女）",
                "Jennifer": "詹妮弗｜电影质感美语女声（女）",
                "Ryan": "甜茶｜节奏拉满、戏感炸裂（男）",
                "Katerina": "卡捷琳娜｜御姐音色（女）",
                "Aiden": "艾登｜美语大男孩（男）",
                "Eldric Sage": "沧明子｜沉稳睿智老者（男）",
                "Mia": "乖小妹｜温顺乖巧（女）",
                "Mochi": "沙小弥｜聪明伶俐小大人（男）",
                "Bellona": "燕铮莺｜洪亮清晰、江湖感（女）",
                "Vincent": "田叔｜沙哑烟嗓（男）",
                "Bunny": "萌小姬｜萌属性小萝莉（女）",
                "Neil": "Neil",
                "Elias": "Elias",
                "Arthur": "Arthur",
                "Nini": "邻家妹妹｜软糯甜妹（女）",
                "Seren": "小婉｜温和舒缓（女）",
                "Pip": "顽屁小孩｜调皮童真（男）",
                "Stella": "少女阿月｜甜美少女音（女）",
                "Bodega": "博德加｜热情西班牙大叔（男）",
                "Sonrisa": "索尼莎｜热情拉美大姐（女）",
                "Alek": "阿列克｜战斗民族冷暖感（男）",
                "Dolce": "多尔切｜慵懒意大利大叔（男）",
                "Sohee": "素熙｜韩国欧尼（女）",
                "Ono Anna": "小野杏｜鬼灵精怪青梅竹马（女）",
                "Lenn": "莱恩｜德国青年（男）",
                "Emilien": "埃米尔安｜法国大哥哥（男）",
                "Andre": "安德雷｜磁性沉稳男声（男）",
                "Radio Gol": "拉迪奥·戈尔｜足球诗人（男）",
                "Jada": "上海-阿珍｜沪上阿姐（女）",
                "Dylan": "北京-晓东｜北京胡同少年（男）",
                "Li": "南京-老李｜耐心瑜伽老师（男）",
                "Marcus": "陕西-秦川｜老陕味道（男）",
                "Roy": "闽南-阿杰｜台湾哥仔（男）",
                "Peter": "天津-李彼得｜天津相声捧哏（男）",
                "Sunny": "四川-晴儿｜川妹子（女）",
                "Eric": "四川-程川｜成都男子（男）",
                "Rocky": "粤语-阿强｜幽默风趣（男）",
                "Kiki": "粤语-阿清｜甜美港妹（女）",
            }
            qwen3_voice_keys = list(qwen3_voice_options.keys())
            current_voice = load_key("qwen3_tts.voice")
            qwen3_voice = st.selectbox(
                t("Qwen3 TTS Voice"),
                options=qwen3_voice_keys,
                format_func=lambda x: f"{x} - {qwen3_voice_options[x]}",
                index=qwen3_voice_keys.index(current_voice) if current_voice in qwen3_voice_keys else 0,
                help=t("Official Qwen/Qwen3 TTS preset voices from Alibaba Cloud Model Studio."),
            )
            if qwen3_voice != current_voice:
                update_key("qwen3_tts.voice", qwen3_voice)
                st.rerun()

            qwen3_languages = [
                "Chinese",
                "English",
                "Japanese",
                "Korean",
                "German",
                "French",
                "Spanish",
                "Italian",
                "Portuguese",
                "Russian",
            ]
            current_language = load_key("qwen3_tts.language_type")
            qwen3_language = st.selectbox(
                t("Qwen3 TTS Language"),
                options=qwen3_languages,
                index=qwen3_languages.index(current_language) if current_language in qwen3_languages else 0,
                help=t("Choose the language matching the input text for better pronunciation."),
            )
            if qwen3_language != current_language:
                update_key("qwen3_tts.language_type", qwen3_language)
                st.rerun()

            qwen3_region = st.selectbox(
                t("Qwen3 TTS Region"),
                options=["beijing", "singapore"],
                index=["beijing", "singapore"].index(load_key("qwen3_tts.region"))
                if load_key("qwen3_tts.region") in ["beijing", "singapore"] else 0,
                help=t("beijing: dashscope.aliyuncs.com; singapore: dashscope-intl.aliyuncs.com"),
            )
            if qwen3_region != load_key("qwen3_tts.region"):
                update_key("qwen3_tts.region", qwen3_region)
                st.rerun()

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            fish_tts_character = st.selectbox(
                t("Fish TTS Character"),
                options=list(load_key("fish_tts.character_id_dict").keys()),
                index=list(load_key("fish_tts.character_id_dict").keys()).index(
                    load_key("fish_tts.character")
                ),
            )
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(
                    load_key("gpt_sovits.refer_mode")
                ),
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")

        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")

        elif select_tts == "mimo_tts":
            config_input(t("Xiaomi MiMo Base URL"), "mimo_tts.base_url",
                         help=t("Default SGP cluster; alt: token-plan-cn.xiaomimimo.com/v1"))
            config_input(t("Xiaomi MiMo API Key"), "mimo_tts.api_key",
                         help=t("Subscription token, form 'tp-xxx'"))
            mimo_model_options = [
                "mimo-v2.5-tts",
                "mimo-v2.5-tts-voicedesign",
                "mimo-v2.5-tts-voiceclone",
            ]
            mimo_cur_model = load_key("mimo_tts.model")
            sel_mimo_model = st.selectbox(
                t("MiMo TTS Model"),
                options=mimo_model_options,
                index=mimo_model_options.index(mimo_cur_model)
                if mimo_cur_model in mimo_model_options
                else 0,
                help=t("preset voice / natural language voice prompt / reference audio clone"),
            )
            if sel_mimo_model != mimo_cur_model:
                update_key("mimo_tts.model", sel_mimo_model)
                st.rerun()
            if sel_mimo_model == "mimo-v2.5-tts":
                # Source: live API response (2026-05-18). Older voices
                # (Sophia/Hannah/Jacob/Owen/Ethan/可乐) have been retired
                # server-side; new additions: mimo_default/苏打/白桦/Mia/Milo/Dean.
                mimo_voices = [
                    "mimo_default",
                    "冰糖", "茉莉", "苏打", "白桦",
                    "Mia", "Chloe",
                    "Milo", "Dean",
                ]
                mimo_cur_voice = load_key("mimo_tts.voice")
                sel_mimo_voice = st.selectbox(
                    t("MiMo Preset Voice"),
                    options=mimo_voices,
                    index=mimo_voices.index(mimo_cur_voice)
                    if mimo_cur_voice in mimo_voices
                    else 0,
                )
                if sel_mimo_voice != mimo_cur_voice:
                    update_key("mimo_tts.voice", sel_mimo_voice)
                    st.rerun()
            elif sel_mimo_model == "mimo-v2.5-tts-voicedesign":
                config_input(
                    t("MiMo Voice Description"),
                    "mimo_tts.voice_description",
                    help=t("Natural-language description of the voice (any language)"),
                )
            elif sel_mimo_model == "mimo-v2.5-tts-voiceclone":
                st.info(t("Voice cloning uses reference audio at output/audio/refers/{number}.wav (auto-extracted by VideoLingo)"))

        elif select_tts == "indextts2":
            # Multi-server support: paste one URL per line (or comma-separated).
            # Each VideoLingo worker thread sticky-binds to one server, so set
            # `TTS Concurrency` >= number of servers for full parallelism.
            try:
                cur_base = load_key("indextts2.base_url")
            except KeyError:
                cur_base = ""
            if isinstance(cur_base, list):
                cur_base_text = "\n".join(str(u).strip() for u in cur_base if str(u).strip())
            elif cur_base is None:
                cur_base_text = ""
            else:
                cur_base_text = str(cur_base)
            new_base_text = st.text_area(
                t("IndexTTS-2 Base URL(s)"),
                value=cur_base_text,
                height=110,
                help=t(
                    "Gradio server URL(s). One per line for multi-server load balancing; "
                    "single URL also works. Example: https://xxx-7860.ap-shanghai2.cloudstudio.club"
                ),
                key="indextts2_base_url_textarea",
                placeholder="https://host-a-7860.example.com\nhttps://host-b-7860.example.com",
            )
            # Persist as list when >1 URL, else as plain string (matches old config).
            parsed_urls = [
                ln.strip()
                for ln in new_base_text.replace(",", "\n").splitlines()
                if ln.strip()
            ]
            new_base_value = parsed_urls if len(parsed_urls) > 1 else (parsed_urls[0] if parsed_urls else "")
            if new_base_value != cur_base:
                update_key("indextts2.base_url", new_base_value)
                st.rerun()

            cur_emo = load_key("indextts2.emo_weight")
            try:
                cur_emo_f = float(cur_emo) if cur_emo is not None else 0.65
            except (TypeError, ValueError):
                cur_emo_f = 0.65
            new_emo = st.slider(
                t("IndexTTS-2 Emotion Weight"),
                min_value=0.0,
                max_value=1.0,
                value=cur_emo_f,
                step=0.05,
                help=t("How strongly the per-sentence original audio drives prosody. 0 = pure timbre, 1 = full emotion follow."),
            )
            if abs(new_emo - cur_emo_f) > 1e-6:
                update_key("indextts2.emo_weight", float(new_emo))
                st.rerun()
            st.info(t("Timbre uses the long reference clip (same as MiMo clone); per-sentence emotion uses output/audio/refers/{number}.wav — both auto-built by VideoLingo."))


def check_api():
    try:
        resp = ask_gpt(
            "This is a test, response 'message':'success' in json format.",
            resp_type="json",
            log_title="None",
        )
        return resp.get("message") == "success"
    except Exception:
        return False


if __name__ == "__main__":
    check_api()
