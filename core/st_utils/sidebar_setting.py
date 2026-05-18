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


def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


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
            "fish_tts",
            "sf_fish_tts",
            "edge_tts",
            "gpt_sovits",
            "custom_tts",
            "sf_cosyvoice2",
            "f5tts",
            "mimo_tts",
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
                mimo_voices = [
                    "Chloe", "Sophia", "Hannah",
                    "Jacob", "Owen", "Ethan",
                    "冰糖", "茉莉", "可乐",
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
