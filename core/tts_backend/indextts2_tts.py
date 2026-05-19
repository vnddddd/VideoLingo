"""IndexTTS-2 remote backend for VideoLingo.

Calls a self-hosted IndexTTS-2 Gradio demo (https://github.com/index-tts/index-tts)
via gradio_client. The remote server is expected to expose the official
``/gen_single`` endpoint as shipped by IndexTeam's Demo Space.

Why this backend exists
-----------------------
MiMo's voice clone keeps timbre but is poor at carrying source emotion. IndexTTS-2
accepts **two** independent audio references in one call:

  * ``prompt``       — timbre reference (whose voice it sounds like)
  * ``emo_ref_path`` — emotion / prosody reference (how it should be spoken)

We map them to VideoLingo's existing artifacts:

  * timbre   = ``output/audio/refers/_long_ref.wav`` built by
               :pyfunc:`core.utils._long_ref_extractor.ensure_long_ref` — a
               20–30 s concatenation of the speaker's longest legacy clips,
               same source MiMo's voiceclone branch already uses.
  * emotion  = ``output/audio/refers/{number}.wav`` produced by
               :pymod:`core._9_refer_audio` — the per-sentence original-audio
               slice that VideoLingo already cuts for every translated line.

No new slicing / pipeline step is required.

Config (``config.yaml``)
------------------------
    tts_method: 'indextts2'
    indextts2:
      # base_url accepts EITHER a single URL string OR a list (for multi-server
      # load balancing). When multiple URLs are provided, each VideoLingo
      # worker thread is sticky-bound to one server on first use, so set
      # `tts_max_workers` >= number of servers for full parallelism.
      #
      # Single server:
      #   base_url: 'https://<your-host>'
      # Multiple servers (any of these three forms works):
      #   base_url: ['https://host-a', 'https://host-b']
      #   base_url: "https://host-a, https://host-b"          # comma-separated
      #   base_url: |                                          # YAML block
      #     https://host-a
      #     https://host-b
      emo_weight: 0.65                  # 0..1, how strongly the emo clip drives prosody

Failure policy
--------------
Any missing reference / API error raises immediately — there is **no silent
fallback to the legacy per-sentence clone** (matches the explicit "hard-fail"
policy the user established for ``_long_ref``).
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

from core.utils import load_key

# ---------------------------------------------------------------------------
# Per-thread Client cache + multi-server sticky load balancing.
#
# VideoLingo's `generate_tts_audio` uses a ThreadPoolExecutor (max_workers =
# config `tts_max_workers`). Two costs we want to amortize across sentences:
#   1. `gradio_client.Client(...)` runs a `view_api` probe (~200-500 ms on a
#      cloudstudio reverse proxy). Cache one Client per worker thread.
#   2. With several IndexTTS-2 servers available, we want each worker pinned
#      to a single server (sticky) so 4 workers x 4 servers gives 1:1 fan-out
#      without any per-call URL juggling.
#
# `threading.local` gives each worker its own attribute namespace -- two
# threads never share the same Client object, which avoids any internal
# session/event-id contention inside gradio_client.
# ---------------------------------------------------------------------------
_thread_local = threading.local()
_assign_lock = threading.Lock()
_next_worker_idx = 0   # global round-robin counter for first-time assignment


def _parse_base_urls(raw) -> list[str]:
    """Normalize the ``indextts2.base_url`` config value to a list of URLs.

    Accepts:
      * list / tuple of strings  -> used as-is (whitespace stripped, empties dropped)
      * single string            -> split on newlines AND commas, so users may
                                    write either ``"a, b"`` or a YAML block scalar
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(u).strip() for u in raw]
    else:
        # treat as string; replace commas with newlines then split
        items = [seg.strip() for seg in str(raw).replace(",", "\n").splitlines()]
    return [u for u in items if u]


def _resolve_base_url() -> str:
    """Pick the IndexTTS-2 server URL for the current thread (sticky).

    The first time a given worker thread calls this, it atomically claims the
    next URL from the configured list (round-robin via a global counter under
    a lock). All subsequent calls from the same thread return the same URL,
    so the per-thread Client cache in :func:`_get_client` stays warm and the
    request stream from any one worker always hits the same backend.
    """
    raw = _cfg("indextts2.base_url", None)
    urls = _parse_base_urls(raw)
    if not urls:
        raise RuntimeError(
            "indextts2.base_url is not configured; set it in config.yaml under "
            "`indextts2:` before selecting tts_method: 'indextts2'. "
            "Accepts a single URL string or a list of URLs."
        )

    # Sticky binding: if this thread already picked one AND that URL is still
    # in the current config, reuse it.
    assigned = getattr(_thread_local, "assigned_url", None)
    if assigned in urls:
        return assigned

    if len(urls) == 1:
        url = urls[0]
    else:
        global _next_worker_idx
        with _assign_lock:
            idx = _next_worker_idx
            _next_worker_idx += 1
        url = urls[idx % len(urls)]

    _thread_local.assigned_url = url
    return url


def _get_client(base_url: str):
    """Return a thread-local ``gradio_client.Client`` for ``base_url``.

    Safe for concurrent use because ``threading.local`` gives each worker its
    own attribute namespace -- two threads never share the same Client object,
    which avoids any internal session/event-id contention inside gradio_client.
    """
    cached_url = getattr(_thread_local, "base_url", None)
    cached_client = getattr(_thread_local, "client", None)
    if cached_client is not None and cached_url == base_url:
        return cached_client

    try:
        from gradio_client import Client
    except ImportError as e:
        raise ImportError(
            "IndexTTS-2 backend requires the `gradio_client` package. "
            "Install it inside the VideoLingo environment, e.g.\n"
            "    .venv\\Scripts\\python -m pip install gradio_client"
        ) from e

    # gradio_client 2.5+ supports httpx_kwargs to override the default 60s httpx
    # timeout. T4 inference + cloudstudio reverse-proxy + multi-worker concurrent
    # file uploads can easily exceed 60s on a saturated home uplink, surfacing as
    # ReadTimeout / WriteTimeout on the client and ClientDisconnect on the server.
    # 300s gives plenty of headroom; user can override via `indextts2.timeout`.
    timeout_s = float(_cfg("indextts2.timeout", 300))
    # download_files=True so result is auto-pulled to a local temp path
    client = Client(
        base_url,
        verbose=False,
        download_files=True,
        httpx_kwargs={"timeout": timeout_s},
    )
    _thread_local.base_url = base_url
    _thread_local.client = client
    return client

# ---------------------------------------------------------------------------
# Defaults — must stay in sync with the IndexTTS-2 Demo `/gen_single` schema.
# Verified against the live cloudstudio instance on 2026-05-19 via
# `gradio_client.Client.view_api`. advanced_params in webui.py:
#   [do_sample, top_p, top_k, temperature,
#    length_penalty, num_beams, repetition_penalty, max_mel_tokens]
# ---------------------------------------------------------------------------
DEFAULT_EMO_WEIGHT = 0.65

ADV_DEFAULTS = dict(
    do_sample=True,
    top_p=0.8,
    top_k=30,
    temperature=0.8,
    length_penalty=0.0,
    num_beams=3,
    repetition_penalty=10.0,
    max_mel_tokens=1500,
)

MAX_TEXT_TOKENS_PER_SEGMENT = 120

EMO_METHOD_AUDIO = "Use emotion reference audio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg(key: str, default):
    """Read a nested config key with safe fallback."""
    try:
        v = load_key(key)
    except Exception:
        return default
    if v in (None, "", {}):
        return default
    return v


def _resolve_timbre_ref(voice_cfg) -> Path:
    """Return the wav path used as IndexTTS-2 ``prompt`` (timbre)."""
    # Multi-speaker override: per-speaker ref clip prepared upstream
    if voice_cfg and voice_cfg.get("is_clone") and voice_cfg.get("ref_wav"):
        p = Path(voice_cfg["ref_wav"])
        if not p.exists():
            raise FileNotFoundError(
                f"IndexTTS-2 multi-speaker ref_wav missing: {p}"
            )
        return p

    # Single-speaker path: reuse the global long reference (same as MiMo clone)
    from core.utils._long_ref_extractor import ensure_long_ref

    p = Path(ensure_long_ref())
    if not p.exists():
        raise FileNotFoundError(
            f"IndexTTS-2 timbre reference (_long_ref) missing after build: {p}"
        )
    return p


def _resolve_emo_ref(number) -> Path:
    """Return the per-sentence original-audio slice used as emotion reference."""
    refers_dir = Path.cwd() / "output" / "audio" / "refers"
    emo = refers_dir / f"{number}.wav"
    if emo.exists():
        return emo

    # First-time run: trigger VideoLingo's standard extractor once
    from core._9_refer_audio import extract_refer_audio_main

    print(f"参考音频文件不存在，触发提取: {emo}")
    extract_refer_audio_main()

    if not emo.exists():
        raise FileNotFoundError(
            f"IndexTTS-2 emotion reference still missing after extract_refer_audio_main: {emo}"
        )
    return emo


def _unwrap_result(result):
    """gradio_client returns various shapes; squeeze out the local file path."""
    out = result
    if isinstance(out, (list, tuple)) and out:
        out = out[0]
    if isinstance(out, dict):
        out = out.get("path") or out.get("name") or out.get("value")
    return out


# ---------------------------------------------------------------------------
# Public entry point — signature matches the other *_tts_for_videolingo so
# tts_main dispatches uniformly.
# ---------------------------------------------------------------------------
def indextts2_tts_for_videolingo(text, save_as, number, task_df=None, voice_cfg=None):
    """Synthesize one sentence via remote IndexTTS-2 ``/gen_single``."""
    # Pick a server for this worker thread. Sticky after first call, so the
    # per-thread Client cache (see _get_client) stays warm. Raises if the
    # config has no URLs at all.
    base_url = _resolve_base_url()
    emo_weight = float(_cfg("indextts2.emo_weight", DEFAULT_EMO_WEIGHT))

    timbre_path = _resolve_timbre_ref(voice_cfg)
    emo_path = _resolve_emo_ref(number)

    try:
        from gradio_client import handle_file
    except ImportError as e:
        raise ImportError(
            "IndexTTS-2 backend requires the `gradio_client` package. "
            "Install it inside the VideoLingo environment, e.g.\n"
            "    .venv\\Scripts\\python -m pip install gradio_client"
        ) from e

    # Per-thread cached client; first call in a worker pays the view_api probe,
    # subsequent sentences in the same worker reuse the warm Client.
    client = _get_client(base_url)

    result = client.predict(
        EMO_METHOD_AUDIO,                       # 0 emo_control_method
        handle_file(str(timbre_path)),          # 1 prompt (timbre)
        text,                                   # 2 text
        handle_file(str(emo_path)),             # 3 emo_ref_path
        emo_weight,                             # 4 emo_weight
        0.0, 0.0, 0.0, 0.0,                     # 5-8  vec1..vec4 (unused in audio mode)
        0.0, 0.0, 0.0, 0.0,                     # 9-12 vec5..vec8
        "",                                     # 13 emo_text
        False,                                  # 14 emo_random
        MAX_TEXT_TOKENS_PER_SEGMENT,            # 15 max_text_tokens_per_segment
        ADV_DEFAULTS["do_sample"],              # 16 param_16
        ADV_DEFAULTS["top_p"],                  # 17 param_17
        ADV_DEFAULTS["top_k"],                  # 18 param_18
        ADV_DEFAULTS["temperature"],            # 19 param_19
        ADV_DEFAULTS["length_penalty"],         # 20 param_20
        ADV_DEFAULTS["num_beams"],              # 21 param_21
        ADV_DEFAULTS["repetition_penalty"],     # 22 param_22
        ADV_DEFAULTS["max_mel_tokens"],         # 23 param_23
        api_name="/gen_single",
    )

    out_file = _unwrap_result(result)
    if not out_file or not Path(out_file).exists():
        raise RuntimeError(
            f"IndexTTS-2 returned no usable audio file. Raw result: {result!r}"
        )

    save_as = Path(save_as)
    save_as.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_file, save_as)
    return str(save_as)
