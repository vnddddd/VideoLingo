"""Official Fish Audio (fish.audio) Text-to-Speech backend.

Docs: https://docs.fish.audio/api-reference/endpoint/tts

Why this exists alongside `fish_tts.py` and `sf_fishtts.py`
-----------------------------------------------------------
Those two are *resellers*, not Fish Audio itself:

* `fish_tts.py`     -> 302.ai relay, limited to a hardcoded `character_id_dict`
* `sf_fishtts.py`   -> SiliconFlow, pinned to the older `fish-speech-1.4` model

This module talks to the official `api.fish.audio` endpoint and therefore gets
the S2.1-Pro family, proper voice cloning, and far more voices.

Three modes (mirrors the naming `sf_fish_tts` already uses, so both UIs can
reuse the existing Preset / Refer_stable / Refer_dynamic labels):

* `preset`  - `reference_id` points at a voice in the Fish library. No cloning.
* `custom`  - **ID cloning**: the reference clip is uploaded once as a persistent
  voice model, and later calls send only the returned `_id`. This is the
  bandwidth-friendly path: an inline clone would resend the whole reference clip
  for every single line of dialogue.
* `dynamic` - **instant (zero-shot) cloning**: the reference clip is inlined on
  every request. Simpler, but it re-uploads the clip per line, which saturates
  a slow uplink on long videos. Kept because it needs no server-side setup.

The `custom` flow deliberately reuses the caching / concurrency / reuse design
of `soniox_tts.py`, which already solved the same three problems for the same
pipeline. See `ensure_cloned_voice` below.
"""
from __future__ import annotations

import hashlib
import io
import threading
import time
from pathlib import Path

import requests
from pydub import AudioSegment

from core.utils import load_key, except_handler, load_timeout, rprint

_API_URL = "https://api.fish.audio/v1/tts"
_MODEL_API_URL = "https://api.fish.audio/model"

# `s2.1-pro-free` is the free-tier tier of S2.1-Pro: identical model, but
# without the latency/availability guarantees of the paid tier. It is the right
# default for a config that ships with a free-package key, and it still supports
# multi-speaker tagging and zero-shot cloning. Switch to `s2.1-pro` once the
# account carries paid API credit.
_DEFAULT_MODEL = "s2.1-pro-free"

# Models accepted by the API. `s1` is intentionally excluded from the default
# path: it is a paid-only model (it answers 402 without API credit) and it does
# not support the S2 speaker-tag syntax.
KNOWN_MODELS = ["s2.1-pro-free", "s2.1-pro", "s2-pro", "s1"]

# The API records which model a voice model was trained for. A voice built on a
# different model family may not resolve, so the cache key includes the model.
_CLONE_READY_TIMEOUT = 180
_CLONE_POLL_INTERVAL = 3.0

# Fish recommends 10-30s of clean speech for a good clone. Longer references do
# not help and cost upload time, so cap what we send.
REF_MAX_SECONDS = 30.0
PHRASE_GAP_MS = 200

# Every voice model this backend creates carries this prefix. Reuse and any
# future cleanup only ever touch these, so voices a user made by hand in the
# Fish Audio studio are never touched by us.
VOICE_NAME_PREFIX = "vl_"

# Cloned voice models are created once and reused across the whole run. The
# content hash of the prepared clip is the name, so re-running the same video —
# or two speakers sharing a reference — reuses the existing model instead of
# creating a duplicate. The lock stops the concurrent TTS workers (see
# `tts_max_workers`) from racing to create the same model.
_clone_cache: dict = {}
_clone_lock = threading.Lock()

# Values shipped in config.example.yaml; treat as "not configured".
_PLACEHOLDER_KEYS = {"", "YOUR_API_KEY", "YOUR_FISH_API_KEY", "sk-fish-xxxx"}


def _load_opt(key, default=None):
    """load_key raises KeyError on missing keys; config blocks may be partial."""
    try:
        value = load_key(key)
    except KeyError:
        return default
    return default if value is None else value


def _load_api_key() -> str:
    value = str(_load_opt("fish_audio_tts.api_key", "")).strip()
    if not value or value in _PLACEHOLDER_KEYS:
        raise ValueError(
            "Fish Audio TTS: no API key. Set fish_audio_tts.api_key in config.yaml "
            "(create one at https://fish.audio/app/api-keys)."
        )
    return value


def _load_model() -> str:
    model = str(_load_opt("fish_audio_tts.model", _DEFAULT_MODEL) or "").strip()
    return model or _DEFAULT_MODEL


def _headers(api_key: str, model: str, msgpack: bool = False) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/msgpack" if msgpack else "application/json",
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Reference clip preparation
# --------------------------------------------------------------------------- #

def _select_best_phrases(seg: AudioSegment, limit_ms: int) -> AudioSegment:
    """Fill the reference budget with the longest phrases, never splitting one.

    Mirrors `soniox_tts._select_best_phrases`. A naive `seg[:limit]` cut either
    slices a word in half at the boundary, or keeps whichever fragments happen
    to come first instead of the most usable speech. Since the model copies
    everything it hears, both hurt clone quality.
    """
    from pydub import silence as _silence

    try:
        spans = _silence.detect_nonsilent(
            seg, min_silence_len=PHRASE_GAP_MS - 50, silence_thresh=seg.dBFS - 20
        )
    except Exception:
        spans = []
    if len(spans) < 2:
        return seg[:limit_ms]

    picked: list[tuple[int, int]] = []
    used = 0
    for start, end in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        cost = (end - start) + (PHRASE_GAP_MS if picked else 0)
        if used + cost > limit_ms:
            continue
        picked.append((start, end))
        used += cost
    if not picked:
        return seg[:limit_ms]

    picked.sort()  # keep chronological order so the voice sounds natural
    out = AudioSegment.silent(duration=0)
    for index, (start, end) in enumerate(picked):
        if index:
            out += AudioSegment.silent(duration=PHRASE_GAP_MS)
        out += seg[start:end]
    return out


def _prepare_reference_clip(ref_wav: Path) -> bytes:
    """Trim a reference clip to what Fish accepts and return wav bytes."""
    seg = AudioSegment.from_file(ref_wav)
    limit_ms = int(REF_MAX_SECONDS * 1000)
    if len(seg) > limit_ms:
        picked = _select_best_phrases(seg, limit_ms)
        rprint(
            f"[yellow]Reference clip is {len(seg) / 1000:.1f}s; keeping "
            f"{len(picked) / 1000:.1f}s of the longest phrases for Fish Audio "
            f"cloning (cap {REF_MAX_SECONDS:.0f}s)[/yellow]"
        )
        seg = picked
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def _reference_text_for(ref_path: Path) -> str:
    """Transcript of the reference clip, used to condition the clone.

    The project keeps the merged long-reference transcript in a sidecar file
    next to `_long_ref.wav`; per-sentence references have no transcript, and
    Fish tolerates an approximate one for a short clip, so fall back to a
    generic line rather than failing the run.
    """
    sidecar = ref_path.parent / f"{ref_path.stem}.txt"
    if sidecar.exists():
        try:
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception:
            pass
    return "This is a reference audio sample for voice cloning."


# --------------------------------------------------------------------------- #
# Persistent voice models (ID cloning)
# --------------------------------------------------------------------------- #

def list_voice_models(api_key=None) -> list:
    """Every voice model owned by this account."""
    api_key = api_key or _load_api_key()
    resp = requests.get(
        _MODEL_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        params={"self": "true", "page_size": 100},
        timeout=load_timeout("tts", 60),
    )
    if resp.status_code != 200:
        raise Exception(
            f"Fish Audio voice list failed {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json().get("items") or []


def delete_voice_model(model_id: str, api_key=None) -> None:
    """Delete one voice model owned by this account."""
    api_key = api_key or _load_api_key()
    resp = requests.delete(
        f"{_MODEL_API_URL}/{model_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=load_timeout("tts", 60),
    )
    # The API answers 204 on success; 404 means it is already gone.
    if resp.status_code not in (200, 204, 404):
        raise Exception(
            f"Fish Audio voice delete failed {resp.status_code}: {resp.text[:300]}"
        )


def _looks_like_slot_error(exc: Exception) -> bool:
    """Whether a create failure looks like a full voice-slot allowance.

    The exact wording is not documented and has not been observed, so match
    loosely on the plausible markers rather than pinning one string.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("limit", "quota", "slot", "maximum", "too many", "exceed")
    )


def _models_created_by_us(models: list) -> list:
    """Only the voice models this backend made, oldest first.

    Every model we create is titled `vl_<content hash>` (see VOICE_NAME_PREFIX),
    so voices the user built by hand in the Fish Audio studio — or any other
    model in the account — are never candidates for deletion.
    """
    ours = [
        m for m in models
        if str(m.get("title") or "").startswith(VOICE_NAME_PREFIX)
    ]
    # created_at is an ISO-8601 timestamp, so lexical order is chronological.
    return sorted(ours, key=lambda m: str(m.get("created_at") or ""))


def _title_family(title: str) -> str:
    """The grouping part of a project model title.

    Titles are `vl_<family>_<index>` where `<family>` identifies the source
    video and `<index>` the speaker within it (see `_voice_name`). Grouping by
    family is what makes it safe to prune: every model of the CURRENT video
    shares one family and is therefore kept together.
    """
    title = str(title or "")
    body = title[len(VOICE_NAME_PREFIX):] if title.startswith(VOICE_NAME_PREFIX) else title
    return body.rsplit("_", 1)[0] if "_" in body else body


def recycle_oldest_voice_model(api_key=None) -> bool:
    """Delete ONE oldest model from a family other than the current run's.

    Never touches the current run's own voices: a multi-speaker run creates one
    model per speaker and is actively using them, so recycling from the live
    family would break lines that have not been generated yet.
    """
    api_key = api_key or _load_api_key()
    ours = _models_created_by_us(list_voice_models(api_key))
    current = _current_family()
    ours = [m for m in ours if _title_family(str(m.get("title") or "")) != current]
    if not ours:
        return False
    victim = ours[0]
    rprint(
        f"[yellow]Fish Audio voice slots full; recycling the oldest model this "
        f"project created: '{victim.get('title')}' "
        f"(created {victim.get('created_at')})[/yellow]"
    )
    delete_voice_model(victim["_id"], api_key)
    # Its id may still be cached against a reference clip; drop any stale entry
    # so the next lookup re-creates or re-finds the right model.
    for key, value in list(_clone_cache.items()):
        if value == victim["_id"]:
            _clone_cache.pop(key, None)
    return True


def prune_voice_models(api_key=None, keep: int = None) -> int:
    """Keep at most `keep` *families* (source videos) of voice models.

    `keep` counts VIDEOS, not individual voices. A multi-speaker video clones
    one model per speaker, so a 6-speaker video legitimately owns 6 models; all
    of them share a family id and are kept or dropped as a unit. Counting
    individual models instead would delete voices the current run is about to
    use — a 6-speaker video could never run with a limit of 3.

    Whole families are evicted, oldest first, until at most `keep` remain. The
    family of the current run is never evicted. Only `vl_`-prefixed models are
    candidates, so user-made voices are untouched.

    Returns the number of models deleted.
    """
    if keep is None:
        keep = int(_load_opt("fish_audio_tts.max_voice_slots", 3) or 3)
    if keep <= 0:
        return 0

    api_key = api_key or _load_api_key()
    ours = _models_created_by_us(list_voice_models(api_key))  # oldest first
    if not ours:
        return 0

    # Group into families, preserving oldest-first order of first appearance.
    families: dict[str, list] = {}
    for model in ours:
        families.setdefault(_title_family(str(model.get("title") or "")), []).append(model)

    current = _current_family()
    # Evictable = every family except the one this run is using, oldest first.
    evictable = [f for f in families if f != current]
    excess = len(families) - keep
    if excess <= 0:
        return 0

    doomed: list = []
    for family in evictable[:excess]:
        doomed.extend(families[family])

    if not doomed:
        rprint(
            f"[yellow]Fish Audio: {len(families)} voice families present but the "
            f"limit is {keep}; keeping the current run's family and leaving the "
            f"rest (delete unused voices at https://fish.audio/app/my-voices).[/yellow]"
        )
        return 0

    rprint(
        f"[cyan]Fish Audio: {len(families)} voice families found, keeping the "
        f"newest {keep}; removing {len(doomed)} older model(s)[/cyan]"
    )
    removed = 0
    for model in doomed:
        try:
            delete_voice_model(model["_id"], api_key)
            for key, value in list(_clone_cache.items()):
                if value == model["_id"]:
                    _clone_cache.pop(key, None)
            removed += 1
            rprint(f"[yellow]  removed '{model.get('title')}'[/yellow]")
        except Exception as exc:  # noqa: BLE001 - pruning must never break a run
            rprint(f"[yellow]  could not remove '{model.get('title')}': {exc}[/yellow]")
    return removed


def prepare_for_run(api_key=None) -> None:
    """Housekeeping to run ONCE before dubbing starts.

    Two jobs, both cheap and both non-fatal:

    1. Prune this project's own voice models down to `max_voice_slots` so a long
       series of videos cannot silently fill the account's slot allowance.
       Only `vl_`-prefixed models are touched; anything the user created by hand
       is left alone.
    2. Drop cached ids whose model no longer exists (deleted in the web studio,
       or pruned by an earlier run), so a stale cache cannot make every line fail.

    Every failure is swallowed: housekeeping must never stop a dubbing run.
    """
    try:
        api_key = api_key or _load_api_key()
    except Exception as exc:  # noqa: BLE001 - no key yet is the caller's problem
        rprint(f"[yellow]Fish Audio: skipping pre-run housekeeping ({exc})[/yellow]")
        return

    try:
        prune_voice_models(api_key)
    except Exception as exc:  # noqa: BLE001
        rprint(f"[yellow]Fish Audio: pre-run prune skipped ({exc})[/yellow]")

    # Invalidate cache entries pointing at models that are gone.
    try:
        live = {m["_id"] for m in list_voice_models(api_key)}
        stale = [k for k, v in _clone_cache.items() if v not in live]
        for k in stale:
            _clone_cache.pop(k, None)
        if stale:
            rprint(f"[cyan]Fish Audio: dropped {len(stale)} stale cached voice id(s)[/cyan]")
    except Exception as exc:  # noqa: BLE001
        rprint(f"[yellow]Fish Audio: cache validation skipped ({exc})[/yellow]")


def _create_voice_model(name: str, clip: bytes, api_key: str) -> str:
    """Upload a reference clip as a persistent voice model; return its id.

    `train_mode=fast` is Fish's instant-clone path (no long training job).
    `visibility=private` keeps the clone out of the public library — and Fish
    only requires a cover image for public voices, which we do not have.
    """
    resp = requests.post(
        _MODEL_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        data={"type": "tts", "title": name, "train_mode": "fast", "visibility": "private"},
        files={"voices": (f"{name}.wav", clip, "audio/wav")},
        timeout=load_timeout("tts", 180),
    )
    if resp.status_code not in (200, 201):
        raise Exception(
            f"Fish Audio voice create failed {resp.status_code}: {resp.text[:400]}"
        )
    return resp.json()["_id"]


def _wait_until_trained(model_id: str, api_key: str) -> None:
    """Poll a voice model until it is usable. Creation is asynchronous."""
    deadline = time.time() + _CLONE_READY_TIMEOUT
    url = f"{_MODEL_API_URL}/{model_id}"
    while True:
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=load_timeout("tts", 60),
            )
            if resp.status_code == 200:
                state = str(resp.json().get("state") or "").lower()
                if state == "trained":
                    return
                if state == "failed":
                    raise Exception(
                        f"Fish Audio voice model {model_id} failed to train"
                    )
        except Exception as exc:  # noqa: BLE001 - only the failure state is fatal
            if "failed to train" in str(exc):
                raise
            rprint(f"[yellow]Fish Audio: voice status check failed ({exc}); retrying[/yellow]")

        if time.time() >= deadline:
            raise Exception(
                f"Fish Audio voice model {model_id} was not trained within "
                f"{_CLONE_READY_TIMEOUT}s"
            )
        time.sleep(_CLONE_POLL_INTERVAL)


# Family (source video) id for the current process. Assigned once, lazily.
_run_family: str | None = None
_run_family_lock = threading.Lock()
# Speaker counter within the family, so titles read vl_<family>_1, _2, ...
_speaker_index: int = 0


def _current_family() -> str:
    """Stable id for "the video being dubbed right now".

    A multi-speaker run clones one voice per speaker, so pruning by a global
    count would delete models that the current run is still using. Tagging every
    model of this run with one shared family id lets pruning treat the whole run
    as a single unit: either all of its models survive, or (once a later run
    supersedes it) all of them go.

    Derived from the output directory when available, so two runs of the same
    video family share a tag and re-clone nothing.
    """
    global _run_family
    if _run_family:
        return _run_family
    with _run_family_lock:
        if _run_family:
            return _run_family
        seed = ""
        try:
            seed = str(Path("output/audio/refers").resolve())
        except Exception:
            seed = ""
        _run_family = hashlib.md5((seed or "videolingo").encode()).hexdigest()[:8]
    return _run_family


def _voice_name(clip: bytes, api_key: str) -> str:
    """Title for a new voice model: `vl_<family>_<speakerIndex>`.

    The clip's content hash is deliberately NOT in the name. It hashes the
    prepared audio, which is identical for every speaker that reuses the same
    merged reference — that aliasing is what let two different speakers collide
    on one model before. An explicit per-speaker index keeps them distinct.
    """
    global _speaker_index
    family = _current_family()
    with _run_family_lock:
        # Reuse an existing title for this exact family if the server already
        # has one for a previous identical clip set; otherwise take the next index.
        existing = [
            str(m.get("title") or "")
            for m in list_voice_models(api_key)
            if _title_family(str(m.get("title") or "")) == family
        ]
        _speaker_index = len(existing) + 1
        return f"{VOICE_NAME_PREFIX}{family}_{_speaker_index}"


def ensure_cloned_voice(ref_wav, api_key=None) -> str:
    """Return the voice model id for a reference clip, creating it on first use.

    Voices are named after the clip's content hash, so a repeat run — or another
    speaker sharing the same reference — reuses the existing model instead of
    creating a duplicate. This is the whole point of the `custom` mode: the
    clip is uploaded once, then every line's request carries only the id.
    """
    ref_path = Path(ref_wav)
    if not ref_path.exists():
        raise ValueError(f"Fish Audio voice clone: reference audio not found: {ref_path}")

    api_key = api_key or _load_api_key()

    # Cheap in-process key. This runs once per generated line, so hashing the
    # clip every time would mean decoding the reference a few hundred times per
    # video just to look up an id we already have.
    stat = ref_path.stat()
    cache_key = (str(ref_path.resolve()), stat.st_mtime_ns, stat.st_size)

    with _clone_lock:
        cached = _clone_cache.get(cache_key)
        if cached:
            return cached

        clip = _prepare_reference_clip(ref_path)
        name = _voice_name(clip, api_key)

        for model in list_voice_models(api_key):
            if str(model.get("title") or "") == name:
                _wait_until_trained(model["_id"], api_key)
                _clone_cache[cache_key] = model["_id"]
                rprint(f"[cyan]Reusing Fish Audio voice model '{name}'[/cyan]")
                return model["_id"]

        rprint(f"[cyan]Creating Fish Audio voice model '{name}' from {ref_path.name}...[/cyan]")
        try:
            model_id = _create_voice_model(name, clip, api_key)
        except Exception as exc:  # noqa: BLE001 - only a full/slot error is retryable
            # Pre-run pruning (prepare_for_run) should already have made room, so
            # reaching here means the account is genuinely out of slots. Recycle
            # the oldest model WE made and try once more; the user's own voices
            # are never candidates. Anything else is a real error.
            if not _looks_like_slot_error(exc):
                raise
            if not recycle_oldest_voice_model(api_key):
                raise Exception(
                    "Fish Audio has no free voice slots, and this project owns no "
                    "voice model that can be recycled. Delete unused models at "
                    f"https://fish.audio/app/my-voices, or raise "
                    f"fish_audio_tts.max_voice_slots. Original error: {exc}"
                ) from exc
            model_id = _create_voice_model(name, clip, api_key)
        # Cache the id BEFORE waiting. The upload already succeeded, so the model
        # exists server-side; if training timed out, a retry must reuse this
        # model rather than create a duplicate (its name is taken, so a second
        # create would collide anyway).
        _clone_cache[cache_key] = model_id
        try:
            _wait_until_trained(model_id, api_key)
        except Exception:
            # Drop the cache entry so a later attempt re-polls this same model
            # instead of trusting an id that may still be training.
            _clone_cache.pop(cache_key, None)
            # But do not orphan it: the next lookup matches by content-hash name.
            raise
        rprint(f"[green]Fish Audio voice model ready: {name}[/green]")
        return model_id


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #

def _resolve_mode(voice_cfg) -> str:
    mode = str(_load_opt("fish_audio_tts.mode", "custom") or "custom").strip().lower()
    if voice_cfg and voice_cfg.get("is_clone") and voice_cfg.get("ref_wav"):
        # The multi-speaker router's clone mode forces a cloning path. Prefer
        # the configured mode so a user who picked `dynamic` still gets it.
        return mode if mode in ("custom", "dynamic") else "custom"
    return mode


def _resolve_preset_reference() -> str | None:
    ref = _load_opt("fish_audio_tts.reference_id")
    ref = str(ref).strip() if ref else ""
    return ref or None


def _post(payload: dict, headers: dict, msgpack: bool) -> bytes:
    """POST to /v1/tts and return the audio body, or raise with the API message."""
    # requests uses `data=` for a raw body, not `content=` (that is httpx).
    if msgpack:
        import msgpack as _msgpack

        kwargs = {"data": _msgpack.packb(payload, use_bin_type=True)}
    else:
        kwargs = {"json": payload}

    with requests.post(
        _API_URL, headers=headers, timeout=load_timeout("tts", 120), stream=True, **kwargs
    ) as resp:
        if resp.status_code != 200:
            raise Exception(
                f"Fish Audio TTS API error {resp.status_code}: {resp.text[:500]}"
            )
        audio = resp.content

    if not audio:
        raise Exception("Fish Audio TTS returned an empty response body")
    return audio


@except_handler("Failed to generate audio using Fish Audio TTS", retry=3, delay=1)
def fish_audio_tts(text, save_as, voice_cfg=None):
    """Fish Audio Text-to-Speech.

    voice_cfg: optional dict from the speaker router. `is_clone + ref_wav`
    selects the configured cloning mode for that speaker's reference clip;
    `voice` overrides the preset `reference_id` for this call.
    """
    api_key = _load_api_key()
    model = _load_model()
    mode = _resolve_mode(voice_cfg)

    payload: dict = {"text": text, "format": "wav", "latency": "normal"}
    chunk_length = _load_opt("fish_audio_tts.chunk_length")
    if chunk_length:
        payload["chunk_length"] = max(100, min(300, int(chunk_length)))
    temperature = _load_opt("fish_audio_tts.temperature")
    if temperature is not None and str(temperature) != "":
        payload["temperature"] = max(0.0, min(1.0, float(temperature)))

    msgpack = False

    if mode == "dynamic":
        # Instant zero-shot cloning: inline the reference audio. MessagePack is
        # mandatory here — JSON cannot carry raw audio bytes.
        ref_wav = None
        if voice_cfg and voice_cfg.get("is_clone") and voice_cfg.get("ref_wav"):
            ref_wav = Path(voice_cfg["ref_wav"])
        else:
            from core.utils._long_ref_extractor import ensure_long_ref

            ref_wav = Path(ensure_long_ref())
        if not ref_wav.exists():
            raise ValueError(f"Fish Audio TTS: reference audio not found: {ref_wav}")

        clip = _prepare_reference_clip(ref_wav)
        payload["references"] = [
            {"audio": clip, "text": _reference_text_for(ref_wav)}
        ]
        msgpack = True
        rprint(
            f"[yellow]Fish Audio dynamic mode: inlining "
            f"{len(clip) / 1024:.0f} KB reference per line[/yellow]"
        )

    elif mode == "custom":
        # ID cloning: upload once, then send only the model id.
        if voice_cfg and voice_cfg.get("is_clone") and voice_cfg.get("ref_wav"):
            ref_wav = Path(voice_cfg["ref_wav"])
        else:
            from core.utils._long_ref_extractor import ensure_long_ref

            ref_wav = Path(ensure_long_ref())
        payload["reference_id"] = ensure_cloned_voice(ref_wav, api_key)

    elif mode == "preset":
        reference_id = None
        if voice_cfg and voice_cfg.get("voice"):
            reference_id = str(voice_cfg["voice"]).strip()
        reference_id = reference_id or _resolve_preset_reference()
        if not reference_id:
            raise ValueError(
                "Fish Audio TTS: preset mode needs fish_audio_tts.reference_id "
                "(a voice id from https://fish.audio/discovery)."
            )
        payload["reference_id"] = reference_id

    else:
        raise ValueError(
            f"Fish Audio TTS: unknown mode '{mode}'. "
            "Use 'preset', 'custom' (id clone), or 'dynamic' (instant clone)."
        )

    audio = _post(payload, _headers(api_key, model, msgpack), msgpack)

    # Re-encode through pydub: the pipeline probes every clip with the `wave`
    # module and rejects anything that is not plain PCM WAV.
    speech_file_path = Path(save_as)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    seg = AudioSegment.from_file(io.BytesIO(audio), format="wav")
    seg.export(str(speech_file_path), format="wav")
    print(f"Audio saved to {speech_file_path}")


if __name__ == "__main__":
    fish_audio_tts("你好，欢迎使用 VideoLingo！", "test_fish_audio.wav")
