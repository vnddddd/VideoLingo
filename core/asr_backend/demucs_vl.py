import os
import shutil
import torch
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import save_audio
from torch.cuda import is_available as is_cuda_available
from typing import Optional
from demucs.api import Separator
from demucs.apply import BagOfModels
import gc
from core.utils.models import *
from core.utils.config_utils import load_key


class PreloadedSeparator(Separator):
    def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)


def _load_key_default(key, default):
    """load_key wrapper: returns default on KeyError (forward-compat with old config.yaml)."""
    try:
        v = load_key(key)
    except KeyError:
        return default
    return v if v is not None else default


def demucs_audio():
    """Dispatcher: separate vocals from background music.

    Backend selected by config.yaml `demucs_backend`:
      - 'local'    (default): run htdemucs on this machine (GPU/MPS/CPU)
      - 'hf_space': offload to a HuggingFace Space (recommended for old GPUs like GT 1030)
    """
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return

    backend = str(_load_key_default("demucs_backend", "local")).strip().lower()
    if backend == "hf_space":
        _demucs_via_hf_space()
    else:
        _demucs_local()


def _demucs_local():
    """Original on-machine demucs separation (GPU/MPS/CPU)."""
    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    console.print("🤖 Loading <htdemucs> model...")
    model = get_model('htdemucs')
    separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)

    console.print("🎵 Separating audio...")
    _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)

    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2,
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}

    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)

    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)

    # Clean up memory
    del outputs, background, model, separator
    gc.collect()

    console.print("[green]✨ Audio separation completed![/green]")


def _demucs_via_hf_space():
    """Offload separation to a HuggingFace Space.

    Designed for `abidlabs/music-separation` (T4 GPU, same htdemucs model).
    Endpoint: /predict(audio: filepath) -> (vocals: filepath, no_vocals: filepath)
    """
    try:
        from gradio_client import Client, handle_file
    except ImportError as e:
        raise RuntimeError(
            "gradio_client is required for demucs_backend='hf_space'. "
            "Install with: pip install gradio_client"
        ) from e

    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    space_id = _load_key_default("hf_demucs.space_id", "abidlabs/music-separation")
    api_name = _load_key_default("hf_demucs.api_name", "/predict")
    hf_token = (_load_key_default("hf_demucs.hf_token", "") or "").strip()
    if not hf_token:
        raise RuntimeError(
            "demucs_backend='hf_space' selected but hf_demucs.hf_token is empty in config.yaml.\n"
            "  -> Get a free token at https://huggingface.co/settings/tokens (read scope is enough),\n"
            "  -> then paste it into config.yaml under hf_demucs.hf_token."
        )

    console.print(f"☁️  Connecting to HF Space [cyan]{space_id}[/cyan]...")
    # gradio_client v2.5.0 uses `token=`, while v4/v5 renamed it to `hf_token=`. Probe to stay forward-compat.
    import inspect
    _client_params = inspect.signature(Client.__init__).parameters
    if "hf_token" in _client_params:
        client = Client(space_id, hf_token=hf_token)
    else:
        client = Client(space_id, token=hf_token)

    console.print(f"📤 Uploading [cyan]{_RAW_AUDIO_FILE}[/cyan] to Space and separating (this may take ~30-90s)...")
    result = client.predict(audio=handle_file(_RAW_AUDIO_FILE), api_name=api_name)

    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError(f"Unexpected response from Space {space_id}: {type(result).__name__} = {result!r}")

    def _to_path(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("path") or x.get("name") or x.get("url")
        return str(x)

    vocals_src = _to_path(result[0])
    background_src = _to_path(result[1])

    console.print(f"🎤 Saving vocals track <- {vocals_src}")
    shutil.copyfile(vocals_src, _VOCAL_AUDIO_FILE)

    console.print(f"🎹 Saving background music <- {background_src}")
    shutil.copyfile(background_src, _BACKGROUND_AUDIO_FILE)

    console.print("[green]✨ Audio separation completed (via HF Space)![/green]")


if __name__ == "__main__":
    demucs_audio()
