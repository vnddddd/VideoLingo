"""
ffmpeg encoder helper - supports CPU / NVENC / QSV / AMF

By default, falls back to libx264 (CPU). Users can opt into hardware
acceleration by setting `ffmpeg_hwaccel` in config.yaml.

Config fields (all optional, sensible defaults applied):
    ffmpeg_hwaccel: cpu | nvenc | qsv | amf | auto
        - cpu  : libx264 (default, most compatible)
        - nvenc: NVIDIA GPUs with NVENC (GTX 10xx+ except GT 1030/MX series)
        - qsv  : Intel iGPU Quick Sync Video (Gen 7+ / Haswell+)
        - amf  : AMD GPUs (Polaris+) on Windows
        - auto : probe at runtime, prefer nvenc > qsv > amf > cpu
    ffmpeg_preset: ultrafast..veryslow (default: medium)
    ffmpeg_quality: 18..28 CRF/CQ value (default: 23)

Backward compatibility:
    Legacy field `ffmpeg_gpu: true` is treated as `ffmpeg_hwaccel: nvenc`
    when the new field is not set. This keeps old configs working.
"""
import subprocess
from core.utils.config_utils import load_key

_encoder_cache = None


def _safe_load(key, default):
    """load_key raises KeyError on missing keys; we want graceful defaults."""
    try:
        v = load_key(key)
        if v is None or v == '':
            return default
        return v
    except (KeyError, Exception):
        return default


def detect_best_encoder():
    """Probe hardware encoders by running a tiny 1-frame encode test.

    Returns one of: 'nvenc', 'qsv', 'amf', 'cpu'
    Result is cached for the process lifetime.
    """
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    for codec, name in [('h264_nvenc', 'nvenc'),
                        ('h264_qsv', 'qsv'),
                        ('h264_amf', 'amf')]:
        try:
            r = subprocess.run(
                ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                 '-f', 'lavfi', '-i', 'nullsrc=s=320x240:d=0.1',
                 '-c:v', codec, '-frames:v', '1', '-f', 'null', '-'],
                capture_output=True, timeout=15)
            if r.returncode == 0:
                _encoder_cache = name
                return name
        except Exception:
            continue
    _encoder_cache = 'cpu'
    return 'cpu'


# Map libx264 presets to QSV preset names (mostly identical, normalize edge cases)
_QSV_PRESET_MAP = {
    'ultrafast': 'veryfast', 'superfast': 'veryfast',
    'veryfast': 'veryfast', 'faster': 'faster', 'fast': 'fast',
    'medium': 'medium', 'slow': 'slow', 'slower': 'slower', 'veryslow': 'veryslow',
}


def get_video_encoder_args():
    """Return (args_list, encoder_name) for ffmpeg -c:v ... selection.

    Reads config.yaml fields:
        ffmpeg_hwaccel  (new, preferred)
        ffmpeg_gpu      (legacy fallback)
        ffmpeg_preset
        ffmpeg_quality
    """
    hwaccel = str(_safe_load('ffmpeg_hwaccel', '')).lower().strip()
    preset = str(_safe_load('ffmpeg_preset', 'medium')).lower().strip()
    quality = str(_safe_load('ffmpeg_quality', 23))

    # Legacy compat: ffmpeg_gpu: true => nvenc (old behavior)
    if not hwaccel:
        hwaccel = 'nvenc' if _safe_load('ffmpeg_gpu', False) else 'cpu'

    if hwaccel == 'auto':
        hwaccel = detect_best_encoder()

    if hwaccel == 'nvenc':
        return ['-c:v', 'h264_nvenc', '-preset', preset, '-cq', quality], 'nvenc'

    if hwaccel == 'qsv':
        qsv_preset = _QSV_PRESET_MAP.get(preset, 'medium')
        return ['-c:v', 'h264_qsv', '-preset', qsv_preset,
                '-global_quality', quality], 'qsv'

    if hwaccel == 'amf':
        # AMF uses -quality {speed,balanced,quality} not preset
        amf_q = 'speed' if preset in ('ultrafast', 'superfast', 'veryfast',
                                       'faster', 'fast') else 'balanced'
        return ['-c:v', 'h264_amf', '-quality', amf_q,
                '-rc', 'cqp', '-qp_i', quality, '-qp_p', quality], 'amf'

    # Default: CPU libx264
    return ['-c:v', 'libx264', '-preset', preset, '-crf', quality], 'cpu'


if __name__ == '__main__':
    # Self-test
    args, name = get_video_encoder_args()
    print(f"Encoder: {name}")
    print(f"Args:    {' '.join(args)}")
    print(f"Probe:   {detect_best_encoder()}")
