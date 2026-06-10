import os,sys
import glob
import re
import subprocess
from core.utils import *

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL

def _apply_cookie_file(ydl_opts, cookies_path=None):
    if cookies_path is None:
        try:
            cookies_path = load_key("youtube.cookies_path")
        except Exception:
            cookies_path = None
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    os.makedirs(save_path, exist_ok=True)
    # Format selection: prefer H.264 (avc1) over AV1/VP9.
    #
    # YouTube offers the same video in AV1 / VP9 / H.264; yt-dlp defaults to
    # picking AV1 because it has the best compression. But AV1 hardware
    # decode requires Intel 11th-gen (Tiger Lake/Xe) or NVIDIA RTX 30-series
    # (Ampere) and newer. On older iGPUs/dGPUs (e.g. UHD 630, GT 1030) AV1
    # falls back to libaom-av1 CPU software decode, which becomes the
    # bottleneck for the later subtitle-burn / encode steps (_7, _12) and
    # leaves the GPU encoder starved at ~50% utilization.
    #
    # Falling back chain (yt-dlp picks the first that matches):
    #   1. avc1 video + audio at the requested resolution (best case: full
    #      GPU pipeline downstream)
    #   2. any video + audio at the requested resolution (e.g. AV1-only
    #      uploads, rare on YouTube but possible on other sites)
    #   3. best single stream at the requested resolution (last resort)
    if resolution == 'best':
        fmt = (
            'bestvideo[vcodec^=avc1]+bestaudio/'
            'bestvideo+bestaudio/'
            'best'
        )
    else:
        fmt = (
            f'bestvideo[vcodec^=avc1][height<={resolution}]+bestaudio/'
            f'bestvideo[height<={resolution}]+bestaudio/'
            f'best[height<={resolution}]'
        )
    ydl_opts = {
        'format': fmt,
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
    }

    # Read Youtube Cookie File
    _apply_cookie_file(ydl_opts)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def download_audio_ytdlp(url, save_path='output', cookies_path=None):
    """Download only the best audio stream and prepare output/audio/raw.mp3."""
    audio_dir = os.path.join(save_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    raw_audio = os.path.join(audio_dir, "raw.mp3")
    if os.path.exists(raw_audio):
        os.remove(raw_audio)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(audio_dir, 'source.%(ext)s'),
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
    }
    _apply_cookie_file(ydl_opts, cookies_path=cookies_path)

    YoutubeDL = update_ytdlp()
    before = set(os.listdir(audio_dir))
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    candidates = []
    for file in os.listdir(audio_dir):
        path = os.path.join(audio_dir, file)
        if not os.path.isfile(path):
            continue
        if file == "raw.mp3":
            continue
        if file not in before or file.startswith("source."):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce an audio source file")

    source_audio = max(candidates, key=os.path.getmtime)
    from core.asr_backend.audio_preprocess import convert_video_to_audio

    convert_video_to_audio(source_audio)

    for path in candidates:
        if os.path.abspath(path) != os.path.abspath(raw_audio) and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return raw_audio

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [file for file in video_files if not file.startswith("output/output")]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
