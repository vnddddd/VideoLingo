import os
import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'
FINAL_AUDIO_LOUDNORM_FILTER = 'loudnorm=I=-13:TP=-1.5:LRA=11'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def merge_video_audio():
    """Merge video with the generated dub audio.

    When Demucs/vocal separation is enabled, keep the separated background
    track and mix it under the translated dub. When Demucs is disabled, do not
    fall back to raw/original audio: mute the source video completely and map
    only output/dub.mp3 into the final video.
    """
    VIDEO_FILE = find_video_files()
    demucs_enabled = bool(load_key("demucs"))
    background_file = _BACKGROUND_AUDIO_FILE if demucs_enabled else None

    if demucs_enabled:
        if not os.path.exists(background_file):
            raise FileNotFoundError(
                f"{background_file} is required when demucs=true; "
                "run vocal separation first or set demucs: false to render "
                "with only the translated dub audio."
            )
        rprint(
            f"[bold green]demucs=True: mixing separated background "
            f"({background_file}) with translated dub.[/bold green]"
        )
    else:
        rprint(
            "[bold yellow]demucs=False: source/original audio is muted; "
            "rendering final video with translated dub only.[/bold yellow]"
        )

    if not load_key("burn_subtitles"):
        rprint("[bold yellow]burn_subtitles=False: rendering dub video without subtitle overlay (libass skipped).[/bold yellow]")

    # Merge video and audio with translated subtitles.
    # Final loudness normalization is applied to the final audio stream so the
    # exported video, not just the standalone dub track, lands at the target
    # perceived loudness.
    rprint(
        f"[bold green]Final audio loudness normalization: "
        f"{FINAL_AUDIO_LOUDNORM_FILTER}[/bold green]"
    )
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    burn = load_key("burn_subtitles")
    # Build the video filter graph. When burn_subtitles is False we skip the
    # libass overlay — it's a single-threaded CPU bottleneck that starves the
    # GPU encoder. Without it the hardware encoder can run flat-out.
    video_chain = (
        f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )
    if burn:
        subtitle_filter = (
            f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )
        video_chain += f",{subtitle_filter}"
    video_chain += "[v]"

    if demucs_enabled:
        cmd = [
            'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', DUB_AUDIO,
            '-filter_complex',
            f'{video_chain};'
            f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[mixed];'
            f'[mixed]{FINAL_AUDIO_LOUDNORM_FILTER}[a]'
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', DUB_AUDIO,
            '-filter_complex',
            f'{video_chain};'
            f'[1:a]{FINAL_AUDIO_LOUDNORM_FILTER}[a]'
        ]

    # Hardware-accelerated encoder selection (cpu/nvenc/qsv/amf/auto)
    # See core/utils/ffmpeg_utils.py for supported config fields.
    from core.utils.ffmpeg_utils import get_video_encoder_args
    encoder_args, encoder_name = get_video_encoder_args()
    rprint(f"[bold green]Video encoder: {encoder_name}[/bold green]")
    cmd.extend(['-map', '[v]', '-map', '[a]'])
    cmd.extend(encoder_args)

    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])

    # Run ffmpeg and verify it actually succeeded. Previously the return code
    # was ignored, so a silent ffmpeg failure (e.g. missing background.mp3)
    # would still print "successfully merged" and leave the user with no
    # output_dub.mp4 file. Now we fail loudly.
    result = subprocess.run(cmd)
    if result.returncode != 0 or not os.path.exists(DUB_VIDEO):
        raise RuntimeError(
            f"ffmpeg failed to produce {DUB_VIDEO} "
            f"(return code {result.returncode}). "
            f"Check the ffmpeg output above for the underlying error."
        )
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
