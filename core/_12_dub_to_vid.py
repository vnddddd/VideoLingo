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
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    # Fallback: when demucs is disabled (remote ASR/TTS users), background.mp3
    # is never generated. Fall back to the raw mixed audio so the ffmpeg amix
    # stage still has a valid second input. amix naturally attenuates each
    # input by ~50%, so the original audio sits softly under the new dub.
    if not os.path.exists(background_file):
        if os.path.exists(_RAW_AUDIO_FILE):
            rprint(
                f"[yellow][WARN] {background_file} not found "
                f"(demucs disabled). Falling back to {_RAW_AUDIO_FILE} "
                f"as the background track.[/yellow]"
            )
            background_file = _RAW_AUDIO_FILE
        else:
            raise FileNotFoundError(
                f"Neither {background_file} nor {_RAW_AUDIO_FILE} exists; "
                f"cannot merge dub video without a background audio source."
            )

    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Merge video and audio with translated subtitles.
    # Final loudness normalization is applied after amix so the exported video,
    # not just the standalone dub track, lands at the target perceived loudness.
    # This avoids the old dBFS-only normalization being attenuated again by amix.
    rprint(
        f"[bold green]Final audio loudness normalization: "
        f"{FINAL_AUDIO_LOUDNORM_FILTER}[/bold green]"
    )
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    subtitle_filter = (
        f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
        f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
        f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
    )
    
    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', DUB_AUDIO,
        '-filter_complex',
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
        f'{subtitle_filter}[v];'
        f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[mixed];'
        f'[mixed]{FINAL_AUDIO_LOUDNORM_FILTER}[a]'
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
