"""Player-friendly export names; pipeline/cache filenames stay stable."""

import shutil
from pathlib import Path


def output_filename(path, video_file=None):
    """Use the source video's full stem and keep the actual file extension."""
    path = Path(path)
    variants = {"dub": "", "dub_loudnorm": "", "trans": ".trans"}
    if path.stem not in variants:
        return path.name
    if video_file is None:
        from core._1_ytdlp import find_video_files

        try:
            video_file = find_video_files()
        except ValueError:
            # Audio-only browser jobs do not have a source video filename.
            return path.name
    return f"{Path(video_file).stem}.zh-CN{variants[path.stem]}{path.suffix}"


def export_output_file(path, video_file=None, destination_dir=None):
    """Copy an existing result to its export name without moving pipeline data."""
    source = Path(path)
    if not source.is_file():
        return None
    directory = Path(destination_dir) if destination_dir is not None else source.parent
    destination = directory / output_filename(source, video_file)
    if source.resolve() != destination.resolve():
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination


def export_dub_outputs(output_dir="output", video_file=None, destination_dir=None):
    """Export the final audio (normalized when available) and its DUB subtitle."""
    output_dir = Path(output_dir)
    audio = output_dir / "dub_loudnorm.mp3"
    if not audio.is_file():
        audio = output_dir / "dub.mp3"
    exported = []
    for source in (audio, output_dir / "dub.srt"):
        destination = export_output_file(source, video_file, destination_dir)
        if destination is not None:
            exported.append(destination)
    return exported
