# utils/audio.py

import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def convert_to_wav(src_path, dst_path=None) -> Path:
    """
    Convert an audio file to WAV format with mono channel and 16 kHz sample rate.

    Args:
        src_path (str or Path): Path to the source audio file.
        dst_path (str or Path, optional): Path where the WAV file will be written.
            If not provided, the destination will be the same as src_path with a `.wav` suffix.

    Returns:
        Path: Path to the resulting WAV file (which may be the original if already WAV).
    """
    src = Path(src_path)
    # Determine destination path
    if dst_path:
        dst = Path(dst_path)
    else:
        # If source already .wav, reuse it
        if src.suffix.lower() == ".wav":
            return src
        dst = src.with_suffix(".wav")

    # If source and destination are the same file, skip conversion
    try:
        if src.resolve() == dst.resolve():
            return dst
    except Exception:
        # In case resolve fails (e.g., file doesn't exist yet), proceed to conversion
        pass

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",           # no video
        "-ac", "1",      # mono audio
        "-ar", "16000",  # 16 kHz sample rate
        "-f", "wav",
        str(dst)
    ]

    # Run conversion
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"convert_to_wav: ffmpeg failed for {src} → {dst}: {e}")
        raise

    return dst