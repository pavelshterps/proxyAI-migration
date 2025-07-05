# utils/audio.py
import subprocess
from pathlib import Path

def convert_to_wav(src_path: Path, dst_path: Path, sample_rate: int = 16000, channels: int = 1):
    """
    Конвертирует любой аудио/видео-файл → моно PCM WAV sample_rate Гц.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vn", "-ac", str(channels), "-ar", str(sample_rate),
        "-f", "wav", str(dst_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)