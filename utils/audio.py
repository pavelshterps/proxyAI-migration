# utils/audio.py

import subprocess
from pathlib import Path
import logging
import requests

logger = logging.getLogger(__name__)

def download_url_to_path(url: str, dst_path: Path) -> Path:
    """
    Скачать файл по URL в локальный путь.

    Args:
        url (str): HTTP(S) ссылка на файл.
        dst_path (Path): Локальный путь (директория или файл).

    Returns:
        Path: Путь к сохранённому файлу.
    """
    dst = Path(dst_path)
    if dst.is_dir():
        dst = dst / Path(url).name
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url} → {dst}")
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(dst, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
    return dst

def convert_to_wav(src_path, dst_path=None) -> Path:
    src = Path(src_path)
    if dst_path:
        dst = Path(dst_path)
    else:
        if src.suffix.lower() == ".wav":
            return src
        dst = src.with_suffix(".wav")

    try:
        if src.resolve() == dst.resolve():
            return dst
    except Exception:
        pass

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        str(dst)
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"convert_to_wav failed for {src} → {dst}: {e}")
        raise

    return dst