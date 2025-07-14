import subprocess
from pathlib import Path
import logging
import requests

logger = logging.getLogger(__name__)

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
        "ffmpeg",
        "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(dst)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"convert_to_wav: ffmpeg failed for {src} → {dst}: {e}")
        raise
    return dst

def download_audio(url: str, folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    ext = Path(url.split("?")[0]).suffix or ".dat"
    out_path = folder / f"{filename}{ext}"
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        logger.error(f"download_audio: failed to download {url}: {e}")
        raise
    return out_path