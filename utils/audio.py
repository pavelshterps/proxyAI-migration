import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse
import requests

logger = logging.getLogger(__name__)

def download_audio(url: str, dst: Path = None, timeout: int = 30) -> Path:
    """
    Download an audio file from a URL to local disk.
    Supports HTTP/HTTPS. Returns local file path.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    if dst is None:
        fname = Path(parsed.path).name or "downloaded_audio"
        dst = Path.cwd() / fname
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading audio from {url} to {dst}")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dst

def convert_to_wav(src_path, dst_path=None) -> Path:
    """
    Convert an audio file to WAV (mono, 16 kHz). If already WAV, reuse.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Source audio not found: {src}")

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
    logger.info(f"Converting to WAV: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        raise
    return dst