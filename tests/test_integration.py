import os
import time
import httpx
import pytest

API_URL = os.getenv("API_URL", "http://api:8000")

def test_transcription_on_sample_wav():
    # 1) Отправляем sample.wav
    with open("tests/fixtures/sample.wav", "rb") as f:
        files = {"file": ("sample.wav", f, "audio/wav")}
        resp = httpx.post(f"{API_URL}/transcribe", files=files, timeout=30.0)
    assert resp.status_code == 200, f"POST /transcribe returned {resp.status_code}: {resp.text}"
    job_id = resp.json().get("job_id")
    assert job_id, "No job_id in response"

    # 2) Ждём окончания (до 60 с)
    for _ in range(30):
        r = httpx.get(f"{API_URL}/result/{job_id}", timeout=10.0)
        data = r.json()
        if data.get("status") == "SUCCESS":
            segments = data.get("segments")
            assert isinstance(segments, list) and segments, "Пустой или неверный формат segments"
            return
        time.sleep(2)
    pytest.fail("Timed out waiting for transcription")