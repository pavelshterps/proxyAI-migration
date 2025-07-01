import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from config.settings import settings

client = TestClient(app)

@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory):
    src = Path(__file__).parent / "fixtures" / "sample.wav"
    dest_dir = tmp_path_factory.mktemp("uploads")
    dest = dest_dir / "sample.wav"
    dest.write_bytes(src.read_bytes())
    # override upload_folder
    settings.UPLOAD_FOLDER = str(dest_dir)
    settings.RESULTS_FOLDER = str(tmp_path_factory.mktemp("results"))
    return "sample.wav"

def test_upload_and_process(sample_wav):
    # Upload
    r1 = client.post("/upload/", files={"file": ("sample.wav", open(Path(settings.UPLOAD_FOLDER) / sample_wav, "rb"), "audio/wav")})
    assert r1.status_code == 200, r1.text
    upload_id = r1.json()["upload_id"]

    # Give Celery some time (in real CI you might mock Celery or run tasks eagerly)
    time.sleep(5)

    # Fetch results
    r2 = client.get(f"/results/{upload_id}")
    assert r2.status_code == 200, r2.text
    data = r2.json()
    # Check that at least transcript or diarization is present
    assert data["transcript"] is not None, "Transcript should not be empty"
    assert data["diarization"] is not None, "Diarization should not be empty"

    # Validate JSON structure
    transcript = json.loads(data["transcript"])
    diar = json.loads(data["diarization"])
    assert isinstance(transcript, list)
    assert isinstance(diar, list)