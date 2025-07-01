import pytest
from httpx import AsyncClient
from fastapi import status
from pathlib import Path
import json

from main import app
from config.settings import settings

@pytest.fixture
async def client(tmp_path, anyio_backend):
    # point storage to tmp dirs
    settings.UPLOAD_FOLDER = str(tmp_path/"uploads")
    settings.results_folder = str(tmp_path/"results")
    Path(settings.upload_folder).mkdir()
    Path(settings.results_folder).mkdir()
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.mark.anyio
async def test_status_endpoints_work(client, monkeypatch):
    # mock DB user & upload record
    async def fake_get_user_by_api_key(db, key): return type("U",(object,),{"id":1})
    monkeypatch.setattr("crud.get_user_by_api_key", fake_get_user_by_api_key)
    # assume no upload => queued
    res = await client.get("/status/doesnotexist", headers={"X-API-Key":"dummy"})
    assert res.status_code == status.HTTP_404_NOT_FOUND

    # simulate file exists but no results => processing
    Path(settings.UPLOAD_FOLDER + "/u1").write_text("x")
    # also set Redis progress
    await client.app.state.limiter.redis.set("progress:u1","42%")
    res = await client.get("/status/u1", headers={"X-API-Key":"dummy"})
    assert res.json()["status"] == "processing"
    assert res.json()["progress"] == "42%"

    # simulate results complete
    base = Path(settings.RESULTS_FOLDER)/"u1"
    base.mkdir()
    (base/"transcript.json").write_text("[]")
    (base/"diarization.json").write_text("[]")
    res = await client.get("/status/u1", headers={"X-API-Key":"dummy"})
    assert res.json()["status"] == "done"

@pytest.mark.anyio
async def test_labels_validation(client, monkeypatch):
    # mock DB and setup files
    async def fake_user(db, key): return type("U",(object,),{"id":1})
    monkeypatch.setattr("crud.get_user_by_api_key", fake_user)
    # create transcript.json
    base = Path(settings.RESULTS_FOLDER)/"u2"
    base.mkdir()
    segs = [{"start":0,"end":1,"speaker":"sp0","text":"hi"}]
    (base/"transcript.json").write_text(json.dumps(segs))
    (base/"diarization.json").write_text(json.dumps(segs))
    # valid mapping
    res = await client.post(
        "/labels/u2",
        headers={"X-API-Key":"dummy","Content-Type":"application/json"},
        json={"sp0":"Alice"}
    )
    assert res.status_code == 200
    assert json.loads((base/"labels.json").read_text()) == {"sp0":"Alice"}

    # invalid key
    res = await client.post(
        "/labels/u2",
        headers={"X-API-Key":"dummy","Content-Type":"application/json"},
        json={"bad":"X"}
    )
    assert res.status_code == 400