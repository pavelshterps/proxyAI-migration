# main.py
import structlog
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from config.settings import settings

logger = structlog.get_logger()
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

if settings.enable_metrics:
    @app.get("/metrics")
    async def metrics():
        data = generate_latest()
        return Response(data, media_type=CONTENT_TYPE_LATEST)

# ... existing tusd-upload callbacks & endpoints ...