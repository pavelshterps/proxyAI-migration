import logging
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from crud import get_user_by_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
logger = logging.getLogger(__name__)


async def get_current_user(
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db),
):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    try:
        user = await get_user_by_api_key(db, api_key)
    except Exception as e:
        logger.error("Database error when fetching user by API key", error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user