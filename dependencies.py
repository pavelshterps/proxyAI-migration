# dependencies.py
from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from asyncpg.exceptions import CannotConnectNowError
from sqlalchemy.exc import InterfaceError

from database import get_db
from crud import get_user_by_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(
    key: str = Depends(api_key_header),
    db: AsyncSession = Depends(get_db)
):
    try:
        if not key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")
        user = await get_user_by_api_key(db, key)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid X-API-Key")
        return user
    except (CannotConnectNowError, InterfaceError):
        # когда БД ещё не приняла соединения
        raise HTTPException(status_code=503, detail="Database not ready")