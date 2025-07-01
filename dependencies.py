from fastapi import HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import APIKeyHeader
from database import get_db
from crud import get_user_by_api_key

api_key_header = APIKeyHeader(name="X-API-Key")

async def get_current_user(
        key: str = Depends(api_key_header),
        db: AsyncSession = Depends(get_db)
):
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    user = await get_user_by_api_key(db, key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return user