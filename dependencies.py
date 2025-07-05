from fastapi import Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from database import get_db
from crud import get_user_by_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(
    api_key: str = Depends(api_key_header),
    db = Depends(get_db),
):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    user = await get_user_by_api_key(db, api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user