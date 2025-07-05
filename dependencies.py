from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from crud import get_user_by_api_key
from database import get_db
from models import User

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(db=Depends(get_db), api_key: str = Depends(api_key_header)) -> User:
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-API-Key")
    user = await get_user_by_api_key(db, api_key)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-API-Key")
    return user