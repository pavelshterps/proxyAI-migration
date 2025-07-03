from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from database import get_db
from crud import create_user, delete_user, list_users
from config.settings import settings

router = APIRouter(prefix="/admin", tags=["admin"])

# Use FastAPI's APIKeyHeader for admin key
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

async def require_admin(key: str = Depends(admin_key_header)):
    if not key or key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Admin-Key"
        )
    return key

@router.post("/users", dependencies=[Depends(require_admin)])
async def admin_create_user(
    name: str,
    db: AsyncSession = Depends(get_db)
):
    key = uuid.uuid4().hex
    user = await create_user(db, name, key)
    return {"id": user.id, "name": user.name, "api_key": user.api_key}

@router.get("/users", dependencies=[Depends(require_admin)])
async def admin_list_users(db: AsyncSession = Depends(get_db)):
    users = await list_users(db)
    return [{"id": u.id, "name": u.name, "api_key": u.api_key} for u in users]

@router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    ok = await delete_user(db, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "user_id": user_id}