# crud.py

import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models import Upload, User

# -----------------------
# Upload-CRUD
# -----------------------

async def create_upload_record(
    db: AsyncSession,
    user_id: int,
    upload_id: str,
    external_id: Optional[str] = None,
    callbacks: Optional[List[str]] = None
) -> Upload:
    rec = Upload(
        user_id=user_id,
        upload_id=upload_id,
        external_id=external_id,
        callback_urls=callbacks or []
    )
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    return rec

async def get_upload_for_user(
    db: AsyncSession,
    user_id: int,
    upload_id: Optional[str] = None,
    external_id: Optional[str] = None
) -> Optional[Upload]:
    stmt = select(Upload).where(Upload.user_id == user_id)
    if upload_id:
        stmt = stmt.where(Upload.upload_id == upload_id)
    if external_id:
        stmt = stmt.where(Upload.external_id == external_id)
    res = await db.execute(stmt)
    return res.scalars().first()

async def update_label_mapping(
    db: AsyncSession,
    user_id: int,
    upload_id: str,
    mapping: dict
) -> Optional[Upload]:
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    if not rec:
        return None
    rec.label_mapping = mapping
    await db.commit()
    await db.refresh(rec)
    return rec

async def get_label_mapping(
    db: AsyncSession,
    user_id: int,
    upload_id: str
) -> dict:
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    return rec.label_mapping or {}

# -----------------------
# User-CRUD (для /admin)
# -----------------------

async def create_user(
    db: AsyncSession,
    name: str,
    api_key: Optional[str] = None
) -> User:
    """
    Создаёт нового пользователя.
    Если api_key не передан, генерирует случайный.
    """
    if api_key is None:
        api_key = uuid.uuid4().hex
    user = User(name=name, api_key=api_key)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

async def get_user_by_api_key(
    db: AsyncSession,
    api_key: str
) -> Optional[User]:
    """
    Ищет пользователя по его api_key (для зависимости get_current_user).
    """
    stmt = select(User).where(User.api_key == api_key)
    res = await db.execute(stmt)
    return res.scalars().first()

async def list_users(
    db: AsyncSession
) -> List[User]:
    """
    Возвращает всех пользователей (для админского списка).
    """
    stmt = select(User)
    res = await db.execute(stmt)
    return res.scalars().all()

async def delete_user(
    db: AsyncSession,
    user_id: int
) -> Optional[User]:
    """
    Удаляет пользователя по его id и возвращает удалённую запись.
    """
    stmt = select(User).where(User.id == user_id)
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        return None
    await db.delete(user)
    await db.commit()
    return user