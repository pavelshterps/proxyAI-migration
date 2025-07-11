# crud.py

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models import Upload, User


async def get_user_by_api_key(db: AsyncSession, api_key: str) -> Optional[User]:
    """
    Найти пользователя по его API-ключу.
    Используется в dependencies.get_current_user для аутентификации.
    """
    stmt = select(User).where(User.api_key == api_key)
    res = await db.execute(stmt)
    return res.scalars().first()


async def create_upload_record(
    db: AsyncSession,
    user_id: int,
    upload_id: str,
    external_id: Optional[str] = None,
    callbacks: Optional[List[str]] = None
) -> Upload:
    """
    Создать запись об загрузке.
    """
    rec = Upload(
        user_id=user_id,
        upload_id=upload_id,
        external_id=external_id,
        callback_urls=callbacks or []
    )
    db.add(rec)
    await db.commit()
    return rec


async def get_upload_for_user(
    db: AsyncSession,
    user_id: int,
    upload_id: Optional[str] = None,
    external_id: Optional[str] = None
) -> Optional[Upload]:
    """
    Получить информацию об upload по внутреннему или внешнему ID,
    принадлежавшую конкретному пользователю.
    """
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
    """
    Обновить пользовательскую маппинг-таблицу спикеров для данного upload.
    """
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    if not rec:
        return None
    rec.label_mapping = mapping
    await db.commit()
    return rec


async def get_label_mapping(
    db: AsyncSession,
    user_id: int,
    upload_id: str
) -> dict:
    """
    Получить текущую пользовательскую маппинг-таблицу спикеров (или пустой dict).
    """
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    return rec.label_mapping or {}