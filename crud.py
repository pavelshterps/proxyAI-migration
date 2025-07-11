# crud.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from models import Upload

async def create_upload_record(
    db: AsyncSession,
    user_id: int,
    upload_id: str,
    external_id: Optional[str] = None,
    callbacks: Optional[List[str]] = None
):
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
):
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
):
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
    rec = await get_upload_for_user(db, user_id, upload_id=upload_id)
    return rec.label_mapping or {}