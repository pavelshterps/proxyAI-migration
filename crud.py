from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import User, Upload

async def get_user_by_api_key(db: AsyncSession, api_key: str) -> User | None:
    result = await db.execute(select(User).where(User.api_key == api_key))
    return result.scalars().first()

async def create_upload_record(
    db: AsyncSession,
    user_id: int,
    upload_id: str
) -> Upload:
    upload = Upload(user_id=user_id, upload_id=upload_id)
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    return upload

async def get_upload_for_user(
    db: AsyncSession,
    user_id: int,
    upload_id: str
) -> Upload | None:
    result = await db.execute(
        select(Upload).where(
            Upload.user_id == user_id,
            Upload.upload_id == upload_id
        )
    )
    return result.scalars().first()