# config/crud.py

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Upload, User  # adjust imports as needed

async def get_user_by_api_key(db: AsyncSession, api_key: str) -> User:
    ...

async def create_user(db: AsyncSession, user: UserCreate) -> User:
    ...

# existing upload helpers
async def create_upload_record(db: AsyncSession, user_id: str, filename: str, **kwargs) -> Upload:
    ...

async def get_upload_for_user(db: AsyncSession, user_id: str, upload_id: str) -> Upload:
    ...

# NEW: update status field of an existing upload
async def update_upload_status(
    db: AsyncSession,
    upload_id: str,
    status: str
) -> Upload:
    """
    Update the `status` of an upload (e.g. 'processing', 'completed', 'failed').
    """
    result = await db.execute(
        select(Upload).where(Upload.id == upload_id)
    )
    upload = result.scalar_one()
    upload.status = status
    db.add(upload)
    await db.commit()  # commit the change  [oai_citation:0‡huggingface.co](https://huggingface.co/docs/huggingface_hub/en/guides/download?utm_source=chatgpt.com)
    await db.refresh(upload)
    return upload