# schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    api_key: str

class UploadCreate(BaseModel):
    filename: str
    status: Optional[str] = "pending"
    created_at: Optional[datetime] = None