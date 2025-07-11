import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Boolean
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True, index=True)
    name     = Column(String, nullable=False)
    api_key  = Column(String, unique=True, index=True)
    uploads  = relationship("Upload", back_populates="user")

class Upload(Base):
    __tablename__ = "uploads"
    id                    = Column(Integer, primary_key=True, index=True)
    user_id               = Column(Integer, ForeignKey("users.id"), nullable=False)
    upload_id             = Column(String, unique=True, index=True)
    external_id           = Column(String, unique=True, index=True, nullable=True)
    status                = Column(String, default="queued", index=True)
    preview_status        = Column(String, default="queued", index=True)
    preview_result        = Column(JSON, nullable=True)
    callback_urls         = Column(JSON, nullable=True)
    diarization_requested = Column(Boolean, default=False, index=True)
    # (опционально) можно хранить статус/результат диаризации в БД, но мы используем Redis:
    # diarization_status   = Column(String, default="queued", index=True)
    # diarization_result   = Column(JSON, nullable=True)
    created_at            = Column(
                              DateTime,
                              default=datetime.datetime.utcnow
                            )
    updated_at            = Column(
                              DateTime,
                              default=datetime.datetime.utcnow,
                              onupdate=datetime.datetime.utcnow
                            )
    user                  = relationship("User", back_populates="uploads")