from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document

async def create_document(
    session: AsyncSession,
    *,
    document_id: UUID,
    filename: str,
    original_filename:str,
    content_type:str,
    storage_path: str,
    status: str = "uploaded",
) -> Document:
    """ 
    Create and persist one document  metadata record.
    
    The file itself has already been stored before this function is called.
    This repository function records the file's metadata in PostgreSQL.
    """
    
    document = Document(
        id=document_id,
        filename=filename,
        original_filename=original_filename,
        content_type=content_type,
        storage_path=storage_path,
        status=status
    )
    
    session.add(document)
    
    try:
        await session.commit()
        await session.refresh(document)
        
    except SQLAlchemyError:
        await session.rollback()
        raise
    
    return document


async def get_document_by_id(
    session: AsyncSession,
    document_id: UUID,
) -> Document | None:
    """
    Retrieve one document using its primary-key UUID.
    
    Returns None when no document exists with the supplied UUID.
    """
    
    return await session.get(
        Document,
        document_id,
    )
    
async def list_documents(
    session: AsyncSession,
    *,
    limit: int = 20,
    offset: int = 0
) -> list[Document]:
    """
    Retrieve documents from newest to oldest
    
    'limit' controls the maximum number returned.
    'offset' controls how many earlier records are skipped.
    """
    
    statement = (
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    
    result = await session.execute(statement)
    
    return list(
        result.scalars().all()
    )