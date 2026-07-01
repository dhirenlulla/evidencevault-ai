from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk

@dataclass(frozen=True, slots=True)
class DocumentChunkInput:
    """ 
    Data required to persist one generated text chunk.
    """
    
    id: UUID
    document_id: UUID
    chunk_index: int
    page_number: int
    page_chunk_index: int
    text: str
    character_count: int
    word_count: int
    content_hash: str
    
async def replace_document_chunks(
    session: AsyncSession, 
    *,
    document: Document,
    chunks: Sequence[DocumentChunkInput],
    final_status: str = "chunked",
) -> list[DocumentChunk]:
    """ 
    Replace all persisted chunks for one document atomically.
    
    Previous chunks are deleted, new chunks are inserted, and the parent
    document's status and count are updated in the same database
    transaction.
    """
    
    chunk_records = [
        DocumentChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            page_chunk_index=(chunk.page_chunk_index),
            text=chunk.text,
            character_count=(chunk.character_count),
            word_count=chunk.word_count,
            content_hash=chunk.content_hash,
        )
        for chunk in chunks
    ]
    
    try:
        await session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document.id
            )
        )
        
        session.add_all(chunk_records)
        
        document.status = final_status
        document.chunk_count = len(chunk_records)
        document.error_message = None
        
        await session.commit()
        await session.refresh(document)
        
    except SQLAlchemyError:
        await session.rollback()
        raise
    
    return chunk_records

async def list_document_chunks(
    session: AsyncSession,
    *,
    document_id: UUID
) -> list[DocumentChunk]:
    """ 
    Return one document's chunks in their original order.
    """
    
    statement = (
        select(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        ).order_by(
            DocumentChunk.chunk_index.asc()
        )
    )
    
    result = await session.execute(statement)
    
    return list(result.scalars().all())