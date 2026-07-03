from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

class DocumentChunkResponse(BaseModel):
    """ 
    Public API representation of one persisted document chunk.
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
    created_at: datetime
    
    model_config = ConfigDict(
        from_attributes=True
    )

    
class DocumentChunkListResponse(BaseModel):
        """ 
        Paginated collection of one document's chunks.
        """
        
        document_id: UUID
        total: int
        limit: int
        offset: int
        chunks: list[DocumentChunkResponse]