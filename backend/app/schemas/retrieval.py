from uuid import UUID

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """ 
    One chunk returned from semantic search.
    """
    
    chunk_id: UUID
    
    document_id: UUID
    
    chunk_index: int
    
    page_number: int
    
    page_chunk_index: int
    
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    
    text: str
    
    character_count: int
    
    word_count: int
    
    
class RetrievalResult(BaseModel):
    """ 
    Complete semantic search response.
    """
    
    query: str
    
    total_results: int
    
    chunks: tuple[RetrievedChunk, ...]
    