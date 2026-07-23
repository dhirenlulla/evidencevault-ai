from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievedChunk


class AnswerRequest(BaseModel):
    """ 
    User question submitted for grounded answering.
    """
    
    query: str = Field(
        min_length=1,
        max_length=4000,
    )
    
    
class AnswerResponse(BaseModel):
    """ 
    Grounded answer returned by the RAG pipeline.
    """
    
    document_id: UUID
    
    query: str
    
    answer: str
    
    model : str
    
    
    grounded: bool
    
    total_sources: int
    
    retrieved_chunks: tuple[
        RetrievedChunk,
        ...
    ]
    
    generation_time_ms: float = Field(
        ge=0,
    )