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
    
class AnswerToken(BaseModel):
    """ 
    One streamed fragment of the generated answer.

    Emitted repeatedly as the LLM produces text, in order.
    Concatenating every AnswerToken.text in a stream, in the
    order received, reproduces the same answer that
    AnswerResponse.answer would contain.
    """
    
    text: str
    
class AnswerComplete(BaseModel):
    """ 
    Final event of a streamed answer.

    Carries the metadata that is only known once generation
    has finished — total sources used, whether the answer
    was grounded, and how long the whole request took. A
    stream consumer should treat this as the signal that no
    more AnswerToken events will follow.
    """
    
    document_id: UUID
    
    query: str
    
    model: str
    
    grounded: bool
    
    total_sources: int
    
    retrieved_chunks: tuple[RetrievedChunk, ...]
    
    generation_time_ms: float = Field(
        ge=0
    )