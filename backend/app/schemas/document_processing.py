from uuid import UUID
from pydantic import BaseModel

class DocumentProcessingResponse(BaseModel):
    """ 
    Final result returned after processing one document.
    """
    
    document_id: UUID
    status: str
    classification: str | None
    page_count: int
    extractable_page_count: int
    image_only_page_count: int
    empty_page_count: int
    total_characters: int
    total_words: int
    chunk_count: int
    ready_for_indexing: bool
    message: str
    
