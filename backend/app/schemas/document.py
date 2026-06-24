from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

class DocumentResponse(BaseModel):
    """
    Public API representation of an uploaded document.
    
    This schema controls which document fields are returned to API clients.
    """
    
    id: UUID
    filename: str
    original_filename: str
    content_type: str
    storage_path:str | None
    status:str
    page_count: int | None
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(
        from_attributes=True
    )