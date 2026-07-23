from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """ 
    Request body for semantic retrieval.
    """
    
    query: str = Field(
        min_length=1,
        max_length=4000,
    )
    
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
    )