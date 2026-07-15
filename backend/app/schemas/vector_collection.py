from pydantic import BaseModel

class VectorCollectionStatusResponse(BaseModel):
    """ 
    Public representation of Qdrant collection status.
    """
    
    collection_name: str
    exists: bool
    vector_size: int | None
    distance: str | None
    expected_vector_size: int
    expected_distance: str
    is_compatible: bool
    points_count: int | None
    indexed_vectors_count: int | None
    message: str