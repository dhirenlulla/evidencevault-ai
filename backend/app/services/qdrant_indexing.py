from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from qdrant_client import models
from qdrant_client.async_qdrant_client import (
    AsyncQdrantClient
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentNotReadyForIndexingError,
    NoChunksAvailableForIndexingError,
    QdrantPointUpsertError,
    VectorIndexingError,
)

from app.db.models.document_chunk import DocumentChunk
from app.repositories.document import get_document_by_id
from app.repositories.document_chunk import (
    count_document_chunks,
    list_document_chunks
)
from app.services.embedding import (
    EmbeddingService
)
from app.services.qdrant_collection import (
    QdrantCollectionService,
)


INDEXABLE_DOCUMENT_STATUSES = {
    "chunked",
    "indexed"
}

@dataclass(frozen=True, slots=True)
class VectorIndexingOptions:
    """ 
    Configuration for chunk-to-vector indexing.
    """
    
    batch_size: int = 16
    
    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise VectorIndexingError(
                "Indexing batch size must be at least 1."
            )
            

@dataclass(frozen=True, slots=True)
class IndexedBatchResult:
    """ 
    Result for one indexed batch.
    """
    
    batch_number: int
    chunk_count: int
    point_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentVectorIndexingResult:
    """ 
    Final result after indexing one document's chunks.
    """
    
    document_id: UUID
    collection_name:str
    total_chunks: int
    indexed_chunks: int
    batch_count: int
    vector_size: int
    point_ids: tuple[str, ...]
    status: str = "indexed"
    
    @property
    def is_complete(self) -> bool:
        """ 
        Return whether all available chunks were indexed.
        """
        
        return self.indexed_chunks == self.total_chunks
    
    
def _build_chunk_payload(chunk: DocumentChunk) -> dict:
    """ 
    Build Qdrant payload metadata for one chunk.

    PostgreSQL remains the source of truth. This payload is a
    denormalized retrieval index copy that makes search results
    immediately useful.
    """
    
    return {
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "page_chunk_index": chunk.page_chunk_index,
        "character_count": chunk.character_count,
        "word_count": chunk.word_count,
        "content_hash": chunk.content_hash,
        "text": chunk.text,
        "source": "postgresql_document_chunks",
    }
    

def _build_qdrant_points(*, chunks: Sequence[DocumentChunk], vectors) -> list[models.PointStruct]:
    """ 
    Convert document chunks and embeddings into Qdrant points.
    """
    
    if len(chunks) != len(vectors):
        raise VectorIndexingError(
            "Chunk count and embedding count do not match."
        )
        
    points: list[models.PointStruct] = []
    
    for chunk, vector in zip(
        chunks,
        vectors,
        strict=True
    ):
        points.append(
            models.PointStruct(
                id=str(chunk.id),
                vector=vector.tolist(),
                payload=_build_chunk_payload(chunk),
            )
        )
        
    return points
    
    
async def _upsert_points(
    *, 
    client: AsyncQdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct]
) -> None:
    """ 
    Upsert one batch of points into Qdrant.
    """
    
    try:
        await client.upsert(
            collection_name=collection_name,
            points=list(points),
            wait=True
        )
        
    except Exception as exc:
        raise QdrantPointUpsertError(
            "Could not upsert chunk vectors into "
            f"Qdrant collection: {collection_name}"
        ) from exc
        
        
async def index_document_chunks(
    *, 
    session: AsyncSession,
    document_id: UUID,
    qdrant_client: AsyncQdrantClient,
    collection_service: QdrantCollectionService,
    embedding_service: EmbeddingService,
    options: VectorIndexingOptions | None = None   
) -> DocumentVectorIndexingResult:
    """ 
    Index one document's persisted chunks into Qdrant.
    
    PostgreSQL provides the durable chunk text. The embedding
    service converts each chunk into a dense vector. Qdrant stores 
    the vector plus useful retrieval payload metadata.
    """
    
    indexing_options = (
        options
        if options is not None
        else VectorIndexingOptions()
    )
    
    document = await get_document_by_id(
        session=session,
        document_id=document_id
    )
    
    if document is None:
        raise DocumentNotFoundError(
            "Document not found."
        )
        
    if document.status not in INDEXABLE_DOCUMENT_STATUSES:
        raise DocumentNotReadyForIndexingError(
            "Document must have status 'chunked' "
            "before vector indexing can run. "
            f"Current status: {document.status}."
        )
        
    total_chunks = await count_document_chunks(
        session = session,
        document_id=document_id,
    )
    
    if total_chunks == 0:
        raise NoChunksAvailableForIndexingError(
            "The document has no persisted chunks to index."
        )
        
    collection_status = await(
        collection_service.ensure_collection()
    )
    
    offset = 0
    indexed_chunks = 0
    batch_number = 0
    
    indexed_batches: list[IndexedBatchResult] = []
    
    while offset < total_chunks:
        chunks = await list_document_chunks(
            session=session,
            document_id=document_id,
            limit=indexing_options.batch_size,
            offset=offset
        )
        
        if not chunks:
            break
        
        texts = [
            chunk.text
            for chunk in chunks
        ]
        
        embedding_result = await(
            embedding_service.embed_documents_async(texts)
        )
        
        points = _build_qdrant_points(
            chunks=chunks,
            vectors=embedding_result.vectors
        )
        
        await _upsert_points(
            client=qdrant_client,
            collection_name=(
                collection_service.collection_name
            ),
            points=points,
        )

        point_ids = tuple(
            str(chunk.id)
            for chunk in chunks
        )

        indexed_batches.append(
            IndexedBatchResult(
                batch_number=batch_number,
                chunk_count=len(chunks),
                point_ids=point_ids,
            )
        )

        indexed_chunks += len(chunks)
        offset += len(chunks)
        batch_number += 1

    if indexed_chunks != total_chunks:
        raise VectorIndexingError(
            "Indexing ended before all chunks were indexed. "
            f"Expected {total_chunks}, indexed {indexed_chunks}."
        )

    all_point_ids = tuple(
        point_id
        for batch in indexed_batches
        for point_id in batch.point_ids
    )

    return DocumentVectorIndexingResult(
        document_id=document_id,
        collection_name=(
            collection_service.collection_name
        ),
        total_chunks=total_chunks,
        indexed_chunks=indexed_chunks,
        batch_count=len(indexed_batches),
        vector_size=(
            collection_status.expected_vector_size
        ),
        point_ids=all_point_ids,
    )