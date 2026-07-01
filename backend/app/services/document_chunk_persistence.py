from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentNotReadyForChunkingError,
    NoChunksGeneratedError,
)
from app.repositories.document import (
    get_document_by_id,
)
from app.repositories.document_chunk import (
    DocumentChunkInput,
    replace_document_chunks,
)
from app.services.document_processing import (
    DocumentProcessingStatus,
    resolve_document_storage_path,
)
from app.services.pdf_extraction import (
    extract_pdf_pages,
)
from app.services.text_chunking import (
    ChunkingOptions,
    ChunkingResult,
    chunk_extracted_document,
)


ALLOWED_SOURCE_STATUSES = {
    DocumentProcessingStatus.EXTRACTED.value,
    (
        DocumentProcessingStatus
        .EXTRACTED_WITH_WARNINGS.value
    ),
    DocumentProcessingStatus.CHUNKED.value,
}


@dataclass(
    frozen=True,
    slots=True,
)
class PersistedChunkingResult:
    """
    Result of generating and persisting one document's chunks.
    """

    document_id: UUID
    status: DocumentProcessingStatus
    chunking_result: ChunkingResult

    @property
    def chunk_count(self) -> int:
        """
        Return the number of persisted chunks.
        """

        return self.chunking_result.chunk_count


def build_chunk_inputs(
    chunking_result: ChunkingResult,
) -> tuple[DocumentChunkInput, ...]:
    """
    Convert generated TextChunk objects into repository inputs.
    """

    return tuple(
        DocumentChunkInput(
            id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            page_chunk_index=(
                chunk.page_chunk_index
            ),
            text=chunk.text,
            character_count=(
                chunk.character_count
            ),
            word_count=chunk.word_count,
            content_hash=chunk.content_hash,
        )
        for chunk in chunking_result.chunks
    )


async def generate_and_persist_document_chunks(
    session: AsyncSession,
    *,
    document_id: UUID,
    options: ChunkingOptions | None = None,
) -> PersistedChunkingResult:
    """
    Extract, chunk and persist one processed document.
    """

    document = await get_document_by_id(
        session=session,
        document_id=document_id,
    )

    if document is None:
        raise DocumentNotFoundError(
            f"Document not found: {document_id}"
        )

    if document.status not in ALLOWED_SOURCE_STATUSES:
        raise DocumentNotReadyForChunkingError(
            "The document must have status "
            "'extracted', 'extracted_with_warnings', "
            "or 'chunked' before chunks can be persisted. "
            f"Current status: {document.status}"
        )

    pdf_path = resolve_document_storage_path(
        document.storage_path
    )

    extracted_document = extract_pdf_pages(
        pdf_path
    )

    chunking_result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=extracted_document,
        options=options,
    )

    if chunking_result.chunk_count == 0:
        raise NoChunksGeneratedError(
            "The document did not produce any usable "
            "text chunks."
        )

    chunk_inputs = build_chunk_inputs(
        chunking_result
    )

    await replace_document_chunks(
        session=session,
        document=document,
        chunks=chunk_inputs,
        final_status=(
            DocumentProcessingStatus.CHUNKED.value
        ),
    )

    return PersistedChunkingResult(
        document_id=document_id,
        status=DocumentProcessingStatus.CHUNKED,
        chunking_result=chunking_result,
    )