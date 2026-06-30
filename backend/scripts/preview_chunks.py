import argparse
import asyncio
import sys
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.db.session import (
    AsyncSessionFactory,
    close_database_engine,
)
from app.repositories.document import (
    get_document_by_id,
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


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


ALLOWED_DOCUMENT_STATUSES = {
    DocumentProcessingStatus.EXTRACTED.value,
    (
        DocumentProcessingStatus
        .EXTRACTED_WITH_WARNINGS.value
    ),
}


def create_preview(
    text: str,
    limit: int = 300,
) -> str:
    """
    Create a compact single-line terminal preview.
    """

    normalized_text = " ".join(
        text.split()
    )

    if len(normalized_text) <= limit:
        return normalized_text

    return (
        normalized_text[:limit].rstrip()
        + "..."
    )


def display_chunking_result(
    result: ChunkingResult,
    *,
    preview_limit: int,
) -> None:
    """
    Display chunking statistics and sample chunks.
    """

    print()
    print("EvidenceVault chunk preview")
    print("=" * 42)

    print(
        f"Document ID: {result.document_id}"
    )

    print(
        f"Source pages: "
        f"{result.source_page_count}"
    )

    print(
        f"Source characters: "
        f"{result.source_character_count}"
    )

    print(
        f"Generated chunks: "
        f"{result.chunk_count}"
    )

    print(
        f"Chunked pages: "
        f"{result.chunked_page_count}"
    )

    print(
        f"Total chunk characters: "
        f"{result.total_chunk_characters}"
    )

    print(
        f"Average chunk characters: "
        f"{result.average_chunk_characters:.1f}"
    )

    print(
        f"Maximum chunk characters: "
        f"{result.options.max_characters}"
    )

    print(
        f"Overlap characters: "
        f"{result.options.overlap_characters}"
    )

    print(
        f"Minimum page characters: "
        f"{result.options.minimum_page_characters}"
    )

    if result.skipped_empty_page_numbers:
        print(
            "Skipped empty pages: "
            + ", ".join(
                str(page_number)
                for page_number
                in result
                .skipped_empty_page_numbers
            )
        )

    if result.skipped_short_page_numbers:
        print(
            "Skipped short pages: "
            + ", ".join(
                str(page_number)
                for page_number
                in result
                .skipped_short_page_numbers
            )
        )

    print()
    print("Chunk samples")
    print("-" * 42)

    chunks_to_display = (
        result.chunks[:preview_limit]
    )

    for chunk in chunks_to_display:
        print()
        print(
            f"Chunk {chunk.chunk_index}"
        )

        print(
            f"  ID: {chunk.chunk_id}"
        )

        print(
            f"  Page: {chunk.page_number}"
        )

        print(
            "  Page chunk index: "
            f"{chunk.page_chunk_index}"
        )

        print(
            f"  Characters: "
            f"{chunk.character_count}"
        )

        print(
            f"  Words: {chunk.word_count}"
        )

        print(
            f"  Hash: {chunk.content_hash[:16]}..."
        )

        print(
            f"  Preview: "
            f"{create_preview(chunk.text)}"
        )


async def generate_real_document_chunks(
    *,
    document_id: UUID,
    options: ChunkingOptions,
) -> ChunkingResult:
    """
    Retrieve, extract, and chunk one real document.
    """

    try:
        async with AsyncSessionFactory() as session:
            document = await get_document_by_id(
                session=session,
                document_id=document_id,
            )

            if document is None:
                raise SystemExit(
                    f"Document not found: "
                    f"{document_id}"
                )

            if (
                document.status
                not in ALLOWED_DOCUMENT_STATUSES
            ):
                raise SystemExit(
                    "The document must have status "
                    "'extracted' or "
                    "'extracted_with_warnings' "
                    "before chunking. "
                    f"Current status: "
                    f"{document.status}"
                )

            pdf_path = (
                resolve_document_storage_path(
                    document.storage_path
                )
            )

            extracted_document = (
                extract_pdf_pages(
                    pdf_path
                )
            )

            return chunk_extracted_document(
                document_id=document_id,
                extracted_document=(
                    extracted_document
                ),
                options=options,
            )

    finally:
        await close_database_engine()


def parse_arguments() -> argparse.Namespace:
    """
    Read document and chunking options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate and preview citation-ready "
            "chunks for an EvidenceVault document."
        )
    )

    parser.add_argument(
        "document_id",
        type=UUID,
        help="PostgreSQL document UUID.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Maximum number of chunk previews "
            "to display."
        ),
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=1200,
        help="Maximum characters per chunk.",
    )

    parser.add_argument(
        "--overlap-characters",
        type=int,
        default=200,
        help=(
            "Maximum overlap between neighboring "
            "chunks."
        ),
    )

    parser.add_argument(
        "--minimum-page-characters",
        type=int,
        default=40,
        help=(
            "Minimum page text required before "
            "chunking."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    arguments = parse_arguments()

    if arguments.limit < 1:
        raise SystemExit(
            "--limit must be at least 1."
        )

    options = ChunkingOptions(
        max_characters=(
            arguments.max_characters
        ),
        overlap_characters=(
            arguments.overlap_characters
        ),
        minimum_page_characters=(
            arguments.minimum_page_characters
        ),
    )

    try:
        result = asyncio.run(
            generate_real_document_chunks(
                document_id=(
                    arguments.document_id
                ),
                options=options,
            )
        )

    except SQLAlchemyError as exc:
        raise SystemExit(
            f"PostgreSQL operation failed: {exc}"
        ) from exc

    display_chunking_result(
        result,
        preview_limit=arguments.limit,
    )


if __name__ == "__main__":
    main()