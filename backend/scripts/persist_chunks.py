import argparse
import asyncio
import sys
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.core.exceptions import (
    DocumentProcessingWorkflowError,
)
from app.db.session import (
    AsyncSessionFactory,
    close_database_engine,
)
from app.services.document_chunk_persistence import (
    PersistedChunkingResult,
    generate_and_persist_document_chunks,
)
from app.services.text_chunking import (
    ChunkingOptions,
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


def display_result(
    result: PersistedChunkingResult,
) -> None:
    """
    Display a persisted chunking result.
    """

    chunking_result = (
        result.chunking_result
    )

    print()
    print("EvidenceVault chunk persistence")
    print("=" * 42)

    print(
        f"Document ID: {result.document_id}"
    )

    print(
        f"Status: {result.status.value}"
    )

    print(
        f"Persisted chunks: "
        f"{result.chunk_count}"
    )

    print(
        f"Source pages: "
        f"{chunking_result.source_page_count}"
    )

    print(
        f"Chunked pages: "
        f"{chunking_result.chunked_page_count}"
    )

    print(
        f"Skipped empty pages: "
        f"{chunking_result.skipped_empty_page_numbers}"
    )

    print(
        f"Skipped short pages: "
        f"{chunking_result.skipped_short_page_numbers}"
    )

    print(
        f"Average chunk characters: "
        f"{chunking_result.average_chunk_characters:.1f}"
    )

    first_chunk = (
        chunking_result.chunks[0]
    )

    print()
    print("First persisted chunk")
    print("-" * 42)

    print(
        f"Chunk ID: {first_chunk.chunk_id}"
    )

    print(
        f"Page: {first_chunk.page_number}"
    )

    print(
        f"Characters: "
        f"{first_chunk.character_count}"
    )

    print(
        f"Content hash: "
        f"{first_chunk.content_hash}"
    )


async def persist_chunks(
    document_id: UUID,
    options: ChunkingOptions,
) -> PersistedChunkingResult:
    """
    Persist chunks using a real PostgreSQL session.
    """

    try:
        async with AsyncSessionFactory() as session:
            return await (
                generate_and_persist_document_chunks(
                    session=session,
                    document_id=document_id,
                    options=options,
                )
            )

    finally:
        await close_database_engine()


def parse_arguments() -> argparse.Namespace:
    """
    Read document and chunking arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate and persist page-aware chunks "
            "for one EvidenceVault document."
        )
    )

    parser.add_argument(
        "document_id",
        type=UUID,
        help="PostgreSQL document UUID.",
    )

    parser.add_argument(
        "--max-characters",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--overlap-characters",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--minimum-page-characters",
        type=int,
        default=40,
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    arguments = parse_arguments()

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
            persist_chunks(
                arguments.document_id,
                options,
            )
        )

    except DocumentProcessingWorkflowError as exc:
        raise SystemExit(
            f"Chunk persistence failed: {exc}"
        ) from exc

    except SQLAlchemyError as exc:
        raise SystemExit(
            f"PostgreSQL operation failed: {exc}"
        ) from exc

    display_result(
        result
    )


if __name__ == "__main__":
    main()