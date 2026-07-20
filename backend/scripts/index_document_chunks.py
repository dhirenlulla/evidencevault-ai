import argparse
import asyncio
import sys
from uuid import UUID

from app.clients.qdrant import (
    close_qdrant_client,
    get_qdrant_client,
)
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentNotReadyForIndexingError,
    NoChunksAvailableForIndexingError,
    QdrantCollectionError,
    VectorIndexingError,
)
from app.db.session import get_db_session
from app.services.embedding import (
    get_embedding_service,
)
from app.services.qdrant_collection import (
    QdrantCollectionService,
    build_vector_collection_config,
)
from app.services.qdrant_indexing import (
    VectorIndexingOptions,
    index_document_chunks,
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Index one document's persisted chunks "
            "into Qdrant."
        )
    )

    parser.add_argument(
        "document_id",
        type=UUID,
        help="PostgreSQL document UUID to index.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of chunks to embed and upsert "
            "per batch."
        ),
    )

    return parser.parse_args()


async def run(
    *,
    document_id: UUID,
    batch_size: int,
) -> None:
    """
    Run chunk indexing for one document.
    """

    qdrant_client = get_qdrant_client()

    collection_config = (
        build_vector_collection_config()
    )

    collection_service = QdrantCollectionService(
        client=qdrant_client,
        config=collection_config,
    )

    embedding_service = get_embedding_service()

    try:
        async for session in get_db_session():
            result = await index_document_chunks(
                session=session,
                document_id=document_id,
                qdrant_client=qdrant_client,
                collection_service=collection_service,
                embedding_service=embedding_service,
                options=VectorIndexingOptions(
                    batch_size=batch_size
                ),
            )

            print()
            print("EvidenceVault vector indexing")
            print("=" * 50)
            print(f"Document ID: {result.document_id}")
            print(f"Collection: {result.collection_name}")
            print(f"Status: {result.status}")
            print(f"Total chunks: {result.total_chunks}")
            print(f"Indexed chunks: {result.indexed_chunks}")
            print(f"Batch count: {result.batch_count}")
            print(f"Vector size: {result.vector_size}")
            print(f"Complete: {result.is_complete}")

            if result.point_ids:
                print()
                print("First indexed point")
                print("-" * 50)
                print(result.point_ids[0])

            break

    finally:
        await close_qdrant_client()


def main() -> None:
    args = parse_arguments()

    try:
        asyncio.run(
            run(
                document_id=args.document_id,
                batch_size=args.batch_size,
            )
        )

    except DocumentNotFoundError as exc:
        print()
        print("Document not found")
        print("=" * 50)
        print(str(exc))
        raise SystemExit(1) from exc

    except DocumentNotReadyForIndexingError as exc:
        print()
        print("Document is not ready for indexing")
        print("=" * 50)
        print(str(exc))
        raise SystemExit(1) from exc

    except NoChunksAvailableForIndexingError as exc:
        print()
        print("No chunks available for indexing")
        print("=" * 50)
        print(str(exc))
        raise SystemExit(1) from exc

    except QdrantCollectionError as exc:
        print()
        print("Qdrant collection error")
        print("=" * 50)
        print(str(exc))
        raise SystemExit(1) from exc

    except VectorIndexingError as exc:
        print()
        print("Vector indexing failed")
        print("=" * 50)
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()