import asyncio
import sys
from uuid import UUID

from app.clients.qdrant import get_qdrant_client
from app.core.config import get_settings
from app.services.embedding import (
    get_embedding_service,
)
from app.services.qdrant_search import (
    QdrantSearchService,
)
from app.services.retrieval import (
    RetrievalService,
)
from app.clients.qdrant import close_qdrant_client

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


async def run(
    document_id: UUID,
    query: str,
) -> None:
    """
    Execute semantic retrieval for one document.
    """

    settings = get_settings()

    retrieval_service = RetrievalService(
        embedding_service=get_embedding_service(),
        qdrant_search_service=QdrantSearchService(
            client=get_qdrant_client(),
        ),
        collection_name=(
            settings.qdrant_collection_name
        ),
    )

    result = await retrieval_service.retrieve(
        document_id=document_id,
        query=query,
        limit=5,
    )

    print()
    print("EvidenceVault Retrieval")
    print("=" * 60)

    print(f"Query: {result.query}")
    print(f"Results: {result.total_results}")

    print()

    for index, chunk in enumerate(
        result.chunks,
        start=1,
    ):
        print("-" * 60)
        print(f"Rank: {index}")
        print(
            f"Similarity: "
            f"{chunk.similarity_score:.4f}"
        )
        print(f"Page: {chunk.page_number}")
        print(
            f"Chunk: {chunk.chunk_index}"
        )
        print()

        preview = (
            chunk.text[:250]
            .replace("\n", " ")
            .strip()
        )

        print(preview)

        print()


    await close_qdrant_client()

def main() -> None:

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "python -m scripts.search_document "
            "<document-id> "
            "\"query\""
        )
        return

    asyncio.run(
        run(
            UUID(sys.argv[1]),
            sys.argv[2],
        )
    )


if __name__ == "__main__":
    main()