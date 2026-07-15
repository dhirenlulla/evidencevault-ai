import asyncio
import sys

from app.clients.qdrant import (
    close_qdrant_client,
    get_qdrant_client,
)
from app.services.qdrant_collection import (
    QdrantCollectionService,
    build_vector_collection_config,
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


async def run() -> None:
    """
    Ensure the Qdrant collection exists and print
    its compatibility status.
    """

    client = get_qdrant_client()

    config = build_vector_collection_config()

    service = QdrantCollectionService(
        client=client,
        config=config,
    )

    print()
    print("EvidenceVault Qdrant collection check")
    print("=" * 50)
    print(f"Collection: {config.collection_name}")
    print(f"Expected vector size: {config.vector_size}")
    print(f"Expected distance: {config.distance.value}")
    print()

    status = await service.ensure_collection()

    print("Collection status")
    print("-" * 50)
    print(f"Exists: {status.exists}")
    print(f"Vector size: {status.vector_size}")
    print(f"Distance: {status.distance}")
    print(f"Compatible: {status.is_compatible}")
    print(f"Points count: {status.points_count}")
    print(
        "Indexed vectors count: "
        f"{status.indexed_vectors_count}"
    )
    print(f"Message: {status.message}")

    await close_qdrant_client()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()