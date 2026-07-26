import asyncio

from app.clients.qdrant import get_qdrant_client


async def main():
    client = get_qdrant_client()

    print("=" * 60)

    collections = await client.get_collections()

    print(collections)

    print("=" * 60)

    info = await client.get_collection(
        "evidencevault_chunks"
    )

    print(info)

    print("=" * 60)

    result = await client.scroll(
        collection_name="evidencevault_chunks",
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    print(result)

    print("=" * 60)


asyncio.run(main())