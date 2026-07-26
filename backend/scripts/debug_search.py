import asyncio

from app.clients.qdrant import get_qdrant_client


async def main():

    client = get_qdrant_client()

    result = await client.scroll(
        collection_name="evidencevault_chunks",
        limit=1,
        with_vectors=True,
        with_payload=True,
    )

    point = result[0][0]

    print("Vector length:", len(point.vector))

    print(point.vector[:10])


asyncio.run(main())