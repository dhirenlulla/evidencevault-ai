import asyncio

from app.clients.qdrant import get_qdrant_client


async def main():

    client = get_qdrant_client()

    result = await client.scroll(
        collection_name="evidencevault_chunks",
        limit=1,
        with_vectors=True,
    )

    point = result[0][0]

    vector = point.vector

    print("Running query...")

    result = await client.query_points(
        collection_name="evidencevault_chunks",
        query=vector,
        limit=5,
    )

    print(result)


asyncio.run(main())