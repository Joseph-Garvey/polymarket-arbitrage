import asyncio
from polymarket_client.api import PolymarketClient
import logging
logging.basicConfig(level=logging.INFO)

async def main():
    client = PolymarketClient(dry_run=True)
    c_data = await client._request("GET", "/markets", params={"active": "true", "limit": 2}, base_url=client.rest_url)
    batch = c_data.get("data", [])
    print(batch[0])

if __name__ == "__main__":
    asyncio.run(main())
