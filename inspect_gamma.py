import asyncio
from polymarket_client.api import PolymarketClient
import logging
logging.basicConfig(level=logging.INFO)

async def main():
    client = PolymarketClient(dry_run=True)
    c_data = await client._request("GET", "/markets", params={"limit": 1, "closed": "false"}, base_url=client.gamma_url)
    print(c_data[0])

if __name__ == "__main__":
    asyncio.run(main())
