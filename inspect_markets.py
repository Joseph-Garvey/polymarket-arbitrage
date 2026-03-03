import asyncio
from polymarket_client.api import PolymarketClient
import logging
logging.basicConfig(level=logging.INFO)

async def main():
    client = PolymarketClient(dry_run=True)
    all_markets = await client.list_markets({"limit": 10})
    for m in all_markets[:5]:
        print(f"Market: {m.market_id}, Condition: {m.condition_id}, Group: {m.group_id}")

if __name__ == "__main__":
    asyncio.run(main())
