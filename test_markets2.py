import json
import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    markets_data = await api._request("GET", "/events", params={"active": "true", "limit": 20})
    await api.disconnect()
    
    print(json.dumps(markets_data[:2], indent=2))

asyncio.run(run())
