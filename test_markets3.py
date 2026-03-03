import json
import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    data = await api._request("GET", "/markets", params={"active": "true", "limit": 20})
    await api.disconnect()
    
    if "data" in data:
        markets = data["data"]
    else:
        markets = data
        
    print(json.dumps(markets[0], indent=2))

asyncio.run(run())
