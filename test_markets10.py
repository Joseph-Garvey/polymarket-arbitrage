import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    
    data = await api._request("GET", "/markets", params={"active": "true", "limit": 100})
    markets = data.get("data", [])
    await api.disconnect()
    
    for m in markets:
        if m.get("neg_risk_market_id"):
            tokens = m.get("tokens", [])
            print(f"Market: {m.get('question')} | Tokens: {[t.get('outcome') for t in tokens]}")
            break

asyncio.run(run())
