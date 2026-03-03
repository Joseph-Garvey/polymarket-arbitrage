import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    
    markets = []
    next_cursor = None
    while len(markets) < 5000:
        params = {"active": "true", "limit": 1000}
        if next_cursor:
            params["next_cursor"] = next_cursor
        data = await api._request("GET", "/markets", params=params)
        batch = data.get("data", [])
        markets.extend(batch)
        next_cursor = data.get("next_cursor")
        if not next_cursor or next_cursor == "LTE=":
            break
            
    await api.disconnect()
    
    for m in markets:
        if m.get("neg_risk_market_id"):
            tokens = m.get("tokens", [])
            print(f"Market: {m.get('question')} | Tokens: {[t.get('outcome') for t in tokens]}")
            break

asyncio.run(run())
