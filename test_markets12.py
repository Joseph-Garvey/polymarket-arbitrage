import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    
    # Just fetch from clob API
    markets = []
    next_cursor = None
    while len(markets) < 10000:
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
    
    neg_map = {}
    for m in markets:
        nrm = m.get("neg_risk_market_id", "")
        if nrm:
            neg_map[m["condition_id"]] = nrm
            
    print(f"Mapped {len(neg_map)} markets with neg_risk_market_id")

asyncio.run(run())
