import json
import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    markets = []
    has_more = True
    next_cursor = None
    
    while has_more and len(markets) < 5000:
        params = {"active": "true", "limit": 1000}
        if next_cursor:
            params["next_cursor"] = next_cursor
        data = await api._request("GET", "/markets", params=params)
        batch = data.get("data", [])
        markets.extend(batch)
        next_cursor = data.get("next_cursor")
        if next_cursor == "LTE=":
            break
        has_more = bool(next_cursor)
    
    await api.disconnect()
    
    questions = {}
    for m in markets:
        questions.setdefault(m.get("question", ""), []).append(m)
        
    multi_qs = {k: v for k, v in questions.items() if len(v) > 1}
    for q, ms in list(multi_qs.items())[:5]:
        print(f"\nQuestion: {q}")
        for m in ms:
            outcomes = [t.get("outcome") for t in m.get("tokens", [])]
            print(f"  - Market {m['condition_id'][:8]}... Outcomes: {outcomes}")

asyncio.run(run())
