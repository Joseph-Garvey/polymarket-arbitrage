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
    
    neg_risk = {}
    for m in markets:
        nrm = m.get("neg_risk_market_id", "")
        if nrm:
            neg_risk.setdefault(nrm, []).append(m)
            
    multi = {k: v for k, v in neg_risk.items() if len(v) > 1}
    
    for k, v in list(multi.items())[:5]:
        has_other = any("other" in m.get("question").lower() for m in v)
        print(f"Group: {v[0].get('question')} | Count: {len(v)} | Has 'Other': {has_other}")

asyncio.run(run())
