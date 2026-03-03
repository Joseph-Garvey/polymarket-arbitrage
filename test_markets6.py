import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    
    # Just paginate through first 5000 markets on CLOB
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
    
    neg_risk = {}
    for m in markets:
        nrm = m.get("neg_risk_market_id", "")
        if nrm:
            neg_risk.setdefault(nrm, []).append(m)
            
    print(f"Total active markets checked: {len(markets)}")
    print(f"Unique neg_risk_market_ids: {len(neg_risk)}")
    
    multi = {k: v for k, v in neg_risk.items() if len(v) > 1}
    print(f"NegRisk groups with >1 market: {len(multi)}")
    
    for k, v in list(multi.items())[:3]:
        print(f"\nGroup ID: {k}")
        print(f"Question: {v[0].get('question')}")
        for m in v:
            outcomes = [t.get("outcome") for t in m.get("tokens", [])]
            print(f"  - {outcomes[0] if outcomes else 'Unknown'} (Market ID: {m.get('condition_id')[:8]})")

asyncio.run(run())
