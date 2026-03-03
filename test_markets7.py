import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    
    params = {"active": "true", "limit": 1000}
    data = await api._request("GET", "/markets", params=params)
    markets = data.get("data", [])
    
    await api.disconnect()
    
    neg_risk = {}
    for m in markets:
        nrm = m.get("neg_risk_market_id", "")
        if nrm:
            neg_risk.setdefault(nrm, []).append(m)
            
    multi = {k: v for k, v in neg_risk.items() if len(v) > 1}
    
    for k, v in list(multi.items())[:2]:
        print(f"\nGroup ID: {k}")
        for m in v[:5]:
            print(f"  - {m.get('question')}")

asyncio.run(run())
