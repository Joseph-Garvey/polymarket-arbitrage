import json
import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    limit = 1000
    data = await api._request("GET", "/markets", params={"active": "true", "limit": limit})
    await api.disconnect()
    
    markets = data.get("data", data)
    
    multi_token_markets = [m for m in markets if len(m.get("tokens", [])) > 2]
    print(f"Markets with > 2 tokens: {len(multi_token_markets)}")
    
    # Are there any grouped by something else? Let's check neg_risk_market_id
    neg_risk = {}
    for m in markets:
        nrm = m.get("neg_risk_market_id", "")
        if nrm:
            neg_risk.setdefault(nrm, []).append(m)
            
    multi_neg = {k: v for k, v in neg_risk.items() if len(v) > 1}
    print(f"Markets sharing neg_risk_market_id: {len(multi_neg)}")
    if multi_neg:
        k, v = list(multi_neg.items())[0]
        print(f"Example group: {k}")
        for m in v:
            print(f"  - {m['question']} ({m['condition_id']})")

asyncio.run(run())
