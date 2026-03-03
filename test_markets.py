import json
import asyncio
from polymarket_client.api import PolymarketClient

async def run():
    api = PolymarketClient()
    await api.connect()
    markets = await api.list_markets({"active": True, "limit": 500, "offset": 0})
    await api.disconnect()
    
    conditions = {}
    for m in markets:
        conditions.setdefault(m.condition_id, []).append(m.question)
    
    multi_conds = {k: v for k, v in conditions.items() if len(v) > 1}
    print(f"Total markets: {len(markets)}")
    print(f"Total unique condition_ids: {len(conditions)}")
    print(f"Conditions with >1 market: {len(multi_conds)}")
    
    questions = {}
    for m in markets:
        questions.setdefault(m.question, []).append(m)
        
    multi_qs = {k: v for k, v in questions.items() if len(v) > 1}
    print(f"Questions with >1 market: {len(multi_qs)}")
    for q, ms in list(multi_qs.items())[:3]:
        print(f"  {q}: {[m.market_id for m in ms]}")

asyncio.run(run())
