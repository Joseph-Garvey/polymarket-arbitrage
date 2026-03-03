import asyncio
from polymarket_client.api import PolymarketClient
from collections import defaultdict

async def main():
    client = PolymarketClient(dry_run=True)
    c_data = await client._request("GET", "/markets", params={"closed": "false", "limit": 1000}, base_url=client.gamma_url)
    
    events = defaultdict(list)
    for m in c_data:
        if m.get("negRisk"):
            ev_id = m.get("events", [{}])[0].get("id")
            if ev_id:
                events[ev_id].append(m.get("question"))
                
    for ev_id, qs in events.items():
        if len(qs) > 1:
            print(f"Event {ev_id}: {len(qs)} markets")
            for q in qs[:3]:
                print(f"  - {q}")

if __name__ == "__main__":
    asyncio.run(main())
