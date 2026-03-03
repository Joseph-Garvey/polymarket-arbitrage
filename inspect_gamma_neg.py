import asyncio
from polymarket_client.api import PolymarketClient

async def main():
    client = PolymarketClient(dry_run=True)
    c_data = await client._request("GET", "/markets", params={"closed": "false", "limit": 100}, base_url=client.gamma_url)
    
    for m in c_data:
        if m.get("negRisk"):
            print("Found Neg Risk Gamma market!")
            print(m.get("question"))
            print("events", [e.get('id') for e in m.get("events", [])])
            print("groupItemTitle", m.get("groupItemTitle"))
            break

if __name__ == "__main__":
    asyncio.run(main())
