import asyncio
from polymarket_client.api import PolymarketClient

async def main():
    client = PolymarketClient(dry_run=True)
    c_data = await client._request("GET", "/markets", params={"closed": "false", "limit": 100}, base_url=client.rest_url)
    batch = c_data.get("data", [])
    
    print("Fetched", len(batch), "markets")
    for m in batch:
        if m.get("neg_risk") or m.get("neg_risk_market_id"):
            print("Found Neg Risk CLOB market!", m.get("question"), m.get("neg_risk_market_id"))
            break

if __name__ == "__main__":
    asyncio.run(main())
