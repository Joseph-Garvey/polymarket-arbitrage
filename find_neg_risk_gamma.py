import asyncio
from polymarket_client.api import PolymarketClient

async def main():
    client = PolymarketClient(dry_run=True)
    c_data = await client._request("GET", "/markets", params={"closed": "false", "limit": 1000}, base_url=client.gamma_url)
    
    for m in c_data:
        if m.get("negRisk") or m.get("negRiskOther"):
            print("Found Neg Risk Gamma market!", m.get("question"), m.get("conditionId"), m.get("negRiskOther"))
            break

if __name__ == "__main__":
    asyncio.run(main())
