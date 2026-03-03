with open('polymarket_client/api.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
in_block = False
for i, line in enumerate(lines):
    if 'event_type = msg.get("event_type")' in line and 'try:' in lines[i-1]:
        # found the start
        in_block = True
        
        # Insert new block
        out.append('                            messages = msg if isinstance(msg, list) else [msg]\n')
        out.append('                            for m in messages:\n')
        out.append('                                event_type = m.get("event_type")\n')
        out.append('                                asset_id = m.get("asset_id")\n')
        out.append('\n')
        out.append('                                if not asset_id or asset_id not in token_to_market:\n')
        out.append('                                    continue\n')
        out.append('\n')
        out.append('                                if asset_id not in live_books:\n')
        out.append('                                    live_books[asset_id] = {"bids": {}, "asks": {}}\n')
        out.append('\n')
        out.append('                                book = live_books[asset_id]\n')
        out.append('\n')
        out.append('                                if event_type == "book":\n')
        out.append('                                    # Full snapshot — replace state entirely\n')
        out.append('                                    book["bids"] = {\n')
        out.append('                                        float(b["price"]): float(b["size"])\n')
        out.append('                                        for b in m.get("bids", [])\n')
        out.append('                                    }\n')
        out.append('                                    book["asks"] = {\n')
        out.append('                                        float(a["price"]): float(a["size"])\n')
        out.append('                                        for a in m.get("asks", [])\n')
        out.append('                                    }\n')
        out.append('                                elif event_type == "price_change":\n')
        out.append('                                    # Delta — apply each change; size "0" removes the level\n')
        out.append('                                    for change in m.get("changes", []):\n')
        out.append('                                        raw_side = change.get("side", "")\n')
        out.append('                                        if not raw_side:\n')
        out.append('                                            logger.warning("price_change missing \'side\' field, skipping")\n')
        out.append('                                            continue\n')
        out.append('                                        price = float(change["price"])\n')
        out.append('                                        size = float(change["size"])\n')
        out.append('                                        side_key = "bids" if raw_side.upper() == "BUY" else "asks"\n')
        out.append('                                        if size == 0:\n')
        out.append('                                            book[side_key].pop(price, None)\n')
        out.append('                                        else:\n')
        out.append('                                            book[side_key][price] = size\n')
        out.append('                                else:\n')
        out.append('                                    continue\n')
        out.append('\n')
        out.append('                                if not got_first_message:\n')
        out.append('                                    got_first_message = True\n')
        out.append('                                    backoff = 1.0\n')
        out.append('\n')
        out.append('                                market_id, _ = token_to_market[asset_id]\n')
        out.append('                                yes_token, no_token = market_tokens[market_id]\n')
        out.append('\n')
        out.append('                                yield (market_id, OrderBook(\n')
        out.append('                                    market_id=market_id,\n')
        out.append('                                    yes=self._build_token_orderbook_from_state(\n')
        out.append('                                        live_books.get(yes_token, {"bids": {}, "asks": {}}),\n')
        out.append('                                        TokenType.YES,\n')
        out.append('                                    ),\n')
        out.append('                                    no=self._build_token_orderbook_from_state(\n')
        out.append('                                        live_books.get(no_token, {"bids": {}, "asks": {}}),\n')
        out.append('                                        TokenType.NO,\n')
        out.append('                                    ),\n')
        out.append('                                    timestamp=datetime.utcnow(),\n')
        out.append('                                ))\n')
        continue
    
    if in_block:
        if 'except Exception as e:' in line and 'logger.warning' in lines[i+1]:
            in_block = False
            out.append(line)
        continue
    
    out.append(line)

with open('polymarket_client/api.py', 'w', encoding='utf-8') as f:
    f.writelines(out)

print("Done")
