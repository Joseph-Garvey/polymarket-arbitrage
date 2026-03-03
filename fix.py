import re

with open('polymarket_client/api.py', 'r') as f:
    content = f.read()

old_str = """                        try:
                            event_type = msg.get("event_type")
                            asset_id = msg.get("asset_id")

                            if not asset_id or asset_id not in token_to_market:
                                continue

                            if asset_id not in live_books:
                                live_books[asset_id] = {"bids": {}, "asks": {}}

                            book = live_books[asset_id]

                            if event_type == "book":
                                # Full snapshot — replace state entirely
                                book["bids"] = {
                                    float(b["price"]): float(b["size"])
                                    for b in msg.get("bids", [])
                                }
                                book["asks"] = {
                                    float(a["price"]): float(a["size"])
                                    for a in msg.get("asks", [])
                                }
                            elif event_type == "price_change":
                                # Delta — apply each change; size "0" removes the level
                                for change in msg.get("changes", []):
                                    raw_side = change.get("side", "")
                                    if not raw_side:
                                        logger.warning("price_change missing 'side' field, skipping")
                                        continue
                                    price = float(change["price"])
                                    size = float(change["size"])
                                    side_key = "bids" if raw_side.upper() == "BUY" else "asks"
                                    if size == 0:
                                        book[side_key].pop(price, None)
                                    else:
                                        book[side_key][price] = size
                            else:
                                continue

                            if not got_first_message:
                                got_first_message = True
                                backoff = 1.0

                            market_id, _ = token_to_market[asset_id]
                            yes_token, no_token = market_tokens[market_id]

                            yield (market_id, OrderBook(
                                market_id=market_id,
                                yes=self._build_token_orderbook_from_state(
                                    live_books.get(yes_token, {"bids": {}, "asks": {}}),
                                    TokenType.YES,
                                ),
                                no=self._build_token_orderbook_from_state(
                                    live_books.get(no_token, {"bids": {}, "asks": {}}),
                                    TokenType.NO,
                                ),
                                timestamp=datetime.utcnow(),
                            ))
                        except Exception as e:
                            logger.warning(f"Error processing WS message: {e}")
                            continue"""

new_str = """                        try:
                            messages = msg if isinstance(msg, list) else [msg]
                            for m in messages:
                                event_type = m.get("event_type")
                                asset_id = m.get("asset_id")

                                if not asset_id or asset_id not in token_to_market:
                                    continue

                                if asset_id not in live_books:
                                    live_books[asset_id] = {"bids": {}, "asks": {}}

                                book = live_books[asset_id]

                                if event_type == "book":
                                    # Full snapshot — replace state entirely
                                    book["bids"] = {
                                        float(b["price"]): float(b["size"])
                                        for b in m.get("bids", [])
                                    }
                                    book["asks"] = {
                                        float(a["price"]): float(a["size"])
                                        for a in m.get("asks", [])
                                    }
                                elif event_type == "price_change":
                                    # Delta — apply each change; size "0" removes the level
                                    for change in m.get("changes", []):
                                        raw_side = change.get("side", "")
                                        if not raw_side:
                                            logger.warning("price_change missing 'side' field, skipping")
                                            continue
                                        price = float(change["price"])
                                        size = float(change["size"])
                                        side_key = "bids" if raw_side.upper() == "BUY" else "asks"
                                        if size == 0:
                                            book[side_key].pop(price, None)
                                        else:
                                            book[side_key][price] = size
                                else:
                                    continue

                                if not got_first_message:
                                    got_first_message = True
                                    backoff = 1.0

                                market_id, _ = token_to_market[asset_id]
                                yes_token, no_token = market_tokens[market_id]

                                yield (market_id, OrderBook(
                                    market_id=market_id,
                                    yes=self._build_token_orderbook_from_state(
                                        live_books.get(yes_token, {"bids": {}, "asks": {}}),
                                        TokenType.YES,
                                    ),
                                    no=self._build_token_orderbook_from_state(
                                        live_books.get(no_token, {"bids": {}, "asks": {}}),
                                        TokenType.NO,
                                    ),
                                    timestamp=datetime.utcnow(),
                                ))
                        except Exception as e:
                            logger.warning(f"Error processing WS message: {e}")
                            continue"""

if old_str in content:
    with open('polymarket_client/api.py', 'w') as f:
        f.write(content.replace(old_str, new_str))
    print("Successfully replaced.")
else:
    print("String not found. Exiting.")
