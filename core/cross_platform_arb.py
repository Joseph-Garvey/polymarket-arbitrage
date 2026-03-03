"""
Cross-Platform Arbitrage Engine
===============================

Detects arbitrage opportunities between Polymarket and Kalshi prediction markets.

When the same prediction is priced differently on both platforms, we can:
- Buy YES on cheaper platform, sell YES on expensive platform
- Or buy NO on cheaper platform, sell NO on expensive platform
"""

import asyncio
import logging
import re
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
import hashlib

from polymarket_client.models import Market, OrderBook, Opportunity, OpportunityType
from utils.llm_client import MarketVerifier

logger = logging.getLogger(__name__)


@dataclass
class MarketPair:
    """A matched pair of markets on Polymarket and Kalshi."""

    polymarket_id: str
    kalshi_ticker: str
    polymarket_question: str
    kalshi_title: str
    similarity_score: float
    category: str = ""
    is_llm_verified: bool = False

    # Timestamps
    matched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def pair_id(self) -> str:
        """Unique identifier for this pair."""
        return f"poly:{self.polymarket_id}|kalshi:{self.kalshi_ticker}"


@dataclass
class CrossPlatformOpportunity:
    """Arbitrage opportunity between Polymarket and Kalshi."""

    opportunity_id: str
    market_pair: MarketPair

    # Direction: which platform to buy/sell on
    buy_platform: str  # "polymarket" or "kalshi"
    sell_platform: str
    token: str  # "YES" or "NO"

    # Prices
    buy_price: float
    sell_price: float

    # Edge calculation
    gross_edge: float  # sell_price - buy_price
    net_edge: float  # After fees
    edge_pct: float  # As percentage

    # Sizing
    suggested_size: float = 0.0
    max_size: float = 0.0  # Limited by liquidity on both sides

    # Liquidity available
    buy_liquidity: float = 0.0
    sell_liquidity: float = 0.0

    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        return (
            f"CrossPlatformArb: Buy {self.token} on {self.buy_platform} @ ${self.buy_price:.3f}, "
            f"Sell on {self.sell_platform} @ ${self.sell_price:.3f} | "
            f"Net Edge: {self.edge_pct:.2%}"
        )


class MarketMatcher:
    """
    Matches similar markets between Polymarket and Kalshi.

    Uses text similarity, hard filters, and LLM verification to find
    markets that represent the same underlying prediction.
    """

    # Keywords to normalize/remove for matching
    NOISE_WORDS = {
        "will",
        "the",
        "a",
        "an",
        "be",
        "to",
        "in",
        "on",
        "by",
        "at",
        "what",
        "who",
        "which",
        "when",
        "is",
        "are",
        "was",
        "were",
        "market",
        "prediction",
        "bet",
        "odds",
        "win",
        "winner",
    }

    # NFL and NBA team mappings
    NFL_TEAMS = {
        "arizona cardinals": ["cardinals", "arizona", "ari"],
        "atlanta falcons": ["falcons", "atlanta", "atl"],
        "baltimore ravens": ["ravens", "baltimore", "bal"],
        "buffalo bills": ["bills", "buffalo", "buf"],
        "carolina panthers": ["panthers", "carolina", "car"],
        "chicago bears": ["bears", "chicago", "chi"],
        "cincinnati bengals": ["bengals", "cincinnati", "cin"],
        "cleveland browns": ["browns", "cleveland", "cle"],
        "dallas cowboys": ["cowboys", "dallas", "dal"],
        "denver broncos": ["broncos", "denver", "den"],
        "detroit lions": ["lions", "detroit", "det"],
        "green bay packers": ["packers", "green bay", "gb"],
        "houston texans": ["texans", "houston", "hou"],
        "indianapolis colts": ["colts", "indianapolis", "ind"],
        "jacksonville jaguars": ["jaguars", "jacksonville", "jax"],
        "kansas city chiefs": ["chiefs", "kansas city", "kc"],
        "las vegas raiders": ["raiders", "las vegas", "lv"],
        "los angeles chargers": ["chargers", "la chargers", "lac"],
        "los angeles rams": ["rams", "la rams", "lar"],
        "miami dolphins": ["dolphins", "miami", "mia"],
        "minnesota vikings": ["vikings", "minnesota", "min"],
        "new england patriots": ["patriots", "new england", "ne"],
        "new orleans saints": ["saints", "new orleans", "no"],
        "new york giants": ["giants", "ny giants", "nyg"],
        "new york jets": ["jets", "ny jets", "nyj"],
        "philadelphia eagles": ["eagles", "philadelphia", "phi"],
        "pittsburgh steelers": ["steelers", "pittsburgh", "pit"],
        "san francisco 49ers": ["49ers", "san francisco", "sf"],
        "seattle seahawks": ["seahawks", "seattle", "sea"],
        "tampa bay buccaneers": ["buccaneers", "tampa bay", "tb"],
        "tennessee titans": ["titans", "tennessee", "ten"],
        "washington commanders": ["commanders", "washington", "was"],
    }
    SOCCER_TEAMS = {
        "chelsea": ["chelsea"],
        "arsenal": ["arsenal"],
        "manchester united": ["manchester united", "man utd"],
        "manchester city": ["manchester city", "man city"],
        "liverpool": ["liverpool"],
        "tottenham": ["tottenham", "spurs"],
        "barcelona": ["barcelona", "fc barcelona"],
        "real madrid": ["real madrid"],
        "bayern munich": ["bayern munich", "bayern"],
        "paris saint-germain": ["paris saint-germain", "psg"],
    }
    NBA_TEAMS = {
        "boston celtics": ["celtics", "boston"],
        "brooklyn nets": ["nets", "brooklyn"],
        "new york knicks": ["knicks", "new york"],
        "philadelphia 76ers": ["76ers", "sixers", "philadelphia"],
        "toronto raptors": ["raptors", "toronto"],
        "chicago bulls": ["bulls", "chicago"],
        "cleveland cavaliers": ["cavaliers", "cavs", "cleveland"],
        "detroit pistons": ["pistons", "detroit"],
        "indiana pacers": ["pacers", "indiana"],
        "milwaukee bucks": ["bucks", "milwaukee"],
        "atlanta hawks": ["hawks", "atlanta"],
        "charlotte hornets": ["hornets", "charlotte"],
        "miami heat": ["heat", "miami"],
        "orlando magic": ["magic", "orlando"],
        "washington wizards": ["wizards", "washington"],
        "denver nuggets": ["nuggets", "denver"],
        "minnesota timberwolves": ["timberwolves", "wolves", "minnesota"],
        "oklahoma city thunder": ["thunder", "okc"],
        "portland trail blazers": ["blazers", "portland"],
        "utah jazz": ["jazz", "utah"],
        "golden state warriors": ["warriors", "golden state"],
        "los angeles clippers": ["clippers", "la clippers"],
        "los angeles lakers": ["lakers", "la lakers"],
        "phoenix suns": ["suns", "phoenix"],
        "sacramento kings": ["kings", "sacramento"],
        "dallas mavericks": ["mavericks", "mavs", "dallas"],
        "houston rockets": ["rockets", "houston"],
        "memphis grizzlies": ["grizzlies", "memphis"],
        "new orleans pelicans": ["pelicans", "new orleans"],
        "san antonio spurs": ["spurs", "san antonio"],
    }

    def __init__(
        self,
        min_similarity: float = 0.70,
        use_llm: bool = True,
        openrouter_api_key: str = "",
        openrouter_model: str = "",
    ):
        self.min_similarity = min_similarity
        self.use_llm = use_llm
        self._matched_pairs: dict[str, MarketPair] = {}
        self._verification_cache_path = "logs/verified_matches.json"
        self._verification_cache = self._load_cache()
        self._llm_verifier = (
            MarketVerifier(api_key=openrouter_api_key, model=openrouter_model)
            if use_llm
            else None
        )

        # Build reverse lookup for team names
        self._team_lookup = {}
        for full_name, variants in {
            **self.NFL_TEAMS,
            **self.NBA_TEAMS,
            **self.SOCCER_TEAMS,
        }.items():
            self._team_lookup[full_name] = full_name
            for variant in variants:
                self._team_lookup[variant.lower()] = full_name

    def _load_cache(self) -> dict:
        if os.path.exists(self._verification_cache_path):
            try:
                with open(self._verification_cache_path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs("logs", exist_ok=True)
        with open(self._verification_cache_path, "w") as f:
            json.dump(self._verification_cache, f, indent=2)

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.split()
        words = [w for w in words if w not in self.NOISE_WORDS]
        return " ".join(words)

    def extract_teams(self, text: str) -> list[str]:
        text_lower = text.lower()
        found_teams = []
        for team_key in sorted(self._team_lookup.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(team_key) + r"\b"
            if re.search(pattern, text_lower):
                canonical = self._team_lookup[team_key]
                if canonical not in found_teams:
                    found_teams.append(canonical)
                    text_lower = re.sub(pattern, "", text_lower)
        return found_teams

    def extract_key_entities(self, text: str) -> set[str]:
        entities = set()
        # Precise Numbers (Normalize k/m)
        raw_nums = re.findall(r"[\$€£]?\d+(?:[\d,\.]*\d+)?%?[kmb]?", text.lower())
        for rn in raw_nums:
            clean = re.sub(r"[\$€£,]", "", rn)
            if clean.endswith("k"):
                try:
                    clean = str(int(float(clean[:-1]) * 1000))
                except ValueError:
                    pass
            elif clean.endswith("m"):
                try:
                    clean = str(int(float(clean[:-1]) * 1000000))
                except ValueError:
                    pass
            entities.add(clean)

        # Better proper noun extraction for matching
        # Look for sequences of capitalized words that aren't common noise
        potential_names = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text)
        for name in potential_names:
            if name.lower() not in self.NOISE_WORDS:
                entities.add(name.lower())

        # Specific keywords that are high signal
        keywords = {
            "trump",
            "biden",
            "harris",
            "election",
            "bitcoin",
            "btc",
            "eth",
            "detonation",
            "nuclear",
            "war",
            "invade",
            "recession",
        }
        for kw in keywords:
            if kw in text.lower():
                entities.add(kw)
        return entities

    def _get_action_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(x in text_lower for x in ["win", "victory", "first"]):
            return "win"
        if any(x in text_lower for x in ["above", "greater", "over", ">"]):
            return "above"
        if any(x in text_lower for x in ["below", "less", "under", "<"]):
            return "below"
        return "generic"

    def extract_date(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        months = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
            "jan": "01",
            "feb": "02",
            "mar": "03",
            "apr": "04",
            "jun": "06",
            "jul": "07",
            "aug": "08",
            "sep": "09",
            "oct": "10",
            "nov": "11",
            "dec": "12",
        }
        for name, num in months.items():
            match = re.search(
                rf"{name}\.?\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?", text_lower
            )
            if match:
                return f"{match.group(2) or '2025'}-{num}-{match.group(1).zfill(2)}"
        return None

    def dates_match(self, d1: Optional[str], d2: Optional[str]) -> bool:
        if not d1 or not d2:
            return True
        return d1 == d2

    def _fast_similarity(self, poly: dict, kalshi: dict) -> float:
        # Score calculation
        if len(poly["teams"]) >= 2 and len(kalshi["teams"]) >= 2:
            shared = poly["teams"] & kalshi["teams"]
            if len(shared) >= 2:
                # Still check for specific keywords that might distinguish markets
                # e.g., "win by 10+" vs "win outright"
                if "above" in kalshi["entities"] or "below" in kalshi["entities"]:
                    if (
                        "above" not in poly["entities"]
                        and "below" not in poly["entities"]
                    ):
                        return (
                            0.75  # Lower score to force LLM check, not 0.95 auto-pass
                        )
                return 0.95 if self.dates_match(poly["date"], kalshi["date"]) else 0.0
            return 0.0

        # HARD FILTER 1: Multi-Outcome / Spread Markets (Keep strict)
        # Kalshi spread markets often contain multiple "yes" or point values
        if any(x in kalshi["text"].lower() for x in ["points", "spread", "by over"]):
            if not any(
                x in poly["text"].lower() for x in ["points", "spread", "by over"]
            ):
                return 0.0

        # Relaxed Filters: We let the LLM handle the nuances if there is basic keyword overlap

        # Check for numeric consistency if both have numbers
        poly_nums = {n for n in poly["entities"] if any(c.isdigit() for c in n)}
        kalshi_nums = {n for n in kalshi["entities"] if any(c.isdigit() for c in n)}
        numeric_conflict = False
        if poly_nums and kalshi_nums and not (poly_nums & kalshi_nums):
            numeric_conflict = True

        # Check for basic entity overlap (Subject matching)
        shared_entities = poly["entities"] & kalshi["entities"]

        # Check for action alignment
        pa, ka = (
            self._get_action_type(poly["text"]),
            self._get_action_type(kalshi["text"]),
        )
        action_conflict = False
        if pa != ka and pa != "generic" and ka != "generic":
            if (pa == "above" and ka == "below") or (pa == "below" and ka == "above"):
                action_conflict = True

        # If there's a hard numeric or action conflict, reject immediately
        if numeric_conflict or action_conflict:
            return 0.0

        # If no shared entities and no significant token overlap, reject
        intersection = poly["tokens"] & kalshi["tokens"]
        if not shared_entities and len(intersection) < 2:
            return 0.0

        # Calculate a "Candidate Score"
        # This score doesn't need to be perfect, just high enough to trigger Stage 2 (LLM)
        text_sim = (
            len(intersection) / len(poly["tokens"] | kalshi["tokens"])
            if poly["tokens"]
            else 0
        )
        ent_overlap = len(shared_entities) / max(
            len(poly["entities"]), len(kalshi["entities"]), 1
        )

        return 0.3 * text_sim + 0.7 * ent_overlap

    @staticmethod
    def _is_multi_outcome_title(title: str) -> bool:
        return bool(re.search(r"yes\s+\w[^,]*,\s*yes\s+\w", title, re.IGNORECASE))

    def _precompute_market_data(self, text: str) -> dict:
        text_lower = text.lower()
        norm = self.normalize_text(text)
        return {
            "text": text,
            "tokens": set(norm.split()) if norm else set(),
            "teams": set(self.extract_teams(text)),
            "entities": self.extract_key_entities(text),
            "date": self.extract_date(text),
            "persons": set(re.findall(r"\b(trump|biden|harris)\b", text_lower)),
            "action_words": set(re.findall(r"\b(win|lose|elect|ipo)\b", text_lower)),
            "sports": {"nba", "nfl", "mlb"} & set(text_lower.split()),
            "crypto": {"btc", "eth", "sol"} & set(text_lower.split()),
        }

    def _categorize_market(self, text: str) -> str:
        tl = text.lower()

        if any(
            x in tl
            for x in [
                "trump",
                "election",
                "biden",
                "harris",
                "congress",
                "senate",
                "supreme",
                "governor",
                "iran",
                "china",
                "russia",
                "israel",
            ]
        ):
            return "politics"
        if any(
            x in tl for x in ["bitcoin", "btc", "eth", "ethereum", "solana", "crypto"]
        ):
            return "crypto"

        has_team = self.extract_teams(text)
        if has_team:
            return "sports"

        sports_keywords = [
            "nba",
            "nfl",
            "mlb",
            "nhl",
            "fifa",
            "uefa",
            "championship",
            "playoffs",
            "regular season",
        ]
        if any(x in tl for x in sports_keywords):
            return "sports"

        return "other"

    async def find_matches(
        self, polymarket_markets: List[Market], kalshi_markets: List, on_progress=None
    ) -> List[MarketPair]:
        matches = []
        candidates = []
        active_poly = [m for m in polymarket_markets if m.active]
        active_kalshi = [
            m
            for m in kalshi_markets
            if m.is_active and not self._is_multi_outcome_title(m.title)
        ]

        poly_by_cat = {}
        for m in active_poly:
            cat = self._categorize_market(m.question)
            poly_by_cat.setdefault(cat, []).append(m)

        kalshi_by_cat = {}
        for m in active_kalshi:
            cat = (m.category or "").lower()
            if cat not in ["sports", "politics", "crypto"]:
                cat = self._categorize_market(m.title)
            kalshi_by_cat.setdefault(cat, []).append(m)

        total_comparisons = sum(
            len(poly_by_cat.get(cat, [])) * len(kalshi_by_cat.get(cat, []))
            for cat in ["sports", "politics", "crypto"]
        )
        comparisons_done = 0

        for cat in ["sports", "politics", "crypto"]:
            p_list = poly_by_cat.get(cat, [])
            k_list = kalshi_by_cat.get(cat, [])
            if not p_list or not k_list:
                continue

            logger.info(f"Matching {cat}: {len(p_list)} x {len(k_list)}")
            p_data = [self._precompute_market_data(m.question) for m in p_list]
            k_data = [self._precompute_market_data(m.title) for m in k_list]

            for i, p_market in enumerate(p_list):
                best_match, best_score = None, 0.0
                for j, k_market in enumerate(k_list):
                    comparisons_done += 1

                    score = self._fast_similarity(p_data[i], k_data[j])
                    if score > best_score:
                        best_score, best_match = score, k_market
                    if best_score >= 0.95:
                        break

                if best_match and best_score >= self.min_similarity:
                    candidates.append((p_market, best_match, best_score, cat))
                    if on_progress:
                        on_progress(
                            comparisons_done, total_comparisons, len(candidates)
                        )

        logger.info(f"Stage 1: Found {len(candidates)} algorithmic candidates")

        for p_market, k_market, score, cat in candidates:
            pair = MarketPair(
                polymarket_id=p_market.market_id,
                kalshi_ticker=k_market.ticker,
                polymarket_question=p_market.question,
                kalshi_title=k_market.title,
                similarity_score=score,
                category=cat,
            )

            if self._llm_verifier and self._llm_verifier.is_cached(
                p_market.market_id, k_market.ticker
            ):
                cached_result = self._llm_verifier.get_cached_result(
                    p_market.market_id, k_market.ticker
                )
                if cached_result is False:
                    logger.debug(
                        f"Cache reject: '{pair.polymarket_question[:30]}' <-> '{pair.kalshi_title[:30]}'"
                    )
                    continue
                pair.is_llm_verified = True
                matches.append(pair)
                continue

            if self._llm_verifier and score >= self.min_similarity:
                llm_result = await self._llm_verifier.verify(
                    p_market.question,
                    k_market.title,
                    p_market.market_id,
                    k_market.ticker,
                )
                if llm_result is False:
                    logger.info(
                        f"LLM rejected: '{pair.polymarket_question[:40]}' <-> '{pair.kalshi_title[:40]}'"
                    )
                    continue
                pair.is_llm_verified = True
            elif score >= 0.95:
                pair.is_llm_verified = True

            matches.append(pair)
            logger.info(
                f"MATCH: '{pair.polymarket_question[:40]}' <-> '{pair.kalshi_title[:40]}' (score: {pair.similarity_score:.2f}, llm: {pair.is_llm_verified})"
            )

        return matches

    def get_cached_pairs(self) -> list[MarketPair]:
        return list(self._matched_pairs.values())


class CrossPlatformArbEngine:
    def __init__(
        self,
        min_edge: float = 0.02,
        polymarket_taker_fee: float = 0.015,
        kalshi_taker_fee: float = 0.01,
        gas_cost: float = 0.02,
        use_llm: bool = True,
        openrouter_api_key: str = "",
        openrouter_model: str = "",
    ):
        self.min_edge = min_edge
        self.polymarket_taker_fee = polymarket_taker_fee
        self.kalshi_taker_fee = kalshi_taker_fee
        self.gas_cost = gas_cost
        self.matcher = MarketMatcher(
            use_llm=use_llm,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model,
        )
        self._opportunities = []

    def get_llm_stats(self) -> dict:
        """Get LLM verification statistics."""
        if self.matcher._llm_verifier:
            return self.matcher._llm_verifier.get_stats()
        return {"total": 0, "verified": 0, "rejected": 0}

    def check_arbitrage(
        self, pair: MarketPair, poly_ob: OrderBook, kalshi_ob: OrderBook
    ) -> Optional[CrossPlatformOpportunity]:
        prices = {
            "p_y_a": poly_ob.best_ask_yes,
            "p_y_b": poly_ob.best_bid_yes,
            "p_n_a": poly_ob.best_ask_no,
            "p_n_b": poly_ob.best_bid_no,
            "k_y_a": kalshi_ob.best_ask_yes,
            "k_y_b": kalshi_ob.best_bid_yes,
            "k_n_a": kalshi_ob.best_ask_no,
            "k_n_b": kalshi_ob.best_bid_no,
        }
        p_y_a = prices["p_y_a"]
        k_y_b = prices["k_y_b"]
        if p_y_a is None or k_y_b is None:
            return None

        # YES Arb: Buy Poly, Sell Kalshi
        gross = k_y_b - p_y_a
        fees = (
            (p_y_a * self.polymarket_taker_fee)
            + (k_y_b * self.kalshi_taker_fee)
            + (self.gas_cost * 2)
        )
        if gross - fees >= self.min_edge:
            return self._create_opp(
                pair,
                "polymarket",
                "kalshi",
                "YES",
                p_y_a,
                k_y_b,
                gross,
                gross - fees,
            )

        return None

    def _create_opp(
        self, pair, bp, sp, tok, bpr, spr, gr, net
    ) -> CrossPlatformOpportunity:
        return CrossPlatformOpportunity(
            opportunity_id=f"xplat_{len(self._opportunities) + 1}",
            market_pair=pair,
            buy_platform=bp,
            sell_platform=sp,
            token=tok,
            buy_price=bpr,
            sell_price=spr,
            gross_edge=gr,
            net_edge=net,
            edge_pct=net / bpr if bpr > 0 else 0,
        )
