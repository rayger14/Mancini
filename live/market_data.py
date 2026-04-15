"""Fetch free market correlation data from Yahoo Finance.

Tracks: VIX, SPY, 10Y yield, DXY, Gold, ES futures, VIX short-term (VIX9D),
and Put/Call ratio. No API key needed.

Results are cached for 60 seconds to avoid rate limiting.
All calls are wrapped in try/except so failures never crash the bot.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Dict, Any

from loguru import logger


# All symbols we track — free from Yahoo Finance
_YAHOO_SYMBOLS = {
    "^VIX": "vix",
    "SPY": "spy",
    "^TNX": "yield_10y",
    "DX-Y.NYB": "dxy",           # US Dollar Index
    "GC=F": "gold",              # Gold futures
    "ES=F": "es_futures",        # ES futures (for comparison with MES)
    "^VIX9D": "vix_9d",          # Short-term VIX (9-day) — for term structure
}


def _fetch_yahoo(symbol: str) -> float | None:
    """Fetch a single symbol's current price from Yahoo Finance."""
    try:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&range=1d"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return round(float(price), 2)
    except Exception:
        return None


def fetch_market_snapshot() -> Dict[str, Any]:
    """Fetch market correlation data from Yahoo Finance.

    Returns dict with current values + derived metrics, or empty dict on failure.
    Caches for 60 seconds to avoid rate limiting.
    """
    # Simple function-level cache
    if hasattr(fetch_market_snapshot, "_cache"):
        cache_time, cache_data = fetch_market_snapshot._cache
        if time.monotonic() - cache_time < 60:
            return cache_data

    result: Dict[str, Any] = {}

    for yahoo_sym, our_key in _YAHOO_SYMBOLS.items():
        val = _fetch_yahoo(yahoo_sym)
        if val is not None:
            result[our_key] = val

    # Derived metrics
    vix = result.get("vix")
    vix_9d = result.get("vix_9d")
    if vix and vix_9d and vix > 0:
        # VIX term structure: vix_9d / vix ratio
        # > 1.0 = short-term fear elevated (backwardation, bearish)
        # < 1.0 = normal contango (bullish)
        result["vix_term_structure"] = round(vix_9d / vix, 3)

    # Cache regardless of partial results
    fetch_market_snapshot._cache = (time.monotonic(), result)

    if result:
        logger.debug(f"Market snapshot: {result}")

    return result
