"""Optional stock price tool (turned off by default; it calls the internet).

Primary source: the free Stooq daily CSV (no API key). If that fails and
``yfinance`` is installed (``uv sync --extra market``), it is used as a fallback.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request
from typing import Callable

logger = logging.getLogger("helm.market")

Fetcher = Callable[[str], str]

_HEADERS = {"User-Agent": "Mozilla/5.0 (PeakWhale Helm)", "Accept": "text/csv,*/*;q=0.8"}


class MarketError(Exception):
    pass


def clean_ticker(ticker: str) -> str:
    text = (ticker or "").strip().upper().replace("$", "")
    return re.sub(r"[^A-Z0-9.\-^=]", "", text)[:15]


def _stooq_symbol(ticker: str) -> str:
    t = ticker.lstrip("^")
    return t.lower() if "." in t else f"{t.lower()}.us"


def _http_get(url: str) -> str:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - fixed https host
        return response.read().decode("utf-8", errors="replace")


def _parse_stooq(text: str) -> tuple[float, str]:
    body = (text or "").strip()
    if not body or body.lstrip().lower().startswith("<"):
        raise MarketError("Stooq did not return CSV data")
    rows = [row for row in csv.DictReader(io.StringIO(body)) if row]
    if not rows:
        raise MarketError("Stooq returned no rows")
    row = rows[-1]
    close = (row.get("Close") or row.get("close") or "").strip()
    if not close or close.lower() in {"nan", "null", "n/d"}:
        raise MarketError("Stooq has no close price for this ticker")
    return float(close), (row.get("Date") or row.get("date") or "").strip()


def get_stock_price(ticker: str, fetch: Fetcher | None = None) -> str:
    symbol = clean_ticker(ticker)
    if not symbol:
        raise MarketError("Missing ticker. Try AAPL or $TSLA.")
    live = fetch is None  # a fetcher passed in (tests) means: do not touch the network
    fetch = fetch or _http_get
    errors = []
    for url in (
        f"https://stooq.com/q/d/l/?s={_stooq_symbol(symbol)}&i=d",
        f"https://stooq.com/q/l/?s={_stooq_symbol(symbol)}&f=sd2t2ohlcv&h&e=csv",
    ):
        try:
            price, as_of = _parse_stooq(fetch(url))
            return f"Ticker: {symbol}\nPrice: ${price:,.2f}\nAs of: {as_of or 'latest'} (daily close, Stooq)"
        except Exception as exc:  # network errors, bad CSV
            errors.append(str(exc))

    yf = None
    if live:
        try:
            import yfinance as yf  # optional extra: uv sync --extra market
        except ImportError:
            yf = None
    if yf is not None:
        try:
            history = yf.Ticker(symbol).history(period="5d", interval="1d")
            if history is not None and not history.empty:
                close = float(history["Close"].dropna().iloc[-1])
                day = str(history.index[-1].date())
                return f"Ticker: {symbol}\nPrice: ${close:,.2f}\nAs of: {day} (daily close, yfinance)"
        except Exception as exc:
            errors.append(f"yfinance: {exc}")
    raise MarketError(f"Market data is unavailable for {symbol}. Last error: {errors[-1] if errors else 'unknown'}")
