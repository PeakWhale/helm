import os
import re
import io
import csv
import time
import duckdb
import yfinance as yf
import requests

from langchain_core.tools import tool
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("⚙️ Loading Peakwhale Helm Tools...")

_VECTOR_CACHE = {}

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _clean_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    t = t.replace("$", "")
    t = re.sub(r"[^A-Z0-9\.\-\^=]", "", t)
    return t


def _to_stooq_symbol(ticker: str) -> str:
    t = _clean_ticker(ticker)
    if not t:
        return ""

    # Stooq uses:
    # US stocks: aapl.us, tsla.us
    # Indices often like ^spx -> spx (but Stooq has its own naming; keep simple for demo)
    if t.startswith("^"):
        t = t[1:]

    if "." in t:
        return t.lower()

    return f"{t.lower()}.us"


def _fetch_stooq_daily_close(ticker: str):
    sym = _to_stooq_symbol(ticker)
    if not sym:
        return None, None, "Invalid ticker format."

    urls = [
        f"https://stooq.com/q/d/l/?s={sym}&i=d",  # Date,Open,High,Low,Close,Volume
        f"https://stooq.com/q/l/?s={sym}&i=d",    # Symbol,Date,Time,Open,High,Low,Close,Volume
    ]

    def looks_like_html(s: str) -> bool:
        h = (s or "").lstrip().lower()
        return h.startswith("<!doctype") or h.startswith("<html") or "<html" in h[:200]

    last_err = None

    for url in urls:
        try:
            r = requests.get(
                url,
                headers={**_HTTP_HEADERS, "Accept": "text/csv,*/*;q=0.8"},
                timeout=10,
                allow_redirects=True,
            )

            if r.status_code != 200:
                last_err = f"HTTP {r.status_code} from Stooq"
                continue

            text = (r.text or "").strip()
            if not text:
                last_err = "Empty response from Stooq"
                continue

            if looks_like_html(text):
                snippet = " ".join(text[:160].split())
                last_err = f"Stooq returned HTML (blocked or redirected). Snippet: {snippet}"
                continue

            reader = csv.DictReader(io.StringIO(text))
            rows = [row for row in reader if row]

            if not rows:
                last_err = "No rows returned from Stooq"
                continue

            row = rows[-1]

            close_key = None
            for k in ["Close", "close", "Zamkniecie", "zamkniecie"]:
                if k in row and (row.get(k) or "").strip():
                    close_key = k
                    break

            date_key = None
            for k in ["Date", "date", "Data", "data"]:
                if k in row and (row.get(k) or "").strip():
                    date_key = k
                    break

            if not close_key:
                last_err = f"Missing Close column in Stooq CSV. Columns: {list(row.keys())}"
                continue

            close_str = (row.get(close_key) or "").strip()
            if close_str.lower() in {"nan", "null"} or close_str == "":
                last_err = "Close value missing in Stooq CSV"
                continue

            price = float(close_str)
            as_of = (row.get(date_key) or "").strip() if date_key else ""

            return price, as_of, None

        except Exception as e:
            last_err = str(e)
            continue

    return None, None, last_err or "Stooq fetch failed"


@tool
def query_ledger(sql_query: str):
    """Executes a SQL query against the internal bank ledger."""
    sanitized_query = sql_query or ""
    try:
        sanitized_query = (
            sanitized_query
            .replace("“", "'")
            .replace("”", "'")
            .replace("’", "'")
        )

        sanitized_query = re.sub(r"COUNT\s*\(\s*\)", "COUNT(*)", sanitized_query, flags=re.IGNORECASE)

        if re.search(r"\bloan_applications\b", sanitized_query, flags=re.IGNORECASE):
            return (
                "SQL Error: Table 'loan_applications' is not available in this ledger database. "
                "Use search_loan_documents to read the PDF loan application."
            )

        con = duckdb.connect("data/ledger.duckdb", read_only=True)
        df = con.execute(sanitized_query).df()
        con.close()

        if df.empty:
            return "Query returned no results."

        return df.to_markdown(index=False)

    except Exception as e:
        return f"SQL Error: {e}\n(Sanitized Query attempted: {sanitized_query})"


@tool
def get_stock_price(ticker: str):
    """
    Fetches a stock price without requiring any API key.
    Primary source is Stooq (daily close).
    Falls back to yfinance if available (may be blocked).
    """
    t = _clean_ticker(ticker)
    if not t:
        return "Error: Missing ticker. Try $AAPL or TSLA."

    # Prefer Stooq (free, no key), usually more reliable than Yahoo via yfinance
    price, as_of, err = _fetch_stooq_daily_close(t)
    if price is not None:
        suffix = f"\nAs of: {as_of} (daily close, Stooq)" if as_of else "\nAs of: latest daily close (Stooq)"
        return f"Ticker: {t}\nPrice: ${price:.2f}{suffix}"

    last_err = err

    # Fallback: yfinance (can be blocked or rate-limited)
    for attempt in range(2):
        try:
            stock = yf.Ticker(t)

            fi = getattr(stock, "fast_info", None)
            if fi:
                for k in [
                    "last_price",
                    "lastPrice",
                    "regularMarketPrice",
                    "last_close",
                    "lastClose",
                    "previous_close",
                    "previousClose",
                ]:
                    try:
                        v = fi.get(k) if hasattr(fi, "get") else None
                    except Exception:
                        v = None
                    if v is not None:
                        return f"Ticker: {t}\nPrice: ${float(v):.2f}\nAs of: recent quote (yfinance fallback)"

            hist = stock.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                close = float(hist["Close"].dropna().iloc[-1])
                return f"Ticker: {t}\nPrice: ${close:.2f}\nAs of: latest daily close (yfinance fallback)"

        except Exception as e:
            last_err = e
            time.sleep(0.4)

    return (
        f"Error: Market data unavailable for {t}. "
        f"Stooq fetch failed, and Yahoo can block yfinance. "
        f"Last error: {last_err}"
    )


try:
    print("   - Loading FinBERT...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception:
    sentiment_pipeline = None


@tool
def analyze_sentiment(text: str):
    """Analyzes financial sentiment using FinBERT."""
    if not sentiment_pipeline:
        return "Error: Sentiment tool unavailable."

    try:
        result = sentiment_pipeline((text or "")[:512])[0]
        label = result.get("label", "UNKNOWN")
        score = result.get("score", 0.0)
        return f"Sentiment: {label} ({score:.2f})"
    except Exception as e:
        return f"Error: {e}"


print("   - Loading Embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    embeddings = None


@tool
def search_loan_documents(customer_id: str, query: str) -> str:
    """Retrieves info from a customer's PDF loan application."""
    if not embeddings:
        return "Error: Embeddings model not loaded."

    cid = (customer_id or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{8}", cid):
        return "Error: Invalid customer id. Expected 8 hex characters, like 2631d00b."

    pdf_path = f"data/vault/{cid}_loan_app.pdf"
    if not os.path.exists(pdf_path):
        return f"Error: Could not find document for {cid} at {pdf_path}"

    try:
        if cid not in _VECTOR_CACHE:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            _VECTOR_CACHE[cid] = FAISS.from_documents(pages, embeddings)

        vectorstore = _VECTOR_CACHE[cid]

        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(query or "income source")

        if not docs:
            return f"Document Excerpt for {cid}:\n(No relevant passages found for query: {query})"

        context = "\n".join([d.page_content for d in docs]).strip()
        context = context[:1200]

        return f"Document Excerpt for {cid}:\n{context}"

    except Exception as e:
        return f"Error: Failed to search document for {cid}. Details: {e}"
