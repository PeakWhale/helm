# tools.py
"""
MCP Tool Server (stdio)

Important:
For stdio based MCP servers, do not print to stdout. Use logging (stderr) instead.
"""

import os
import re
import io
import csv
import time
import sys
import logging
from typing import Optional, Tuple


from mcp.server.fastmcp import FastMCP


logger = logging.getLogger("helm.mcp.tools")
logging.basicConfig(
    level=os.environ.get("HELM_LOG_LEVEL", "INFO"),
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

mcp = FastMCP("helm_tools")

_VECTOR_CACHE = {}

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
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

    if t.startswith("^"):
        t = t[1:]

    if "." in t:
        return t.lower()

    return f"{t.lower()}.us"


def _fetch_stooq_daily_close(ticker: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    sym = _to_stooq_symbol(ticker)
    if not sym:
        return None, None, "Invalid ticker format."

    urls = [
        f"https://stooq.com/q/d/l/?s={sym}&i=d",
        f"https://stooq.com/q/l/?s={sym}&i=d",
    ]

    def looks_like_html(s: str) -> bool:
        h = (s or "").lstrip().lower()
        return h.startswith("<!doctype") or h.startswith("<html") or "<html" in h[:200]

    last_err = None

    for url in urls:
        try:
            import requests
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
                last_err = f"Stooq returned HTML. Snippet: {snippet}"
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


_SENTIMENT_PIPELINE = None
_EMBEDDINGS = None


import subprocess
import threading
import queue
import json

class SentimentClient:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self._start_process()

    def _start_process(self):
        script_path = os.path.join(os.path.dirname(__file__), "sentiment_runner.py")
        if not os.path.exists(script_path):
            script_path = "src/sentiment_runner.py"

        try:
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            # Wait for ready signal with a 60s timeout
            if self.process and self.process.stdout:
                # We use a simple polling approach for the ready signal
                start_t = time.time()
                line = ""
                while time.time() - start_t < 60:
                    # Check if process died
                    if self.process.poll() is not None:
                        break
                    
                    # Try to read line (this is still slightly tricky with non-blocking)
                    # For simplicity on Mac/Unix, we can use a small sleep and peek
                    # But for now, let's just use the fact it's a separate process
                    # and we don't strictly HAVE to wait here if we don't want to block server start.
                    # HOWEVER, user wants "upfront", so let's use a thread to read it.
                    
                    line = self.process.stdout.readline()
                    if line:
                        break
                    time.sleep(0.1)

                if "ready" not in line:
                    logger.error("Sentiment runner failed to signal readiness: %s", line or "Timeout")
                    # We don't kill it yet, maybe it's just slow, but we log the issue
        except Exception as e:
            logger.error("Failed to start sentiment subprocess: %s", e)
            self.process = None

    def analyze(self, text: str) -> str:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                logger.info("Restarting sentiment subprocess...")
                self._start_process()
                if self.process is None:
                    return "Error: Sentiment subprocess unavailable."

            try:
                # Send request
                req = json.dumps({"text": text or ""}) + "\n"
                self.process.stdin.write(req)
                self.process.stdin.flush()
                
                # Read response
                resp_line = self.process.stdout.readline()
                if not resp_line:
                    return "Error: Subprocess output empty (crashed?)"
                
                data = json.loads(resp_line)
                if "error" in data:
                    return f"Error: {data['error']}"
                    
                label = data.get("label", "UNKNOWN")
                score = data.get("score", 0.0)
                return f"Sentiment: {label} ({score:.2f})"
                
            except Exception as e:
                logger.error("Sentiment RPC error: %s", e)
                # Force restart on next call
                if self.process:
                    try:
                        self.process.kill()
                    except:
                        pass
                self.process = None
                return f"Error: RPC failed ({e})"

# Initialize upfront (in background to not block import entirely, 
# or strictly upfront if user wants 'wait for load')
# The user asked for "upfront", so let's let it start now.
_SENTIMENT_CLIENT = SentimentClient()



def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS:
        return _EMBEDDINGS
    
    logger.info("Lazy loading embeddings model...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return _EMBEDDINGS
    except Exception as e:
        logger.warning("Embeddings load failed: %s", e)
        return None


@mcp.tool()
def query_ledger(sql_query: str) -> str:
    """
    Executes a Structured Query Language (SQL) query against the internal bank ledger (DuckDB).
    Returns a markdown table.
    """
    sanitized_query = sql_query or ""
    try:
        sanitized_query = (
            sanitized_query
            .replace("“", "'")
            .replace("”", "'")
            .replace("’", "'")
        )

        sanitized_query = re.sub(
            r"COUNT\s*\(\s*\)",
            "COUNT(*)",
            sanitized_query,
            flags=re.IGNORECASE,
        )

        if re.search(r"\bloan_applications\b", sanitized_query, flags=re.IGNORECASE):
            return (
                "SQL Error: Table 'loan_applications' is not available in this ledger database. "
                "Use search_loan_documents to read the loan application PDF."
            )

        import duckdb
        con = duckdb.connect("data/ledger.duckdb", read_only=True)
        df = con.execute(sanitized_query).df()
        con.close()

        if df.empty:
            return "Query returned no results."

        return df.to_markdown(index=False)

    except Exception as e:
        return f"SQL Error: {e}\n(Sanitized Query attempted: {sanitized_query})"


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """
    Fetches a stock price without requiring any API key.
    Primary source is Stooq daily close, falls back to yfinance.
    """
    t = _clean_ticker(ticker)
    if not t:
        return "Error: Missing ticker. Try $AAPL or TSLA."

    price, as_of, err = _fetch_stooq_daily_close(t)
    if price is not None:
        suffix = f"\nAs of: {as_of} (daily close, Stooq)" if as_of else "\nAs of: latest daily close (Stooq)"
        return f"Ticker: {t}\nPrice: ${price:.2f}{suffix}"

    last_err = err

    for attempt in range(2):
        try:
            import yfinance as yf
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


@mcp.tool()
def analyze_sentiment(text: str) -> str:
    """
    Analyzes financial sentiment using FinBERT.
    Uses a persistent background process for low-latency inference.
    """
    # Simply delegate to the persistent client
    return _SENTIMENT_CLIENT.analyze(text)


@mcp.tool()
def search_loan_documents(customer_id: str, query: str) -> str:
    """
    Retrieves information from a customer's Portable Document Format (PDF) loan application
    using embeddings plus FAISS vector search.
    """
    emb_model = _get_embeddings()
    if not emb_model:
        return "Error: Embeddings model not loaded."

    cid = (customer_id or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{8}", cid):
        return "Error: Invalid customer id. Expected 8 hex characters, like 2631d00b."

    pdf_path = f"data/vault/{cid}_loan_app.pdf"
    if not os.path.exists(pdf_path):
        return f"Error: Could not find document for {cid} at {pdf_path}"

    try:
        sys.stderr.write(f"[Researcher] Processing loan document for {cid}...\n")
        
        if cid not in _VECTOR_CACHE:
            sys.stderr.write("[Researcher] Loading PDF and creating vector index... (this may take 5-10s)\n")
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_community.vectorstores import FAISS
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            _VECTOR_CACHE[cid] = FAISS.from_documents(pages, emb_model)
            sys.stderr.write("[Researcher] Vector index created.\n")

        sys.stderr.write("[Researcher] Querying vector store...\n")
        vectorstore = _VECTOR_CACHE[cid]
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(query or "income source")

        if not docs:
            return f"Document Excerpt for {cid}:\n(No relevant passages found for query: {query})"

        context = "\n".join([d.page_content for d in docs]).strip()
        context = context[:1200]
        
        sys.stderr.write("[Researcher] Document search complete.\n")
        return f"Document Excerpt for {cid}:\n{context}"

    except Exception as e:
        return f"Error: Failed to search document for {cid}. Details: {e}"


def main():
    # Also trigger embeddings load upfront for visibility (only when running as server)
    logger.info("Initializing Embeddings model...")
    if _get_embeddings():
        logger.info("Embeddings model loaded.")

    # MCP server runs over stdio by default for local host subprocess usage
    mcp.run(transport=os.environ.get("HELM_MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()
