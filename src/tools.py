import os
import re
import duckdb
import yfinance as yf
from langchain_core.tools import tool
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("⚙️ Loading Peakwhale Helm Tools...")

_VECTOR_CACHE = {}


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
    """Fetches the latest stock price."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("currentPrice") or stock.info.get("previousClose")
        return f"Ticker: {ticker.upper()}\nPrice: ${price}"
    except Exception as e:
        return f"Error: {e}"


try:
    print("   - Loading FinBERT...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception:
    sentiment_pipeline = None


@tool
def analyze_sentiment(text: str):
    """Analyzes financial sentiment."""
    if not sentiment_pipeline:
        return "Sentiment tool unavailable."
    try:
        result = sentiment_pipeline((text or "")[:512])[0]
        return f"Sentiment: {result['label']} ({result['score']:.2f})"
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

    pdf_path = f"data/vault/{customer_id}_loan_app.pdf"
    if not os.path.exists(pdf_path):
        return f"Error: Could not find document for {customer_id} at {pdf_path}"

    try:
        if customer_id not in _VECTOR_CACHE:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            _VECTOR_CACHE[customer_id] = FAISS.from_documents(pages, embeddings)

        vectorstore = _VECTOR_CACHE[customer_id]

        # k=1 keeps the excerpt smaller and faster for demo purposes
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(query)

        if not docs:
            return f"Document Excerpt for {customer_id}:\n(No relevant passages found for query: {query})"

        context = "\n".join([d.page_content for d in docs]).strip()
        context = context[:1200]

        return f"Document Excerpt for {customer_id}:\n{context}"

    except Exception as e:
        return f"Error: Failed to search document for {customer_id}. Details: {e}"
