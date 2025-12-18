import os
import glob
import duckdb
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CONN_PATH = "data/ledger.duckdb"
VAULT_DIR = "data/vault"

def create_loan_application(customer_id, name, income_source, annual_income, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 40, "SYNTHETIC DEMO DOCUMENT")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 58, "All contents are fictional and generated for demonstration purposes only.")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 95, "PEAKWHALE BANK LOAN APPLICATION")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 135, f"Application ID: DEMO APP {customer_id}")
    c.drawString(50, height - 155, f"Customer ID: {customer_id}")
    c.drawString(50, height - 175, f"Applicant Name: {name}")

    c.drawString(50, height - 215, f"Stated Income Source: {income_source}")
    c.drawString(50, height - 235, f"Annual Income: ${annual_income:,.0f}")

    c.setFont("Helvetica-Oblique", 11)
    c.drawString(50, height - 285, "Applicant declares that they do NOT engage in high risk speculative trading.")

    c.save()
    print(f"✅ Generated loan application PDF for {name} ({customer_id}) at {filename}")

def get_top_high_risk_customers(con, limit=2):
    return con.execute(
        "SELECT customer_id, COUNT(*) AS high_risk_count "
        "FROM transactions "
        "WHERE category = 'Gambling/High Risk' "
        "GROUP BY customer_id "
        "ORDER BY 2 DESC "
        f"LIMIT {limit}"
    ).fetchall()

def get_customer_name(con, customer_id):
    row = con.execute(
        "SELECT name FROM customers WHERE customer_id = ? LIMIT 1",
        [customer_id],
    ).fetchone()
    return row[0] if row else f"Demo Customer {customer_id}"

if __name__ == "__main__":
    os.makedirs(VAULT_DIR, exist_ok=True)

    for path in glob.glob(os.path.join(VAULT_DIR, "*_loan_app.pdf")):
        os.remove(path)

    con = duckdb.connect(CONN_PATH, read_only=True)

    top = get_top_high_risk_customers(con, limit=2)
    if not top:
        print("❌ No high risk customers found. Seed ledger first.")
        raise SystemExit(1)

    for customer_id, count in top:
        name = get_customer_name(con, customer_id)

        if customer_id == "2631d00b":
            income_source = "Freelance Consulting"
            annual_income = 120000
        else:
            income_source = "W2 Employment"
            annual_income = 95000

        pdf_path = os.path.join(VAULT_DIR, f"{customer_id}_loan_app.pdf")
        create_loan_application(customer_id, name, income_source, annual_income, pdf_path)

    con.close()
