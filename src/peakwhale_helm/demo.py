"""Build the synthetic demo data: a small bank ledger and a few documents.

Everything is fake and made from a fixed random seed, so the data is the same
on every machine. The evals in ``evals/demo_cases.json`` depend on that.

Run: ``uv run helm-demo`` (writes ``data/ledger.duckdb`` and ``data/vault/``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import SavedQuery

SEED = 42

FIRST_NAMES = [
    "Ava", "Ben", "Chloe", "Daniel", "Elena", "Farah", "George", "Hana", "Ivan", "Julia",
    "Kofi", "Lena", "Mateo", "Nora", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara",
    "Umar", "Vera", "Wei", "Ximena", "Yusuf", "Zoe",
]
LAST_NAMES = [
    "Adams", "Brooks", "Chen", "Diaz", "Evans", "Fischer", "Garcia", "Hughes", "Ito", "Jensen",
    "Khan", "Lopez", "Morris", "Novak", "Okafor", "Patel", "Quinn", "Reyes", "Silva", "Tanaka",
    "Usman", "Varga", "Walsh", "Young", "Zhang",
]
SEGMENTS = ["Retail", "Wealth", "SME", "Corporate"]
MERCHANTS = ["DemoMart", "DemoGrocer", "DemoElectronics", "DemoAir", "DemoUtilities", "DemoTravelCo"]
CATEGORIES = {
    "Groceries": (-200, -50),
    "Tech/Gadgets": (-2000, -100),
    "Travel": (-3000, -500),
    "Utilities": (-300, -100),
    "Salary": (3000, 9000),
    "Wire Transfer": (-5000, 5000),
}
HIGH_RISK = "Gambling/High Risk"

# Two fixed customers the demo story is about.
DEMO_CUSTOMERS = [
    ("2631d00b", "Demo Tiffany Miller", 420),
    ("f81dbaa1", "Demo Marcus Sanders", 480),
]
FORCED_HIGH_RISK = {"2631d00b": 8, "f81dbaa1": 4}
LOAN_APPLICATIONS = {
    "2631d00b": ("Freelance Consulting", 120_000),
    "f81dbaa1": ("W2 Employment", 95_000),
}

# The same saved query is in helm.toml. A test checks that both stay equal.
DEMO_SAVED_QUERIES = (
    SavedQuery(
        name="top_high_risk_customers",
        description="Customers ranked by their number of Gambling/High Risk transactions",
        keywords=("top high risk", "most high risk", "most gambling", "high risk customer", "high risk transactions"),
        sql=(
            "SELECT customer_id, COUNT(*) AS high_risk_count FROM transactions "
            "WHERE category = 'Gambling/High Risk' GROUP BY customer_id "
            "ORDER BY high_risk_count DESC, customer_id"
        ),
    ),
)

POLICY_MD = """\
# PeakWhale Bank Credit Policy (Synthetic Demo)

All content in this file is fictional. It exists only to show how Helm searches documents.

## High risk spending

Gambling and other high risk transactions are a warning sign. A customer with more than 5 transactions in the "Gambling/High Risk" category in the last 12 months needs a manual review by a credit officer before any new loan is approved.

## Income checks

The income source stated on a loan application must match the deposits in the ledger. Freelance or self-employed income needs two years of tax returns.

## Speculative trading

Applicants must declare whether they engage in high risk speculative trading. A false declaration is a reason to decline the loan.
"""


@dataclass(frozen=True)
class DemoPaths:
    ledger: Path
    vault: Path


def _customers(rng: random.Random, n: int) -> list[tuple]:
    start = dt.date(2020, 1, 1)
    rows = []
    for cid, name, risk in DEMO_CUSTOMERS:
        rows.append((cid, name, f"{cid}@example.com", risk, "Retail", start + dt.timedelta(days=rng.randrange(1800))))
    used = {r[0] for r in rows}
    while len(rows) < n:
        cid = f"{rng.getrandbits(32):08x}"
        if cid in used:
            continue
        used.add(cid)
        name = f"Demo {rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        rows.append(
            (cid, name, f"{cid}@example.com", rng.randint(300, 850), rng.choice(SEGMENTS),
             start + dt.timedelta(days=rng.randrange(1800)))
        )
    return rows


def _transactions(rng: random.Random, customers: list[tuple], per_customer: int) -> list[tuple]:
    start = dt.datetime(2025, 1, 1)
    year = 365 * 24 * 3600

    def when() -> dt.datetime:
        return start + dt.timedelta(seconds=rng.randrange(year))

    rows = []
    for cid, _name, _email, risk, _segment, _joined in customers:
        background_high_risk = 0
        for _ in range(per_customer):
            category = rng.choice(list(CATEGORIES))
            low, high = CATEGORIES[category]
            amount = round(rng.uniform(low, high), 2)
            merchant = "PeakWhale Demo Payroll" if amount > 0 else rng.choice(MERCHANTS)
            if cid not in FORCED_HIGH_RISK and risk < 500 and rng.random() < 0.02 and background_high_risk < 2:
                category, amount, merchant = HIGH_RISK, -9999.99, "DemoCasino"
                background_high_risk += 1
            rows.append((f"{rng.getrandbits(48):012x}", cid, amount, category, merchant, when()))
    for cid, count in FORCED_HIGH_RISK.items():
        for _ in range(count):
            rows.append((f"{rng.getrandbits(48):012x}", cid, -9999.99, HIGH_RISK, "DemoCasino", when()))
    return rows


def build_ledger(path: Path, seed: int = SEED, customers: int = 100, per_customer: int = 50) -> None:
    import duckdb

    rng = random.Random(seed)
    customer_rows = _customers(rng, customers)
    transaction_rows = _transactions(rng, customer_rows, per_customer)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        tables = {
            "customers": (
                ["customer_id", "name", "email", "risk_score", "segment", "joined_date"],
                ["VARCHAR", "VARCHAR", "VARCHAR", "INTEGER", "VARCHAR", "DATE"],
                customer_rows,
            ),
            "transactions": (
                ["trans_id", "customer_id", "amount", "category", "merchant", "timestamp"],
                ["VARCHAR", "VARCHAR", "DOUBLE", "VARCHAR", "VARCHAR", "TIMESTAMP"],
                transaction_rows,
            ),
        }
        con = duckdb.connect(str(path))
        try:
            for table, (columns, types, rows) in tables.items():
                csv_path = Path(tmp) / f"{table}.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    writer.writerows(rows)
                spec = "{" + ", ".join(f"'{c}': '{t}'" for c, t in zip(columns, types)) + "}"
                con.execute(
                    f"CREATE TABLE {table} AS SELECT * FROM read_csv(?, header = true, columns = {spec})",
                    [str(csv_path)],
                )
        finally:
            con.close()


def write_loan_application(path: Path, customer_id: str, name: str, income_source: str, annual_income: int) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    _width, height = letter
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


def build_documents(vault: Path) -> list[Path]:
    vault.mkdir(parents=True, exist_ok=True)
    for old in vault.glob("*_loan_app.pdf"):
        old.unlink()
    names = {cid: name for cid, name, _risk in DEMO_CUSTOMERS}
    written = []
    for cid, (source, income) in LOAN_APPLICATIONS.items():
        path = vault / f"{cid}_loan_app.pdf"
        write_loan_application(path, cid, names[cid], source, income)
        written.append(path)
    policy = vault / "credit_policy.md"
    policy.write_text(POLICY_MD, encoding="utf-8")
    written.append(policy)
    return written


def generate(out_dir: str | Path = "data") -> DemoPaths:
    out = Path(out_dir)
    paths = DemoPaths(ledger=out / "ledger.duckdb", vault=out / "vault")
    build_ledger(paths.ledger)
    build_documents(paths.vault)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="helm-demo", description="Create the synthetic demo data")
    parser.add_argument("--out", default="data", help="Output folder (default: data)")
    args = parser.parse_args(argv)
    paths = generate(args.out)
    print("Helm demo data is ready (synthetic, for demos only):")
    print(f"  ledger : {paths.ledger}")
    print(f"  docs   : {paths.vault}")


if __name__ == "__main__":
    main()
