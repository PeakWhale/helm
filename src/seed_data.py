import duckdb
import pandas as pd
import random
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

CONN_PATH = "data/ledger.duckdb"

print("🌊 Peakwhale Helm: Generating synthetic banking data (demo only)")

DEMO_CUSTOMERS = [
    {"customer_id": "2631d00b", "name": "Demo Tiffany Miller", "risk_score": 420},
    {"customer_id": "f81dbaa1", "name": "Demo Marcus Sanders", "risk_score": 480},
]

FORCED_HIGH_RISK_COUNTS = {
    "2631d00b": 8,
    "f81dbaa1": 4,
}

SEGMENTS = ["Retail", "Wealth", "SME", "Corporate"]

DEMO_MERCHANTS = [
    "DemoMart",
    "DemoGrocer",
    "DemoElectronics",
    "DemoAir",
    "DemoUtilities",
    "DemoCasino",
    "DemoTravelCo",
]

def generate_customers(n=100):
    print(f"   - Minting {n} customer profiles")
    customers = []

    for fixed in DEMO_CUSTOMERS:
        customers.append(
            {
                "customer_id": fixed["customer_id"],
                "name": fixed["name"],
                "email": f"{fixed['customer_id']}@example.com",
                "risk_score": fixed["risk_score"],
                "segment": "Retail",
                "joined_date": fake.date_between(start_date="-5y", end_date="today"),
            }
        )

    existing_ids = {c["customer_id"] for c in customers}

    while len(customers) < n:
        cust_id = fake.uuid4()[:8]
        if cust_id in existing_ids:
            continue
        existing_ids.add(cust_id)

        first = fake.first_name()
        last = fake.last_name()
        name = f"Demo {first} {last}"

        customers.append(
            {
                "customer_id": cust_id,
                "name": name,
                "email": f"{cust_id}@example.com",
                "risk_score": random.randint(300, 850),
                "segment": random.choice(SEGMENTS),
                "joined_date": fake.date_between(start_date="-5y", end_date="today"),
            }
        )

    return pd.DataFrame(customers)

def generate_transactions(customers_df, n_per_cust=50):
    print(f"   - Simulating {len(customers_df) * n_per_cust} transactions")
    transactions = []

    categories = {
        "Groceries": (-200, -50),
        "Tech/Gadgets": (-2000, -100),
        "Travel": (-3000, -500),
        "Utilities": (-300, -100),
        "Salary": (3000, 9000),
        "Wire Transfer": (-5000, 5000),
    }

    background_high_risk_cap = 2

    for _, customer in customers_df.iterrows():
        cust_id = customer["customer_id"]
        risk = customer["risk_score"]

        background_high_risk_count = 0

        for _ in range(n_per_cust):
            cat = random.choice(list(categories.keys()))
            min_amt, max_amt = categories[cat]
            amount = round(random.uniform(min_amt, max_amt), 2)

            merchant = "Peakwhale Demo Payroll" if amount > 0 else random.choice(DEMO_MERCHANTS)

            if (
                cust_id not in FORCED_HIGH_RISK_COUNTS
                and risk < 500
                and random.random() < 0.02
                and background_high_risk_count < background_high_risk_cap
            ):
                amount = -9999.99
                cat = "Gambling/High Risk"
                merchant = "DemoCasino"
                background_high_risk_count += 1

            transactions.append(
                {
                    "trans_id": fake.uuid4()[:12],
                    "customer_id": cust_id,
                    "amount": amount,
                    "category": cat,
                    "merchant": merchant,
                    "timestamp": fake.date_time_between(start_date="-1y", end_date="now"),
                }
            )

    for cust_id, k in FORCED_HIGH_RISK_COUNTS.items():
        for _ in range(k):
            transactions.append(
                {
                    "trans_id": fake.uuid4()[:12],
                    "customer_id": cust_id,
                    "amount": -9999.99,
                    "category": "Gambling/High Risk",
                    "merchant": "DemoCasino",
                    "timestamp": fake.date_time_between(start_date="-1y", end_date="now"),
                }
            )

    return pd.DataFrame(transactions)

try:
    df_cust = generate_customers(n=100)
    df_trans = generate_transactions(df_cust, n_per_cust=50)

    con = duckdb.connect(CONN_PATH)
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM df_cust")
    con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM df_trans")

    top = con.execute(
        "SELECT customer_id, COUNT(*) AS high_risk_count "
        "FROM transactions "
        "WHERE category = 'Gambling/High Risk' "
        "GROUP BY customer_id "
        "ORDER BY 2 DESC "
        "LIMIT 5"
    ).df()

    count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"✅ Success. Ledger written to {CONN_PATH}. Total transactions: {count}")

    print("\n[Preview: Top high risk customers]")
    print(top.to_markdown(index=False))

    con.close()

except Exception as e:
    print(f"❌ Error seeding data: {e}")
