import sqlite3

import duckdb
import pytest

from peakwhale_helm.database import Database, DatabaseError


@pytest.fixture()
def demo_db(demo_dir):
    db = Database("duckdb", str(demo_dir / "ledger.duckdb"), max_rows=10)
    yield db
    db.close()


@pytest.fixture()
def sqlite_db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE orders (order_id TEXT, customer TEXT, total REAL)")
    con.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(f"o{i}", f"c{i % 3}", i * 1.5) for i in range(30)])
    con.execute("CREATE VIEW big_orders AS SELECT * FROM orders WHERE total > 20")
    con.commit()
    con.close()
    return Database("sqlite", str(path), max_rows=5)


@pytest.fixture()
def files_db(tmp_path):
    folder = tmp_path / "exports"
    (folder / "2024").mkdir(parents=True)
    (folder / "sales.csv").write_text("region,amount\nnorth,10\nsouth,20\nnorth,5\n")
    (folder / "stock.tsv").write_text("sku\tqty\nA\t3\nB\t0\n")
    (folder / ".hidden.csv").write_text("x\n1\n")
    (folder / "notes.txt").write_text("not a table")
    con = duckdb.connect()
    con.execute(f"COPY (SELECT range AS id, range * 2 AS value FROM range(7)) TO '{folder / '2024' / 'events.parquet'}' (FORMAT parquet)")
    con.close()
    db = Database("files", str(folder), max_rows=50)
    yield db
    db.close()


def test_duckdb_list_describe_query(demo_db):
    tables = {t.name: t for t in demo_db.list_tables()}
    assert set(tables) == {"customers", "transactions"}
    assert tables["customers"].rows == 100
    assert tables["transactions"].rows == 5012
    info, columns, sample = demo_db.describe_table("TRANSACTIONS")
    assert info.name == "transactions"
    assert [c for c, _ in columns] == ["trans_id", "customer_id", "amount", "category", "merchant", "timestamp"]
    assert len(sample.rows) == 3
    result = demo_db.query(
        "SELECT customer_id, COUNT(*) AS n FROM transactions WHERE category = 'Gambling/High Risk' "
        "GROUP BY 1 ORDER BY n DESC, customer_id LIMIT 2"
    )
    assert result.rows == [("2631d00b", 8), ("f81dbaa1", 4)]
    assert result.to_markdown().endswith("Rows: 2")


def test_row_limit_is_enforced(demo_db):
    result = demo_db.query("SELECT * FROM transactions")
    assert len(result.rows) == 10 and result.truncated
    assert "row limit of 10" in result.to_markdown()
    assert len(demo_db.query("SELECT * FROM transactions", max_rows=3).rows) == 3
    # a caller cannot raise the limit above the configured one
    assert len(demo_db.query("SELECT * FROM transactions", max_rows=10_000).rows) == 10


def test_guard_runs_before_the_database(demo_db):
    with pytest.raises(DatabaseError, match="Blocked"):
        demo_db.query("DROP TABLE customers")
    with pytest.raises(DatabaseError, match="Blocked"):
        demo_db.query("SELECT 1; DELETE FROM customers")
    assert demo_db.list_tables()[0].rows == 100  # nothing changed


def test_duckdb_connection_itself_is_read_only_and_closed_to_files(demo_db):
    # Skip the guard on purpose: the engine must still refuse.
    with demo_db._connect() as con:
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE x (i INT)")
        with pytest.raises(duckdb.Error):
            con.execute("SELECT * FROM read_csv('/etc/passwd')")
        with pytest.raises(duckdb.Error):
            con.execute("SET enable_external_access = true")


def test_sql_errors_are_clear(demo_db):
    with pytest.raises(DatabaseError, match="SQL error"):
        demo_db.query("SELECT no_such_column FROM customers")
    with pytest.raises(DatabaseError, match="Unknown table 'nope'. Known tables: customers, transactions"):
        demo_db.describe_table("nope")


def test_missing_database_says_what_to_do(tmp_path):
    db = Database("duckdb", str(tmp_path / "none.duckdb"))
    with pytest.raises(DatabaseError, match="helm-demo"):
        db.list_tables()


def test_timeout_stops_long_queries(demo_dir):
    db = Database("duckdb", str(demo_dir / "ledger.duckdb"), timeout=0.2)
    slow = "SELECT COUNT(*) FROM range(10000000000) a, range(1000) b WHERE a.range * b.range = -1"
    with pytest.raises(DatabaseError, match="time limit"):
        db.query(slow)


def test_sqlite_file(sqlite_db):
    tables = {t.name: t for t in sqlite_db.list_tables()}
    assert tables["orders"].rows == 30 and tables["big_orders"].kind == "view"
    assert sqlite_db.dialect == "sqlite"
    info, columns, _ = sqlite_db.describe_table("orders")
    assert columns == [("order_id", "TEXT"), ("customer", "TEXT"), ("total", "REAL")]
    result = sqlite_db.query("SELECT customer, COUNT(*) AS n FROM orders GROUP BY customer ORDER BY customer")
    assert result.rows == [("c0", 10), ("c1", 10), ("c2", 10)]
    assert sqlite_db.query("SELECT * FROM orders").truncated


def test_sqlite_engine_level_guard(sqlite_db):
    # Even if a statement got past the text guard, SQLite is read-only and the authorizer says no.
    with sqlite_db._connect() as con:
        with pytest.raises(sqlite3.Error):
            con.execute("DELETE FROM orders")
    with pytest.raises(DatabaseError, match="not authorized"):
        sqlite_db.query("WITH x AS (SELECT 1) SELECT * FROM pragma_table_info('orders') JOIN x")
    assert {t.name: t.rows for t in sqlite_db.list_tables()}["orders"] == 30


def test_folder_of_csv_tsv_and_parquet(files_db):
    tables = {t.name: t for t in files_db.list_tables()}
    assert set(tables) == {"sales", "stock", "t_2024_events"}
    assert tables["sales"].kind == "view" and tables["sales"].rows == 3
    assert tables["t_2024_events"].rows == 7
    result = files_db.query("SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY region")
    assert result.rows == [("north", 15), ("south", 20)]
    assert files_db.query("SELECT qty FROM stock WHERE sku = 'B'").rows == [(0,)]
    # files outside the folder stay closed, even without the text guard
    with files_db._connect() as con:
        with pytest.raises(duckdb.Error):
            con.execute("SELECT * FROM read_csv('/etc/passwd')")


def test_folder_picks_up_new_files(files_db, tmp_path):
    (tmp_path / "exports" / "returns.csv").write_text("sku,qty\nA,1\n")
    assert "returns" in {t.name for t in files_db.list_tables()}


def test_single_csv_file(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,age\nAna,30\nBo,41\n")
    db = Database("files", str(path))
    assert [t.name for t in db.list_tables()] == ["people"]
    assert db.query("SELECT MAX(age) AS oldest FROM people").rows == [(41,)]
    db.close()


def test_postgres_is_attached_read_only(monkeypatch):
    """Experimental path: check the statements we send, without a real server."""
    sent = []

    class FakeConnection:
        def execute(self, sql, params=None):
            sent.append(sql)
            return self

        def close(self):
            pass

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: FakeConnection())
    db = Database("postgres", "postgresql://me@localhost/shop")
    db._shared_connection()
    assert sent[:2] == ["INSTALL postgres", "LOAD postgres"]
    assert "READ_ONLY" in sent[2] and "postgresql://me@localhost/shop" in sent[2]
    assert "SET enable_external_access = false" in sent
    assert "SET lock_configuration = true" in sent
