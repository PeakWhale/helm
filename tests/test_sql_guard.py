import duckdb
import pytest

from peakwhale_helm.sql_guard import SQLGuardError, check_sql, limit_rows

ALLOWED = [
    "SELECT 1",
    "select * from customers",
    "SELECT * FROM customers;",
    "SELECT * FROM customers ;  ;",
    "  WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "(SELECT 1)",
    "SELECT 'drop table customers; --' AS text_is_fine",
    'SELECT "select" FROM "weird table"',
    "SELECT name FROM customers -- a comment at the end",
    "SELECT /* a comment */ name FROM customers",
    "SELECT replace(name, 'Demo ', '') FROM customers",
    "SELECT * FROM t WHERE x IN ('a', 'b')",
    "SELECT 'it''s fine' AS quote_inside",
    "SELECT 'costs $5 or $$5' AS dollars_in_text",
    'SELECT "a$b" FROM t',
    "SELECT E'no backslash here' AS e_text",
    "SELECT 'C:\\temp\\x' AS backslash_in_plain_text",
    "SELECT * FROM main.customers",
    "SELECT * FROM t JOIN u ON t.a = u.a, v",
    "SELECT * FROM (DESCRIBE customers)",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(sql):
    cleaned = check_sql(sql)
    assert cleaned
    assert not cleaned.rstrip().endswith(";")


BLOCKED = {
    # not a SELECT
    "DROP TABLE customers": "SELECT or WITH",
    "drop table customers;": "SELECT or WITH",
    "DELETE FROM customers": "SELECT or WITH",
    "INSERT INTO customers VALUES (1)": "SELECT or WITH",
    "UPDATE customers SET name = 'x'": "SELECT or WITH",
    "CREATE TABLE x AS SELECT 1": "SELECT or WITH",
    "ATTACH 'other.db' AS other": "SELECT or WITH",
    "DETACH other": "SELECT or WITH",
    "COPY customers TO '/tmp/out.csv'": "SELECT or WITH",
    "INSTALL httpfs": "SELECT or WITH",
    "LOAD httpfs": "SELECT or WITH",
    "PRAGMA database_list": "SELECT or WITH",
    "SET enable_external_access = true": "SELECT or WITH",
    "CALL pragma_version()": "SELECT or WITH",
    "EXPORT DATABASE '/tmp/x'": "SELECT or WITH",
    "SHOW TABLES": "SELECT or WITH",
    "( DROP TABLE x )": "SELECT or WITH",
    # several statements
    "SELECT 1; DROP TABLE customers": "one statement",
    "SELECT 1;DROP TABLE customers;": "one statement",
    "SELECT 1; SELECT 2": "one statement",
    # comments hiding statements
    "SELECT 1 /* harmless */; DROP TABLE customers": "one statement",
    "SELECT 1 -- note\n; DROP TABLE customers": "one statement",
    "/* SELECT */ DROP TABLE customers": "SELECT or WITH",
    "-- SELECT\nDROP TABLE customers": "SELECT or WITH",
    "SELECT 1 /* never closed": "not closed",
    "SELECT 1 /* /* nested */ ; DROP TABLE x */": "one statement",
    # write keywords inside a SELECT / WITH
    "WITH d AS (DELETE FROM customers RETURNING *) SELECT * FROM d": "DELETE",
    "SELECT * FROM customers WHERE 1 = 1 UNION SELECT * FROM (INSERT INTO x VALUES (1))": "INSERT",
    "WITH x AS (SELECT 1) ATTACH 'a.db'": "ATTACH",
    "SELECT 1 FROM t WHERE a = (COPY t TO 'x')": "COPY",
    # file, network and SQL-in-a-string functions
    "SELECT * FROM read_csv('/etc/passwd')": "read_csv",
    "SELECT * FROM READ_CSV_AUTO('/etc/passwd')": "read_csv_auto",
    "SELECT * FROM read_parquet('s3://bucket/x.parquet')": "read_parquet",
    "SELECT * FROM glob('/home/*')": "glob",
    "SELECT * FROM read_text('/etc/hostname')": "read_text",
    'SELECT * FROM "read_csv"(\'/etc/passwd\')': "read_csv",
    "SELECT * FROM main.read_json('x.json')": "read_json",
    "SELECT * FROM query('DROP TABLE customers')": "query",
    "SELECT load_extension('evil.so')": "load_extension",
    "SELECT pg_read_file('/etc/passwd')": "pg_read_file",
    "SELECT * FROM sqlite_scan('other.db', 't')": "sqlite_scan",
    # replacement scans: a file name used as a table
    "SELECT * FROM '/etc/passwd.csv'": "FROM '...'",
    "SELECT * FROM customers JOIN 'secret.parquet' USING (id)": "FROM '...'",
    "SELECT * FROM customers, 'secret.csv'": "FROM '...'",
    "SELECT * FROM t JOIN u ON true, 'secret.csv'": "FROM '...'",
    "SELECT * FROM ONLY 'secret.csv'": "FROM '...'",
    "SELECT * FROM (SUMMARIZE 'secret.csv')": "FROM '...'",
    "SELECT * FROM (PIVOT 'secret.csv' ON a USING sum(b))": "FROM '...'",
    'SELECT * FROM "secret.csv"': "file name",
    'SELECT * FROM ".tmp/leak.csv"': "file name",
    "SELECT * FROM secret.csv": "file name",
    'SELECT * FROM "secret".csv': "file name",
    'SELECT * FROM main."secret.csv"': "file name",
    "SELECT * FROM secret.csv.gz": "file name",
    'SELECT * FROM t JOIN "/tmp/a.parquet" USING (id)': "file name",
    'SELECT * FROM t, "C:\\data\\x"': "file name",
    'SELECT * FROM (TABLE "secret.csv")': "file name",
    'SELECT * FROM "https://example.com/data"': "file name",
    # DuckDB reads $$...$$, $tag$...$tag$ and E'\'' as text. If the guard did not know,
    # a ' inside them would make it treat real code as text (found in a review).
    "SELECT $$'$$ AS a, (SELECT content FROM read_text('data/.env')) AS leaked, $$'$$ AS c": "$ sign",
    "SELECT $$'$$ AS a) AS x; COPY (SELECT 1 AS id) TO 'sales.csv' (HEADER); "
    "SELECT * FROM (SELECT $$'$$ AS b": "$ sign",
    "SELECT $t$'$t$ AS q, x.* FROM query('SELECT 42') x --'": "$ sign",
    "SELECT $$'$$ AS q, x.* FROM duckdb_secrets() x --'": "$ sign",
    "SELECT 1 AS a$$'$$, x.* FROM read_csv('/etc/passwd') x --'": "$ sign",
    "SELECT $1": "$ sign",
    "SELECT E'\\'' AS a, (SELECT content FROM read_text('/etc/hostname')) AS b --'": "backslash",
    "SELECT 1e'\\'' AS a, x.* FROM query('SELECT 42') x --'": "backslash",
    # DuckDB ends a -- comment at \r too
    "SELECT 1 --\r; DROP TABLE customers": "one statement",
    # broken input
    "": "empty",
    "   ;  ": "empty",
    "SELECT 'not closed": "not closed",
    "SELECT 1\x00": "NUL",
}


@pytest.mark.parametrize("sql, reason", BLOCKED.items())
def test_blocked(sql, reason):
    with pytest.raises(SQLGuardError) as info:
        check_sql(sql)
    assert reason.lower() in str(info.value).lower()
    assert str(info.value).startswith("Blocked:")


def test_column_named_like_a_keyword_can_be_quoted():
    with pytest.raises(SQLGuardError, match='double quotes'):
        check_sql("SELECT load FROM servers")
    assert check_sql('SELECT "load" FROM servers') == 'SELECT "load" FROM servers'


def test_smart_quotes_are_made_plain():
    assert check_sql("SELECT * FROM t WHERE a = ‘x’") == "SELECT * FROM t WHERE a = 'x'"
    # double smart quotes are name quotes, so a quoted column stays a column
    assert check_sql("SELECT “category” FROM t") == 'SELECT "category" FROM t'


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT $$'$$ AS a, x.* FROM query('SELECT 42') x --'",
        "SELECT E'\\'' AS a, x.* FROM query('SELECT 42') x --'",
    ],
)
def test_blocked_tricks_really_run_in_duckdb(sql):
    # Proof that these are real attacks, not just odd text: DuckDB itself runs them.
    duckdb.connect().execute(sql).fetchall()
    with pytest.raises(SQLGuardError):
        check_sql(sql)


def test_comments_are_removed_from_the_cleaned_sql():
    assert check_sql("SELECT 1 -- trailing note") == "SELECT 1"
    assert check_sql("SELECT /* c */ 1;") == "SELECT   1"
    # a -- comment ends at \r as well, like in DuckDB
    assert check_sql("SELECT 1 -- c\r, 2").split() == ["SELECT", "1", ",", "2"]


def test_limit_rows_wraps_query_and_keeps_order():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT range AS i FROM range(100)")
    sql = limit_rows(check_sql("SELECT i FROM t ORDER BY i DESC -- newest first"), 5)
    rows = [r[0] for r in con.execute(sql).fetchall()]
    assert rows == [99, 98, 97, 96, 95, 94]  # max_rows + 1 so the caller can see it was cut


def test_limit_rows_rejects_bad_limit():
    with pytest.raises(ValueError):
        limit_rows("SELECT 1", 0)
