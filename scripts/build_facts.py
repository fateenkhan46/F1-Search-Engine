# scripts/build_facts.py
import duckdb
from pathlib import Path

DATA_DIR = Path("data")

def main():
    if not DATA_DIR.exists():
        raise SystemExit("data/ folder not found. Create it and put CSVs inside.")

    con = duckdb.connect("f1.duckdb")
    con.execute("PRAGMA threads=4;")

    # Ingest every .csv in data/, table name = filename (lowercase, no extension)
    for p in sorted(DATA_DIR.glob("*.csv")):
        table = p.stem.lower()  # e.g., results.csv -> results
        print(f"Ingesting {p.name} -> table {table}")
        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv_auto('{p.as_posix()}', header=True);
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  rows: {n}")

    # Helpful views for simple name joins
    con.execute("""
        CREATE OR REPLACE VIEW driver_name AS
        SELECT driverId, (forename || ' ' || surname) AS name FROM drivers;
    """)
    con.execute("""
        CREATE OR REPLACE VIEW constructor_name AS
        SELECT constructorId, name FROM constructors;
    """)
    print("Created views: driver_name, constructor_name")

    con.close()
    print("✅ Built f1.duckdb")

if __name__ == "__main__":
    main()
