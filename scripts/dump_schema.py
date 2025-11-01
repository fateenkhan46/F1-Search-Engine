# scripts/dump_schema.py
import duckdb, json, os
os.makedirs("artifacts", exist_ok=True)

con = duckdb.connect("f1.duckdb")
tables = [r[0] for r in con.execute("""
  SELECT table_name FROM information_schema.tables
  WHERE table_schema='main' ORDER BY 1
""").fetchall()]

schema = {}
for t in tables:
    cols = con.execute(f"PRAGMA table_info('{t}')").fetchall()
    schema[t] = [{"name": c[1], "type": c[2]} for c in cols]

with open("artifacts/schema.json","w") as f: json.dump(schema, f, indent=2)
print("Wrote artifacts/schema.json")
