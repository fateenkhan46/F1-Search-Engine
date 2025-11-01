# backend/sql_agent.py
import os, json, re, duckdb
from dotenv import load_dotenv
load_dotenv()

# --- LLM client (Gemini) ---
from google import genai
from google.genai import types
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- load schema ---
with open("artifacts/schema.json") as f:
    SCHEMA = json.load(f)

# -------------------- Prompting --------------------
SYSTEM_RULES = """You are a SQL generator for DuckDB over Formula 1 tables.

RULES
1) Respect every explicit filter in the question (driver/constructor, year(s), race(s), session).
2) Do NOT assume "latest" unless the question explicitly says latest/recent/current.
3) Prefer detailed answers that match the user's wording:
   - If they ask "by race", "per round", "timeline", "per circuit/track", "by season", return multiple rows with a clear ORDER BY.
   - If they ask "summary/total/overall", return aggregates.
4) Safe SQL only: a single SELECT/WITH statement. No writes/DDL. No semicolon chaining.
5) Use explicit JOINs with correct keys (e.g., results.raceId = races.raceId; results.driverId = drivers.driverId).
6) Driver full name: (drivers.forename || ' ' || drivers.surname) AS name.
7) When returning a breakdown, include useful columns (year, round, race/circuit, driver/constructor) and an ORDER BY.
8) If the user didn’t specify a year and you need one, do NOT invent it—answer across all years (but still return an ORDER BY and LIMIT if the result could be huge).
"""

# Few-shots (include a breakdown example)
FEW_SHOTS = [
    {
      "q": "wins for Max Verstappen in 2023",
      "sql": """SELECT COUNT(*) AS wins
FROM results rs
JOIN races r ON r.raceId = rs.raceId
JOIN (SELECT driverId, (forename || ' ' || surname) AS name FROM drivers) d
  ON d.driverId = rs.driverId
WHERE d.name = 'Max Verstappen' AND r.year = 2023 AND rs.positionText = '1';"""
    },
    {
      "q": "points for Lewis Hamilton in 2021",
      "sql": """WITH x AS (
  SELECT rs.points
  FROM results rs
  JOIN races r ON r.raceId = rs.raceId
  JOIN (SELECT driverId, (forename || ' ' || surname) AS name FROM drivers) d
    ON d.driverId = rs.driverId
  WHERE d.name = 'Lewis Hamilton' AND r.year = 2021
)
SELECT COALESCE(SUM(points),0) AS total_points FROM x;"""
    },
    {
      # breakdown example (constructor points by race, ordered)
      "q": "Ferrari points by race in 2024 (per round, most recent first)",
      "sql": """SELECT r.year, r.round, r.name AS race_name,
       c.name AS constructor, SUM(rs.points) AS points
FROM results rs
JOIN races r ON r.raceId = rs.raceId
JOIN constructors c ON c.constructorId = rs.constructorId
WHERE c.name = 'Ferrari' AND r.year = 2024
GROUP BY 1,2,3,4
ORDER BY r.year DESC, r.round DESC;"""
    },
    {
      # another breakdown across seasons
      "q": "Sergio Perez podiums by season since 2020",
      "sql": """SELECT r.year, d.name, COUNT(*) AS podiums
FROM results rs
JOIN races r ON r.raceId = rs.raceId
JOIN (SELECT driverId, (forename || ' ' || surname) AS name FROM drivers) d
  ON d.driverId = rs.driverId
WHERE d.name = 'Sergio Pérez'
  AND try_cast(rs.positionText AS INTEGER) BETWEEN 1 AND 3
  AND r.year >= 2020
GROUP BY 1,2
ORDER BY r.year;"""
    },
    {
      # pit stop example
      "q": "Average pit stop duration for Ferrari in 2024 by race",
      "sql": """SELECT r.year, r.round, r.name AS race_name,
       c.name AS constructor,
       AVG(try_cast(ps.milliseconds AS DOUBLE)) AS avg_pit_ms
FROM pit_stops ps
JOIN races r ON r.raceId = ps.raceId
JOIN results rs ON rs.raceId = ps.raceId AND rs.driverId = ps.driverId
JOIN constructors c ON c.constructorId = rs.constructorId
WHERE c.name = 'Ferrari' AND r.year = 2024
GROUP BY 1,2,3,4
ORDER BY r.round;"""
    }
]

def _schema_text():
    parts = []
    for t, cols in SCHEMA.items():
        coltxt = ", ".join([f"{c['name']}:{c['type']}" for c in cols])
        parts.append(f"- {t}({coltxt})")
    return "\n".join(parts)

# Small helper: if user hints "breakdown", we add a gentle instruction
_BREAKDOWN_HINTS = re.compile(r"\b(by race|per round|timeline|by season|per circuit|per track|breakdown)\b", re.I)

def _augment_question(q: str) -> str:
    if _BREAKDOWN_HINTS.search(q):
        return q + " — return a multi-row breakdown with clear columns and an ORDER BY."
    return q

def llm_to_sql(question: str) -> str:
    shots = "\n\n".join([f"Q: {s['q']}\nSQL:\n```sql\n{s['sql']}\n```" for s in FEW_SHOTS])
    prompt = (
      f"{SYSTEM_RULES}\n\n"
      f"Tables:\n{_schema_text()}\n\n"
      f"{shots}\n\n"
      f"Q: {_augment_question(question)}\nSQL:"
    )
    resp = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig()
    )
    text = resp.text or ""
    m = re.search(r"```sql\s*(.+?)```", text, re.S|re.I)
    if not m:
        raise ValueError("LLM did not return SQL in a fenced block.")
    sql = m.group(1).strip()

    # ---- Safety checks ----
    if not re.match(r"^\s*(with|select)\b", sql, re.I):
        raise ValueError("Only SELECT/WITH queries are allowed.")
    # allow trailing semicolon but not chaining
    if ";" in sql.strip()[:-1]:
        raise ValueError("Multiple statements not allowed.")
    # very light table whitelist (optional): ensure it only uses known tables/views
    tbls = "|".join(map(re.escape, SCHEMA.keys()))
    if not re.search(rf"\b({tbls})\b", sql, re.I):
        # not fatal, but it helps avoid hallucinated tables
        pass

    return sql

def answer(question: str, max_rows: int = 200):
    sql = llm_to_sql(question)

    # Append LIMIT only when:
    # - there is no LIMIT already
    # - and there is NO aggregation (count/sum/avg/min/max) AND NO GROUP BY
    #   (i.e., a potentially huge raw row set)
    needs_limit = (
        re.search(r"\blimit\b", sql, re.I) is None
        and re.search(r"\b(count|sum|avg|min|max)\s*\(", sql, re.I) is None
        and re.search(r"\bgroup\s+by\b", sql, re.I) is None
    )
    if needs_limit:
        sql += f"\nLIMIT {max_rows}"

    con = duckdb.connect("f1.duckdb")
    data = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    return sql, cols, data
