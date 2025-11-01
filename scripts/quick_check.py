import os, duckdb

print("CWD:", os.getcwd())
print("DB exists:", os.path.exists("f1.duckdb"))

con = duckdb.connect("f1.duckdb")

# quick table counts
print("results rows:",  con.execute("SELECT COUNT(*) FROM results").fetchone()[0])
print("races rows:",    con.execute("SELECT COUNT(*) FROM races").fetchone()[0])
print("drivers rows:",  con.execute("SELECT COUNT(*) FROM drivers").fetchone()[0])

driver, year = "Max Verstappen", 2023
wins = con.execute("""
  SELECT COUNT(*) AS wins
  FROM results rs
  JOIN races r ON r.raceId = rs.raceId
  JOIN (SELECT driverId, (forename || ' ' || surname) AS name FROM drivers) dn
    ON dn.driverId = rs.driverId
  WHERE dn.name = ? AND r.year = ? AND rs.positionText = '1';
""", [driver, year]).fetchone()[0]

print(f"{driver} wins in {year} = {int(wins)}")
