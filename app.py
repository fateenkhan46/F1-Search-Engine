import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Secrets / Environment
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY  # some clients read only env

DB_URL = st.secrets.get("DB_URL", "")  # optional: URL to download f1.duckdb on first boot
APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "f1.duckdb"

# ----------------------------
# Caching helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def _download_db_if_needed() -> None:
    """
    Download f1.duckdb from DB_URL once if it doesn't exist.
    Use a public link (e.g., GitHub Release, S3, etc.).
    """
    if DB_PATH.exists() or not DB_URL:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading F1 database…"):
        with requests.get(DB_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(DB_PATH, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

@st.cache_resource(show_spinner=False)
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    _download_db_if_needed()
    if not DB_PATH.exists():
        raise FileNotFoundError("f1.duckdb not found (and no DB_URL provided).")
    # read-only is safer for hosted environments
    con = duckdb.connect(str(DB_PATH), read_only=True)
    return con

def _safe_query(sql: str, params: Tuple = ()) -> pd.DataFrame:
    """Run a DuckDB query safely; returns empty DF on any error."""
    try:
        con = get_duckdb_connection()
        return con.execute(sql, params).df()
    except Exception:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)

# ----------------------------
# Backend flags
# ----------------------------
def db_available() -> bool:
    try:
        con = get_duckdb_connection()
        con.execute("SELECT 1").fetchone()
        return True
    except Exception:
        return False

def live_available() -> bool:
    try:
        if not GEMINI_API_KEY:
            return False
        _ = get_gemini_client()
        return True
    except Exception:
        return False

DB_OK = db_available()
LIVE_OK = live_available()

# ----------------------------
# SQL utilities (F1 dataset)
# ----------------------------
def _latest_year() -> Optional[int]:
    df = _safe_query("SELECT MAX(year) AS y FROM races")
    if not df.empty and pd.notna(df.iloc[0,0]):
        return int(df.iloc[0,0])
    return None

def _normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()

def q_constructor_drivers(team: str, year: Optional[int]) -> pd.DataFrame:
    """
    Drivers who raced for a constructor in a given year.
    """
    if year is None:
        year = _latest_year()
    sql = """
    WITH names AS (
        SELECT driverId, (forename || ' ' || surname) AS name
        FROM drivers
    )
    SELECT DISTINCT n.name AS driver, c.name AS constructor, r.year
    FROM results rs
    JOIN races r ON r.raceId = rs.raceId
    JOIN constructors c ON c.constructorId = rs.constructorId
    JOIN names n ON n.driverId = rs.driverId
    WHERE r.year = ? AND LOWER(c.name) = LOWER(?)
    ORDER BY driver
    """
    return _safe_query(sql, (year, team))

def q_driver_wins(driver: str, year: Optional[int]) -> pd.DataFrame:
    if year is None:
        year = _latest_year()
    sql = """
    WITH names AS (
        SELECT driverId, (forename || ' ' || surname) AS name
        FROM drivers
    )
    SELECT COUNT(*) AS wins
    FROM results rs
    JOIN races r ON r.raceId = rs.raceId
    JOIN names n ON n.driverId = rs.driverId
    WHERE r.year = ? AND n.name = ? AND rs.positionText = '1'
    """
    return _safe_query(sql, (year, driver))

def q_driver_champion(year: int) -> pd.DataFrame:
    """
    Driver champion for given year using driver_standings at last round.
    """
    sql = """
    WITH names AS (
      SELECT driverId, (forename || ' ' || surname) AS name FROM drivers
    ),
    last_round AS (
      SELECT year, MAX(round) AS max_round
      FROM races
      WHERE year = ?
      GROUP BY year
    )
    SELECT n.name AS champion, ds.points, ds.wins
    FROM driver_standings ds
    JOIN races r ON r.raceId = ds.raceId
    JOIN last_round lr ON lr.year = r.year AND lr.max_round = r.round
    JOIN names n ON n.driverId = ds.driverId
    WHERE ds.position = 1
    """
    return _safe_query(sql, (year, ))

def q_constructor_champion(year: int) -> pd.DataFrame:
    sql = """
    WITH last_round AS (
      SELECT year, MAX(round) AS max_round
      FROM races
      WHERE year = ?
      GROUP BY year
    )
    SELECT c.name AS champion, cs.points, cs.wins
    FROM constructor_standings cs
    JOIN races r ON r.raceId = cs.raceId
    JOIN last_round lr ON lr.year = r.year AND lr.max_round = r.round
    JOIN constructors c ON c.constructorId = cs.constructorId
    WHERE cs.position = 1
    """
    return _safe_query(sql, (year, ))

def q_points_by_race_for_constructor(team: str, year: Optional[int]) -> pd.DataFrame:
    if year is None:
        year = _latest_year()
    sql = """
    SELECT r.round, r.name AS race, SUM(rs.points) AS team_points
    FROM results rs
    JOIN races r ON r.raceId = rs.raceId
    JOIN constructors c ON c.constructorId = rs.constructorId
    WHERE r.year = ? AND LOWER(c.name) = LOWER(?)
    GROUP BY r.round, r.name
    ORDER BY r.round
    """
    return _safe_query(sql, (year, team))

# ----------------------------
# NL → action router
# ----------------------------
YEAR_RE = r"(19|20)\d{2}"
TEAM_RE = r"(ferrari|mercedes|red bull|mclaren|aston martin|williams|alpine|sauber|haas|rb|alphatauri|toro rosso|renault)"

def parse_question(q: str):
    """
    Very light intent parsing to keep the app fast and dependency-free.
    Returns (intent, args_dict) where intent in {team_drivers, driver_wins, champ, team_points}.
    """
    text = q.strip().lower()
    year = None
    m = re.search(YEAR_RE, text)
    if m:
        year = int(m.group(0))

    # team name
    team = None
    m2 = re.search(TEAM_RE, text)
    if m2:
        team = m2.group(0).title()
        # normalize some historical variants
        if team in ["Alpha Tauri", "AlphaTauri", "Toro Rosso"]:
            team = "AlphaTauri"
        if team == "Rb":
            team = "RB"

    # driver full name heuristic: two words
    driver = None
    words = q.strip().split()
    # crude: look for "Firstname Lastname"
    for i in range(len(words) - 1):
        cand = f"{words[i]} {words[i+1]}"
        if re.match(r"^[A-Za-z][a-z]+ [A-Za-z][a-z\-']+$", cand):
            driver = _normalize_name(cand)
            break

    # Map intents
    if "who drives for" in text or "drivers for" in text or "who drives" in text:
        if team:
            return ("team_drivers", {"team": team, "year": year})
    if "wins" in text and driver:
        return ("driver_wins", {"driver": driver, "year": year})
    if "who won" in text or ("champion" in text and "driver" in text):
        if year:
            return ("champ_driver", {"year": year})
    if "constructor" in text and "champion" in text and year:
        return ("champ_constructor", {"year": year})
    if ("points by race" in text or ("points" in text and "by race" in text)) and team:
        return ("team_points", {"team": team, "year": year})

    # default: try a sensible F1 question pattern
    return ("auto", {"team": team, "driver": driver, "year": year})

# ----------------------------
# Answer engine
# ----------------------------
def answer_with_db(intent: str, args: dict) -> Optional[str]:
    """Return a markdown string or None if nothing found / no DB."""
    if not DB_OK:
        return None

    if intent == "team_drivers":
        team = args.get("team")
        year = args.get("year") or _latest_year()
        df = q_constructor_drivers(team, year)
        if df.empty:
            return None
        drivers = ", ".join(sorted(df["driver"].unique()))
        return f"**{team}** drivers in **{year}**: {drivers}."

    if intent == "driver_wins":
        driver = args.get("driver")
        year = args.get("year") or _latest_year()
        df = q_driver_wins(driver, year)
        if df.empty:
            return None
        w = int(df.iloc[0]["wins"] or 0)
        return f"**{driver}** wins in **{year}**: **{w}**."

    if intent == "champ_driver":
        year = int(args["year"])
        df = q_driver_champion(year)
        if df.empty:
            return None
        row = df.iloc[0]
        return f"**{year} F1 Driver Champion:** **{row['champion']}**  \nPoints: {int(row['points'])} • Wins: {int(row['wins'])}"

    if intent == "champ_constructor":
        year = int(args["year"])
        df = q_constructor_champion(year)
        if df.empty:
            return None
        row = df.iloc[0]
        return f"**{year} F1 Constructors’ Champion:** **{row['champion']}**  \nPoints: {int(row['points'])} • Wins: {int(row['wins'])}"

    if intent == "team_points":
        team = args.get("team")
        year = args.get("year") or _latest_year()
        df = q_points_by_race_for_constructor(team, year)
        if df.empty:
            return None
        lines = [f"**{team} points by race in {year}:**"]
        for _, r in df.iterrows():
            lines.append(f"- Round {int(r['round'])}: {r['race']} — **{int(r['team_points'])}** pts")
        return "\n".join(lines)

    # "auto" heuristic: try team drivers → driver champion → constructor champion
    if intent == "auto":
        team = args.get("team")
        driver = args.get("driver")
        year = args.get("year") or _latest_year()
        if team:
            out = answer_with_db("team_drivers", {"team": team, "year": year})
            if out:
                return out
        if year:
            out = answer_with_db("champ_driver", {"year": year})
            if out:
                return out
            out = answer_with_db("champ_constructor", {"year": year})
            if out:
                return out
        if driver:
            out = answer_with_db("driver_wins", {"driver": driver, "year": year})
            if out:
                return out
        return None

    return None

def answer_with_live(q: str) -> Optional[str]:
    """Very lightweight Live ‘summary’ using Gemini (no web search)."""
    if not LIVE_OK:
        return None
    try:
        client = get_gemini_client()
        prompt = (
            "Answer the F1-related question concisely (2–4 sentences). "
            "If the question is not F1-specific, gently steer back to F1. "
            "Question: " + q
        )
        rsp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        txt = (rsp.text or "").strip()
        return txt if txt else None
    except Exception:
        return None

def build_answer(q: str) -> str:
    intent, args = parse_question(q)

    # 1) Try DB precise answer
    out = answer_with_db(intent, args)
    if out:
        return out

    # 2) Try Live (generic) if enabled
    out = answer_with_live(q)
    if out:
        return out

    # 3) Final fallback
    return ("I didn’t find a result with the current back-ends. "
            "Try a more specific F1 query, or enable DB/LIVE in your deployment.")

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="F1 Smart Search", page_icon="🏁", layout="wide")

title_col = st.columns([1, 6, 1])[1]
with title_col:
    st.markdown(
        "<h1 style='text-align:center'>🏁 F1 Smart Search</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;opacity:.8'>One clean answer — stats + latest info when relevant.</p>",
        unsafe_allow_html=True,
    )

status = f"DB agent: {'ON' if DB_OK else 'OFF'} • Live RAG: {'ON' if LIVE_OK else 'OFF'}"
st.caption(status)

# Chat history in session
if "msgs" not in st.session_state:
    st.session_state.msgs = []

# Display history
for role, content in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(content)

# Chat input at the bottom
prompt = st.chat_input("Ask anything about F1 (e.g., 'who drives for Ferrari this year')")

if prompt:
    st.session_state.msgs.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans = build_answer(prompt)
        st.markdown(ans)

    st.session_state.msgs.append(("assistant", ans))
