# app.py — F1 Smart Search (chat transcript + robust answering + F1 scoping)

import re
import streamlit as st

import os, streamlit as st
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
LIVE_OK = bool(GEMINI_API_KEY)


# ---- Optional backends (use your existing backend files if present) ----
LLM_OK, sql_answer = False, None
try:
    # should return (sql, cols, rows)
    from backend.sql_agent import answer as _sql_answer
    sql_answer, LLM_OK = _sql_answer, True
except Exception:
    pass

LIVE_OK, ask_live = False, None
try:
    # should return {"text": "...", "citations": [...]}
    from backend.rag_web import ask_live as _ask_live
    ask_live, LIVE_OK = _ask_live, True
except Exception:
    pass

# ---- Optional local DuckDB fallback for "who is <driver>" ----
DUCK_OK, duckdb = False, None
try:
    import duckdb  # used only by the local driver lookup
    DUCK_OK = True
except Exception:
    DUCK_OK = False


# ============================ Page / Styles ============================

st.set_page_config(page_title="F1 Smart Search", layout="wide")

RAIL_WIDTH_PX = 820          # left-rail width
RAIL_LEFT_MARGIN_PX = 60     # left gutter

st.markdown(
    f"""
    <style>
      :root {{
        --card-bg: rgba(255,255,255,0.06);
        --border: rgba(255,255,255,0.12);
      }}
      .block-container {{ max-width: 1100px; padding-top: 1rem; }}

      /* Title */
      .title-wrap {{ text-align:center; margin:.25rem 0 0 0; }}
      .title-wrap h1 {{ font-size:2.2rem; letter-spacing:.3px; margin:0; }}
      .subtitle {{ text-align:center; opacity:.75; margin:.4rem 0 1rem 0; }}

      /* Transcript rail */
      .rail-left {{ max-width:{RAIL_WIDTH_PX}px; margin-left:{RAIL_LEFT_MARGIN_PX}px; }}

      /* Chat bubbles */
      .msg {{
          display:block; max-width:calc({RAIL_WIDTH_PX}px - 16px);
          padding:.8rem 1rem; border-radius:14px; margin:.35rem 0;
          border:1px solid var(--border); white-space:pre-wrap;
      }}
      .msg.user {{ margin-left:auto; background:rgba(125, 162, 255, .12); }}
      .msg.assistant {{ margin-right:auto; background:var(--card-bg); }}

      /* Fixed bottom chat */
      .chat-fixed {{
        position:fixed; left:{RAIL_LEFT_MARGIN_PX}px; bottom:18px; z-index:1000;
        width:{RAIL_WIDTH_PX}px; background:var(--card-bg); border:1px solid var(--border);
        border-radius:14px; padding:.9rem;
      }}

      /* Bottom spacer so content isn't hidden by fixed chat */
      .bottom-pad {{ height: 120px; }}

      /* Inputs / buttons */
      .stTextInput > div > div > input {{ border-radius:12px !important; height:52px; }}
      .stButton>button {{ border-radius:12px; font-weight:600; padding:.55rem 1rem; border:1px solid var(--border); }}
      .dataframe {{ border-radius:10px; overflow:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title-wrap"><h1>🏁 F1 Smart Search</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">One clean answer — stats + latest info when relevant.</div>', unsafe_allow_html=True)

# Backends banner
st.caption(f"DB agent: {'ON' if LLM_OK else 'OFF'} • Live RAG: {'ON' if LIVE_OK else 'OFF'} • Local DuckDB fallback: {'ON' if DUCK_OK else 'OFF'}")


# ===================== F1 heuristics & normalizers =====================

RE_DB = re.compile(
    r"\b(wins?|points?|podiums?|poles?|qualifying|grid|fastest|laps?|"
    r"pit\s*stops?|constructor[s']?|driver standings?|positions?|"
    r"by race|per round|season|since \d{4}|in \d{4}|championship)\b",
    re.I,
)
RE_LIVE = re.compile(
    r"\b(live|today|latest|news|update|upgrades?|package|penalt(y|ies)|"
    r"investigation|hearing|appeal|fia|stewards|press|statement|rumou?rs?)\b",
    re.I,
)
RE_YEAR_WINNER = re.compile(r"^\s*(who\s+won|winner)\s+in\s+(\d{4})\s*\??$", re.I)
F1_HINT_WORDS = ("f1", "formula 1", "formula one", "grand prix", "fia", "constructor", "driver", "pit", "pole")

def wants_db(q: str) -> bool:   return bool(RE_DB.search(q))
def wants_live(q: str) -> bool: return bool(RE_LIVE.search(q))
def looks_f1(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in F1_HINT_WORDS)

def normalize_to_f1(q: str) -> tuple[str, bool]:
    """Return (normalized_query, prefer_db_first)."""
    m = RE_YEAR_WINNER.match(q)
    if m:
        year = m.group(2)
        return f"In Formula 1, who won the Drivers' Championship in {year}?", True
    if not looks_f1(q):
        q = f"In Formula 1 context: {q}"
    return q, wants_db(q)

def summarize_sql_rows(cols, rows, fallback="No data found."):
    if not rows:
        return fallback, None
    if len(rows) == 1 and len(cols) <= 3:
        return ", ".join(f"{cols[i]}: {rows[0][i]}" for i in range(len(cols))), None
    return None, [dict(zip(cols, r)) for r in rows]


# ===================== Local fallback (DuckDB) =====================

def local_driver_lookup(question: str):
    """
    Lightweight local bio for: 'who is <driver>'
    Requires f1.duckdb with a drivers table.
    """
    if not DUCK_OK:
        return None
    m = re.search(r"who\s+is\s+([a-z\s\.\-']+)$", question.strip(), re.I)
    if not m:
        return None
    candidate = m.group(1).strip().lower()
    try:
        con = duckdb.connect("f1.duckdb", read_only=True)
        rs = con.execute("""
            WITH d AS (
              SELECT driverId,
                     lower(trim(forename || ' ' || surname)) AS fullname,
                     forename, surname, nationality, dob, code, "number" AS number, url
              FROM drivers
            )
            SELECT forename, surname, nationality, dob, code, number, url
            FROM d
            WHERE fullname LIKE ? OR fullname SIMILAR TO ?
            ORDER BY levenshtein(fullname, ?) ASC
            LIMIT 1
        """, [f"%{candidate}%", f"%{candidate}%", candidate]).fetchone()
        con.close()
        if not rs:
            return None
        forename, surname, nationality, dob, code, number, url = rs
        num = f" #{number}" if number is not None else ""
        code = f" ({code})" if code else ""
        return (f"**{forename} {surname}**{code}{num} — {nationality}, born {dob}. "
                f"A Formula 1 driver. More: {url}")
    except Exception:
        return None


# =============================== State ===============================

if "history" not in st.session_state:
    # Each item: {"role": "user"|"assistant", "text": str, "tables": Optional[list[dict]]}
    st.session_state.history = []
if "q" not in st.session_state:
    st.session_state.q = ""


# ========================== Render transcript ==========================

st.markdown('<div class="rail-left">', unsafe_allow_html=True)
for m in st.session_state.history:
    css = "user" if m["role"] == "user" else "assistant"
    st.markdown(f'<div class="msg {css}">{m["text"]}</div>', unsafe_allow_html=True)
    if m["role"] == "assistant" and m.get("tables"):
        for t in m["tables"]:
            st.dataframe(t, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ============================ Bottom chat ============================

st.markdown('<div class="chat-fixed">', unsafe_allow_html=True)
st.session_state.q = st.text_input(
    "Ask anything:",
    value=st.session_state.q,
    label_visibility="collapsed",
    key="chat_input",
    placeholder="e.g., who won in 2021? or Ferrari points by race in 2024",
)
asked = st.button("Ask", key="ask_btn")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="bottom-pad"></div>', unsafe_allow_html=True)


# ============================ Submit logic ============================

if asked and st.session_state.q.strip():
    raw_q = st.session_state.q.strip()
    # 1) Add USER bubble
    st.session_state.history.append({"role": "user", "text": raw_q})

    # 2) Normalize and route
    q, prefer_db = normalize_to_f1(raw_q)

    parts, tables = [], []
    try:
        did_db = False

        # DB first if indicated
        if LLM_OK and (prefer_db or wants_db(q)):
            try:
                sql, cols, rows = sql_answer(q)
                did_db = True
                sent, table = summarize_sql_rows(cols, rows)
                if sent: parts.append(sent)
                if table: tables.append(table)
            except Exception as e:
                parts.append(f"_(DB error: {e})_")

        # Live RAG if looks newsy or DB returned nothing
        do_live = (not did_db) or wants_live(q)
        if do_live and LIVE_OK:
            try:
                # Prefer topic_hint if available
                try:
                    live = ask_live(q, topic_hint="f1")
                except TypeError:
                    live = ask_live(f"Formula 1 / F1 only: {q}")
                txt = (live.get("text") or "").strip()
                if txt:
                    parts.append(txt)
            except Exception as e:
                parts.append(f"_(Live error: {e})_")

        # Local fallback for "who is <driver>"
        if not parts and not tables:
            bio = local_driver_lookup(raw_q)
            if bio:
                parts.append(bio)

        # Final guaranteed assistant bubble
        if not parts and not tables:
            parts.append(
                "I didn’t find a result with the current back-ends. "
                "Try a more specific F1 query, or enable DB/LIVE in your environment."
            )

        st.session_state.history.append({
            "role": "assistant",
            "text": "\n\n".join(parts),
            "tables": tables if tables else None
        })

    except Exception as e:
        # Absolute safety net
        st.session_state.history.append({
            "role": "assistant",
            "text": f"Something went wrong while answering: `{e}`"
        })

    # Clear input and re-render so new bubbles appear
    st.session_state.q = ""
    st.rerun()
