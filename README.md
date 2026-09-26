# F1 Smart Search Engine

A dual-agent AI search system for querying Formula 1 data — combining a structured database agent for precise statistics with an LLM-powered fallback agent for open-ended, natural-language questions.

## Overview

🔗 **Live demo:** https://f1-smart-search.onrender.com
*(Free tier: first load may take ~1 min to wake up)*

**Stack:** Streamlit · DuckDB · Gemini · Jolpica F1 API · Docker · Render (auto-deploy on push)

F1 Smart Search Engine lets users ask questions about Formula 1 — drivers, races, lap times, standings, historical results — through a single conversational interface. Rather than relying on one retrieval method, the system intelligently routes each query to the agent best suited to answer it, balancing speed/precision with flexibility.

## How It Works

The system uses a **dual-agent architecture**:

1. **Database Agent (Structured Queries)**
   A custom natural-language-to-SQL intent router parses the user's question and translates it into a query against a **DuckDB** analytical backend. This agent handles precise, fact-based questions (e.g., "What was Verstappen's fastest lap at Monza in 2023?") with high accuracy and low latency.

2. **LLM Fallback Agent (Open-Ended Queries)**
   When a query doesn't map cleanly to structured data — or falls outside what the database can answer — the system falls back to **Gemini 2.5 Flash** as a live Retrieval-Augmented Generation (RAG) agent. This allows the system to gracefully handle broader, conversational, or ambiguous questions.

3. **Graceful Degradation**
   The routing logic is designed around reliability: the DB agent handles what it can answer precisely, and only hands off to the LLM agent when needed — rather than defaulting to an LLM for everything. This keeps responses fast and accurate for the majority of structured queries while still supporting natural conversation.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend / Chat UI | Streamlit |
| Analytical Database | DuckDB |
| NL → SQL Routing | Custom intent router |
| LLM Fallback / RAG | Gemini 2.5 Flash |
| Language | Python |

## Architecture

```
                 ┌─────────────────────┐
                 │   User Query (UI)   │
                 │     Streamlit       │
                 └──────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │  Intent Router      │
                  │  (NL → SQL classifier) │
                  └─────────┬──────────┘
                            │
            ┌───────────────┴───────────────┐
            │                                │
   ┌────────▼─────────┐           ┌──────────▼─────────┐
   │   DB Agent         │           │  LLM Fallback Agent │
   │   (DuckDB / SQL)    │           │  (Gemini 2.5 Flash)  │
   │   Structured stats  │           │  Open-ended / RAG     │
   └────────┬─────────┘           └──────────┬─────────┘
            │                                │
            └───────────────┬───────────────┘
                            │
                  ┌─────────▼──────────┐
                  │   Response to User  │
                  └─────────────────────┘
```

## Getting Started

> **Note:** Update the steps below to match your actual setup (dependencies, dataset source, API keys, etc.)

### Prerequisites

- Python 3.10+
- A Gemini API key (for the LLM fallback agent)

### Installation

```bash
git clone https://github.com/fateenkhan46/f1-smart-search.git
cd f1-smart-search
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
DUCKDB_PATH=./data/f1_data.duckdb
```

### Running the App

```bash
streamlit run app.py
```

## Example Queries

- *"Who won the 2021 Abu Dhabi Grand Prix?"* → handled by the DB agent
- *"What made the 2021 title fight so controversial?"* → handled by the LLM fallback agent
- *"Compare Hamilton and Verstappen's win rates in 2021"* → handled by the DB agent

## Key Design Highlights

- **Production-grade reliability thinking**: the system doesn't treat the LLM as a universal answer engine — it's a fallback, not the default, which keeps costs and latency low for the bulk of queries.
- **Modular routing**: the intent router can be extended with additional agents or data sources without restructuring the core pipeline.

## Future Improvements

- Expand structured dataset coverage (qualifying, pit stops, weather conditions)
- Add caching for repeated queries
- Support voice input for hands-free querying

## License

MIT (or update as appropriate)
