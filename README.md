# Bigdata.com and WorldQuant Data Creation Challenge

Example notebooks and best practices to help participants get started and compete effectively in Bigdata.com and WorldQuant Data Creation Challenges. This repository contains **eight Jupyter notebooks**: four focused API tutorials (Search, Volume, Co-mentions, Knowledge Graph), plus two workflow-oriented notebooks that combine APIs and optional LLM tooling for thematic search and signal construction.

The goal is to build **quant signals**: numeric series per entity over time (from news/sentiment) that you can use for ranking, screening, or backtesting. Start with the API tutorials to learn each endpoint; then use **Smart_Batching** to run efficient searches over large universes, and **Workflow_example** to see an end-to-end thematic-search pipeline that produces a signal table ready for backtesting or screening.

---

## Notebooks (all six in one place)

Each row links to a folder. Inside you’ll find the notebook, helper modules, and a README with setup and run instructions.

| Notebook | What you’ll learn |
|----------|-------------------|
| [**Search_API**](./Search_API/) | **Semantic document search.** Run meaning-based search over news with filters (entity, time, sentiment, source). Control ranking (reranker on/off, relevance threshold, source_boost, freshness_boost) and tune `max_chunks`. Compare relevance distributions with and without the reranker. Use the Volume API inside the notebook to compare retrieval vs available volume. |
| [**Volume_API**](./Volume_API/) | **Aggregated counts over time.** Get document and chunk counts per day for a query—no document payload—so you can measure coverage and sentiment trends before running heavier Search API calls. Same semantic query and filters as Search; useful for backtesting and deciding when to run full search. |
| [**CoMentions_API**](./CoMentions_API/) | **Who is talked about with your topic.** Discover entities (companies, places, people, products, concepts) that are frequently co-mentioned with your query. Build network graphs and thematic baskets; resolve entity IDs to names via the Knowledge Graph. |
| [**Knowledge_Graph_API**](./Knowledge_Graph_API/) | **Names ↔ IDs and source discovery.** Resolve company and entity names to IDs (for use in Search/Volume/Co-mentions filters) and IDs back to names and metadata. Find sources by name and filter by rank, category, or country. Walk through a full workflow: company lookup → Search with entity filter → resolve IDs from chunks. |
| [**Smart_Batching**](./Smart_Batching/) | **Efficient search at scale.** Two-step system: smart batching and proportional sampling over a large universe of entities. Use the notebook `test_smart_batching.ipynb` to run batched searches and inspect plans; includes helpers, tests, and usage examples. Best after you’re comfortable with the Search API. |
| [**Workflow_example**](./Workflow_example/) | **End-to-end thematic search and signals.** Break a theme into sub-themes, build and execute optimized search plans with Smart Batching, mask company names and validate relevance/impact with an LLM, then build rolling sentiment signals. Output: a signal table (one row per entity, date) with `signal_7d` and `signal_30d` ready for backtesting or screening. Uses WorldQuant Brain and optional OpenAI. |
| [**Workflow_multi_theme_sentiment**](./Workflow_multi_theme_sentiment/) | **Multi-theme sentiment signals** Orthogonal financial themes with direction-aware weighting to build robust composite sentiment signals. Pre-scan data availability via Volume API, execute batch searches across themes × months × entities, then construct MATRIX data fields for direct upload to WorldQuant BRAIN for alpha simulation. |
| [**Competition_Full_Workflow_Demo**](./Competition_Full_Workflow_Demo/) | **Data Creation Competition Workflow.** End-to-end workflow for the Data Creation competition, transforming raw company IDs into actionable sentiment signals. |
---

## Suggested order (learning path)

Recommended sequence so each notebook builds on the previous:

1. **Search API** — Semantic search and filters; raw chunks are the input for downstream signals.
2. **Volume API** — Aggregated counts over time; use to see where you have enough data for a stable signal.
3. **Knowledge Graph API** — Resolve company/entity names to IDs (used in Search/Volume filters and in Workflow_example).
4. **CoMentions API** — Thematic baskets and entity context; same entity IDs feed into search and signals.
5. **Smart_Batching** — Efficient search at scale so you can build signals across many entities.
6. **Workflow_example** — Full pipeline from theme to rolling impact signal table.
7. **Competition_Full_Workflow_Demo** — Data Creation Competition Workflow.

---

## Key terms

- **Chunk:** A text snippet plus metadata from a document returned by the API.
- **Entity ID:** Stable identifier for companies, people, places, etc.; obtained via the Knowledge Graph API.
- **Point-in-time / backtesting:** Using only data available as of each date (no lookahead) so backtests are valid.

### Data files and folder layout

- **Data files:** Some CSV files (e.g. `id_name_mapping_us_top_3000.csv`) appear in both `Smart_Batching/` and `Workflow_example/`. Each folder is self-contained; if you change the universe or mapping, update the copy in both folders.
- **Universe scope:** 'Competition_Full_Workflow_Demo/universe.json' contains the instrument IDs in TOP3000 universe. 
- **Shared utility code:** `Smart_Batching/src/` and `Workflow_example/src/` contain similar helper and session code. They are kept as separate copies so each folder runs independently; there is no shared package.

---

## Setup and how to run

### Credentials

- **WorldQuant Brain:** In each tutorial folder, copy `.env.example` to `.env` and set `BRAIN_EMAIL` and `BRAIN_PASSWORD`.
- **Workflow_example (LLM step):** Optional `OPENAI_API_KEY` in `.env` for the validation step.

Credentials are usually loaded at import time; restart the kernel after changing `.env`.

### Install and run a notebook

1. Open the folder for the notebook you want (e.g. `Search_API`, `Smart_Batching`).
2. Create a virtual environment and install dependencies (each folder has a `requirements.txt`; use `uv` or `pip` as in the Quick Start below).
3. Configure `.env` in that folder as above.
4. Start Jupyter and open the notebook (e.g. `Search_API_Tutorial.ipynb`, `test_smart_batching.ipynb`, `Workflow_example.ipynb`).
5. Run cells from top to bottom; follow the markdown for context and next steps.

### Quick Start (Smart_Batching)

If you want to run the Smart Batching notebook first:

1. Go to the Smart_Batching directory:
   ```bash
   cd Smart_Batching
   ```

2. Install dependencies (use **uv** if possible):
   ```bash
   # With uv
   uv pip install -r requirements.txt

   # Or with pip
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set your WorldQuant Brain credentials:
   ```bash
   cp .env.example .env
   # Edit .env: BRAIN_EMAIL=your_email, BRAIN_PASSWORD=your_password
   ```

4. Run the notebook:
   ```bash
   jupyter notebook test_smart_batching.ipynb
   ```

For more detail, see [Smart_Batching/README.md](./Smart_Batching/README.md) and the docstrings in the `src` modules.

---

## License

This project is part of the Bigdata.com and WorldQuant Data Creation Challenge.

**Disclaimer:** This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors assume no responsibility for the accuracy, completeness, or usefulness of any information, results, or processes provided. This software is for educational and research purposes only and is not intended to be used as financial advice. Any use of this software for investment or trading decisions is at your own risk. The authors and contributors shall not be liable for any damages arising from the use of this software.
