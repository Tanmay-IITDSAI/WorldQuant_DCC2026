# Smart Batching Tutorial

A focused tutorial for high-performance semantic search over large universes using smart batching and proportional sampling. This notebook demonstrates how to reduce API queries by **67–99%** (varies by topic) through intelligent company grouping and parallel execution. Authentication uses WorldQuant Brain via `BRAIN_EMAIL` and `BRAIN_PASSWORD`.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
5. [Tutorial Sections](#tutorial-sections)
6. [How It Works](#how-it-works)
7. [Key Parameters](#key-parameters)
8. [File Structure](#file-structure)
9. [Examples](#examples)
10. [Documentation](#documentation)
11. [Notes](#notes)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

## Features

- **Smart Batching Planning**: Organize searches using intelligent basket creation based on chunk volumes (co-mention API)
- **Proportional Sampling**: Retrieve a percentage of total chunks while preserving distribution across baskets
- **Parallel Execution**: Rate-limited concurrent requests with semaphore control
- **Plan Persistence**: Save and load search plans for reuse with different sampling percentages
- **67–99% Query Reduction**: Niche topics up to ~99.85% reduction; specialized topics ~96–97%; broad topics ~32–67%

## Installation

### Option 1: Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:
```
BRAIN_EMAIL=your_email@example.com
BRAIN_PASSWORD=your_password
```

3. Optional: override API base URL

Load `.env` with `python-dotenv` before creating the session if you use a `.env` file.

## Quick Start

1. Install dependencies (see above)
2. Set up your `.env` file with credentials
3. Open the notebook:
```bash
jupyter notebook test_smart_batching.ipynb
```

4. Run cells sequentially from the top (run from the `Smart_Batching` directory so `from src import ...` resolves)

### Basic usage (Python script)

```python
from src import plan_search, execute_search, deduplicate_documents, convert_to_dataframe

session = BrainSession()

plan = plan_search(
    text="earnings revenue profit",
    universe_csv_path="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31",
    session=session,
)

results_raw = execute_search(search_plan=plan, chunk_percentage=0.1, session=session)
results = deduplicate_documents(results_raw)
df = convert_to_dataframe(results)  # one row per chunk
```

## Tutorial Sections

The notebook walks through:

- **Planning**: Load universe from CSV, query co-mention endpoint for chunk volumes, create baskets (high/medium/low volume), build search plan
- **Execution**: Proportional sampling per basket, parallel search with rate limiting (default 100 RPM), collect and deduplicate results
- **Helpers**: `save_plan` / `load_plan`, `load_universe_from_csv`, `convert_to_dataframe`

## How It Works

**Two-step flow:**

1. **Planning** — `plan_search()` loads the universe, gets chunk volumes via the co-mention API, groups companies into baskets by volume (high → individual baskets; low → large grouped baskets), and returns a plan with total expected chunks and basket configs.
2. **Execution** — `execute_search()` computes proportional chunks per basket (`expected_chunks * chunk_percentage`), runs searches in parallel with a sliding-window rate limiter and semaphore (default 40 workers), and returns a list of documents. Use `deduplicate_documents()` then optionally `convert_to_dataframe()`.

```
Universe CSV → Co-mention API (volumes) → Basket creation → Search plan
       → Proportional sampling → Parallel search (rate limited) → Results
```

Query reduction is largest for niche/specialized topics; broader topics (e.g. "Earnings") still see significant reduction but yield more chunks.

## Key Parameters

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BRAIN_EMAIL` | Yes | WorldQuant Brain email |
| `BRAIN_PASSWORD` | Yes | WorldQuant Brain password |

### plan_search()

| Parameter | Description                               |
|-----------|-------------------------------------------|
| `text` | Search query text                         |
| `universe_csv_path` | Path to CSV with entity IDs               |
| `start_date`, `end_date` | YYYY-MM-DD                                |
| `session` | `BrainSession()` instance (recommended)   |
| `volume_query_mode` | `"three_pass"` (default) or `"iterative"` |

Returns a dict with `total_expected_chunks`, `baskets`, and `planning_metadata`.

### execute_search()

| Parameter | Description | Default |
|-----------|-------------|---------|
| `search_plan` | Result from `plan_search()` | — |
| `chunk_percentage` | Fraction of chunks to retrieve (0.0–1.0) | — |
| `session` | `BigDataSessionWQ()` instance (recommended) | — |
| `requests_per_minute` | Rate limit (sliding window) | 100 |
| `max_workers` | Max concurrent connections | 40 |

Returns a list of document dicts; use `deduplicate_documents()` then optionally `convert_to_dataframe()`.

### Performance (summary)

| Metric | Naive search | Smart batching |
|--------|--------------|----------------|
| Queries (4,732 companies, example) | 11,357 | 17–3,699 (67–99% reduction) |
| Use when | Small universe (&lt; 100), exact per-company | Large universe (&gt; 500), scalable search, rate limits |

## File Structure

```
Smart_Batching/
├── test_smart_batching.ipynb  # Main tutorial notebook
├── src/
│   ├── __init__.py            # Exports session, plan_search, execute_search, etc.
│   ├── bigdata_session_wq.py  # WorldQuant Brain session
│   ├── search_function.py      # plan_search, execute_search, deduplicate_documents, save_plan, load_plan, load_universe_from_csv
│   ├── output_converter.py     # convert_to_dataframe
│   ├── smart_batching.py      # Batching logic
│   └── smart_batching_config.py # Config and API base URL
├── test_search_plan.json      # Example saved plan (optional)
├── requirements.txt          # Python dependencies
├── .env.example               # BRAIN_EMAIL, BRAIN_PASSWORD template
└── README.md                  # This file
```

## Examples

### Basic search

```python
from src import plan_search, execute_search, deduplicate_documents, convert_to_dataframe

session = BrainSession()
plan = plan_search(
    text="earnings revenue profit",
    universe_csv_path="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31",
    session=session,
)
results_raw = execute_search(plan, chunk_percentage=0.1, session=session)
results = deduplicate_documents(results_raw)
df = convert_to_dataframe(results)
```

### Save and load plan

```python
from src import plan_search, execute_search, save_plan, load_plan, deduplicate_documents

session = BrainSession()
plan = plan_search(
    text="merger acquisition",
    universe_csv_path="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31",
    session=session,
)
save_plan(plan, "my_search_plan.json")

# Later: reuse plan with different chunk_percentage
plan = load_plan("my_search_plan.json")
raw_10 = execute_search(plan, chunk_percentage=0.1, session=session)
raw_50 = execute_search(plan, chunk_percentage=0.5, session=session)
```

## Documentation

- **Search API**: https://docs.bigdata.com/api-reference/search/search-documents
- **Co-mentions API** (used in planning): https://docs.bigdata.com/api-reference/search/get-co-mentions

## Notes

- Use Smart Batching for large universes (&gt; 500 companies), rate-limited or cost-sensitive use cases, and when you need proportional sampling. Prefer naive per-company search for very small universes or when exact per-company results are required.
- Query reduction is most dramatic for niche/specialized topics; broader topics still benefit but return more chunks.
- For full API details and helper function signatures, see docstrings in `src`.

## Troubleshooting

- **Authentication**: Set `BRAIN_EMAIL` and `BRAIN_PASSWORD` in `.env` or the environment; run `load_dotenv()` before `BrainSession()` if using a `.env` file.
- **Rate limits (429)**: Lower `requests_per_minute` (e.g. 50) or `max_workers` (e.g. 20) in `execute_search()`.
- **Memory**: Use a smaller `chunk_percentage` (e.g. 0.05) or process results in batches.
- **Slow execution**: Increase `max_workers` or `requests_per_minute` only if your API limits allow.
- **Planning slow**: Planning queries the co-mention API per company; cache plans with `save_plan` for reuse.

## License

This project is part of the WorldQuant Data Creation Challenge.

**Disclaimer**: This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors assume no responsibility for the accuracy, completeness, or usefulness of any information, results, or processes provided. This software is for educational and research purposes only and is not intended to be used as financial advice. Any use of this software for investment or trading decisions is at your own risk. The authors and contributors shall not be liable for any damages arising from the use of this software.
