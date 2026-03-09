# Workflow Example Tutorial

A focused tutorial for end-to-end thematic search and signal construction using WorldQuant Brain. This notebook demonstrates how to break a theme into sub-themes, run optimized searches with Smart Batching or Full Grid Search, validate relevance and impact with an LLM (with company masking to avoid bias), and build rolling sentiment signals.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
5. [Workflow Sections](#workflow-sections)
6. [File Structure](#file-structure)
7. [Environment Variables](#environment-variables)
8. [Notes](#notes)

## Features

- **Theme Decomposition**: Break a main theme into sub-themes for higher recall
- **Search Planning & Execution**: Create and execute optimized search plans using Smart Batching
- **Masking & LLM Validation**: Mask company names to avoid bias; validate theme relevance and impact with an optional LLM step
- **Signal Construction**: Build rolling sentiment signals and visualize top companies
- **Optional OpenAI**: Use `OPENAI_API_KEY` for the validation step; workflow runs without it (skip or mock that step)

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

3. Optional (for LLM validation step):
```
OPENAI_API_KEY=your_openai_key
```

4. Optional (override API URL):

Credentials are read at import time; restart the kernel after changing `.env`.

## Quick Start

1. Install dependencies (see above)
2. Set up your `.env` file with credentials
3. Open the notebook:
```bash
jupyter notebook Workflow_example.ipynb
```

4. Run cells sequentially from the top

## Workflow Sections

### 1. Theme Decomposition

Break down a main theme into sub-themes to improve search recall.

### 2. Search Planning & Execution

Two approaches are available:

- **Smart Batching**: Create and execute optimized search plans over a large universe of entities. This approach distributes queries evenly to avoid high media-attention companies dominating results. See [Smart_Batching/README.md](../Smart_Batching/README.md) for details on planning and execution.

- **Full Grid Search**: Standard approach that queries a full grid of entities using the same parameters (text query, date range, entity filter). More efficient when preserving distribution across companies is not an issue (e.g., small time windows with limited data volumes or niche topics). However, since each call is limited to 1,000 chunks, companies with very high chunk volumes can consume most of the quota, leaving other companies under-represented or missing. 

### 3. Masking & LLM Validation

Mask company names in chunks to avoid bias, then validate theme relevance and impact using an LLM (optional; requires `OPENAI_API_KEY`).

### 4. Signal Construction

Build rolling sentiment signals and visualize top companies by volume or signal strength.

## File Structure

```
Workflow_example/
├── Workflow_example.ipynb    # Main tutorial notebook
├── src/
│   ├── __init__.py           # Package exports
│   ├── bigdata_session_wq.py # WorldQuant Brain session
│   ├── helper.py             # Data and visualization helpers
│   ├── output_converter.py   # Convert results to DataFrame
│   ├── processing_results.py # Result aggregation and entity processing
│   ├── search_function.py    # plan_search, execute_search, etc.
│   ├── smart_batching_config.py
│   ├── smart_batching.py     # Smart batching logic
│   ├── labeler/              # Screener/labeler components
│   ├── mindmap/              # Mindmap generation utilities
│   └── prompts/              # LLM prompt helpers
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BRAIN_EMAIL` | Yes | WorldQuant Brain email |
| `BRAIN_PASSWORD` | Yes | WorldQuant Brain password |
| `OPENAI_API_KEY` | No | OpenAI API key for LLM validation step |

## Notes

- Restart the kernel after changing credentials in `.env`, as they are read at import time.
- The LLM validation step is optional; you can skip it or use a mock if you do not set `OPENAI_API_KEY`.
- For search planning and execution details (baskets, proportional sampling, rate limits), see [Smart_Batching/README.md](../Smart_Batching/README.md).
