# Search API Tutorial

A comprehensive tutorial for the WorldQuant Bigdata.com Search API. This notebook demonstrates how to use the Search API for semantic document search with entity and keyword filters, query optimization, and source boosting.

## Overview

The Search API allows you to search for documents based on text queries with advanced filtering capabilities. This tutorial covers:

- **Basic Search**: Text-based semantic search
- **Entity Filtering**: Filter results by specific entities
- **Query Optimization**: Best practices for `max_chunks` parameter
- **Source Boosting**: Controlling source quality with `source_boost`
- **Response Structure**: Understanding API responses and chunk data

## Features

- Text-based semantic search with relevance ranking
- Entity and keyword filtering (any_of, all_of, none_of)
- Timestamp filtering for date ranges
- Source quality control via `source_boost`
- Query optimization techniques
- Interactive visualizations for parameter analysis

## Installation

### Option 1: Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd Search_API
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
cd Search_API
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

## Quick Start

1. Install dependencies (see above)
2. Set up your `.env` file with credentials
3. Open the notebook:
```bash
jupyter notebook Search_API_Tutorial.ipynb
```

4. Run cells sequentially from the top

## API Endpoint

| Endpoint | Description |
|----------|-------------|
| `/bigdata/v1/search` | Semantic document search |
| `/bigdata/v1/search/volume` | Document volume aggregation (used for comparison in optimization) |

## Tutorial Sections

### 1. Search API
Basic text-based search with timestamp filtering.

### 1.1 Response Structure
Understanding the API response format, including document structure, chunks, and metadata.

### 2. Advanced Options in the Search API: Filters and Ranking
In this section we look at filters and ranking in detail.

### 3. Query Chunk Size: Best Practices
How many chunks to request and how that affects quality and coverage. 

### 4. Source Boosting: Controlling Source Quality
How to favor premium sources and the trade-off with coverage.

### 5. Effect of freshness on volume distribution
We compare freshness_boost = 0 vs freshness_boost = 10 and how that shifts the time distribution of results.

### 5. Search API with Entity Filter
Filtering search results by specific entity IDs.

### 6. Effect of reranker and threshold on relevance distribution
Comparing three configurations shows how the distribution of relevance scores changes.

## Key Parameters

### Search API

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `filters` | Generic filtering settup, includes timestamp, source or entity among others. |  |
| `auto_enrich_filters` | Auto-extract entities from text | `False` (for full control) |
| `ranking_params` | Groups parameters that affect the ranking algorithm and determine chunk relevance. | Described below |
| `max_chunks` | Maximum chunks to retrieve | Start with 100-500 |


### Entity Filter

Entity filter to retrieve chunks where entities were identified.

| Mode | Description |
|------|-------------|
| `any_of` | Match any entity in list |
| `all_of` | Match all entities in list |
| `none_of` | Exclude entities in list |

### Ranking Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `freshness_boost` | Boost for recent documents | 0-10 |
| `source_boost` | Boost for premium sources | 0-10 (default 1.0) |
| `reranker` | Re-ranking with relevance threshold | Enabled by default |
| `threshold` | Relevance score filter (0.0-1.0) | 0.2 (higher = fewer but more relevant) |

## Important Notes

1. **Parameter Validation**: The provider APIs do not have full coverage for parameter validation. Make sure you write parameters correctly. Refer to the [official documentation](https://docs.bigdata.com/api-reference/search/search-documents).

2. **Challenge Requirement**: For querying data from 2021 to 2022, the `freshness_boost` parameter MUST be set to 0. This ensures historical data is retrieved without recency bias.

3. **Chunks are Ranked by Relevance**: The API returns chunks ordered from MOST relevant to LEAST relevant. Increasing `max_chunks` adds progressively less relevant chunks.

4. **Quality vs Quantity Trade-off**: Higher `max_chunks` = more data but lower average quality. Lower `max_chunks` = less data but higher average quality.

## File Structure

```
Search_API/
├── Search_API_Tutorial.ipynb    # Main tutorial notebook
├── api_helpers.py                # Helper functions for search and visualization
├── print_helpers.py              # Pretty printing utilities
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## Helper Functions

### print_helpers.py
- `print_search_results()`: Pretty print search API results

### api_helpers.py
- `run_search()`: Execute a single search query
- `grid_parameter_search()`: Run grid search over parameters
- `get_volume_dataframe()`: Get volume data for comparison (used in optimization)
- `plot_freshness_comparison()`: Visualize results across freshness_boost values
- `plot_source_distribution()`: Analyze source distribution by source_boost
- `plot_source_rank_distribution()`: Analyze source rank distribution
- `plot_chunks_vs_max_chunks()`: Visualize chunks retrieved vs max_chunks parameter

## Documentation
- **Search API Reference**: https://docs.bigdata.com/api-reference/search/search-documents

## Examples

### Basic Search
```python
search_query = {
    "query": {
        "text": "Global semiconductor shortage impacts",
        "auto_enrich_filters": False,
        "filters": {
            "timestamp": {
                "start": "2021-01-01T00:00:00Z",
                "end": "2021-12-30T23:59:59Z"
            }
        },
        "ranking_params": {
            "freshness_boost": 0
        },
        "max_chunks": 10
    }
}

response = session.post(SEARCH_ENDPOINT, json=search_query)
data = response.json()
```

### Search with Entity Filter
```python
search_with_entity = {
    "query": {
        "text": "Global semiconductor shortage impacts",
        "auto_enrich_filters": False,
        "filters": {
            "timestamp": {
                "start": "2021-01-01T00:00:00Z",
                "end": "2021-12-30T23:59:59Z"
            },
            "entity": {
                "any_of": ["D8442A"]  # Apple Inc
            }
        },
        "ranking_params": {
            "freshness_boost": 0
        },
        "max_chunks": 10
    }
}
```

## License

This tutorial is part of the WorldQuant Bigdata.com API Tutorial collection.
