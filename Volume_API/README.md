# Volume API Tutorial

A focused tutorial for the WorldQuant Brain / Bigdata.com Volume API. This notebook demonstrates how to use the Volume API to get aggregated document and chunk counts over time.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
5. [API Endpoint](#api-endpoint)
6. [Tutorial Sections](#tutorial-sections)
7. [File Structure](#file-structure)
8. [Key Parameters](#key-parameters)

## Features

- **Volume API**: Aggregated document and chunk counts per day
- **Time Series Analysis**: Understand data distribution over time
- **Visualizations**: Interactive Plotly charts showing volume evolution with sentiment
- **Entity Filtering**: Filter volume data by specific entities
- **Helper Functions**: Utilities for data retrieval and visualization

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

## Quick Start

1. Install dependencies (see above)
2. Set up your `.env` file with credentials
3. Open the notebook:
```bash
jupyter notebook Volume_API_Tutorial.ipynb
```

4. Run cells sequentially from the top

## API Endpoint

| Endpoint | Description |
|----------|-------------|
| `/bigdata/v1/search/volume` | Document volume aggregation |

**Documentation:** https://docs.bigdata.com/api-reference/search/get-volume-data

## Tutorial Sections

### Basic Usage
1. **Setup & Configuration** - Authentication and endpoint setup
2. **Volume API** - Basic volume query without entity filters
3. **Volume Evolution Visualization** - Time series visualization with sentiment analysis

## File Structure

```
Volume_API/
├── Volume_API_Tutorial.ipynb    # Main tutorial notebook
├── api_helpers.py                 # Helper functions for visualization
├── print_helpers.py               # Pretty printing utilities
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## Key Parameters

### Volume API Query

| Parameter | Description | Required |
|-----------|-------------|----------|
| `text` | Search query text | Yes |
| `auto_enrich_filters` | Auto-extract entities from text | No (recommended: `False`) |
| `filters.timestamp` | Time range (start/end in ISO 8601 format) | Yes |
| `filters.entity` | Filter by entities (any_of, all_of, none_of) | No |
| `ranking_params.freshness_boost` | Recency bias (0 = historical) | No (recommended: `0` for historical data) |

### Response Structure

The Volume API returns:
- **`total`**: Overall totals for documents and chunks
- **`volume`**: Array of daily data with:
  - `date`: Date in YYYY-MM-DD format
  - `documents`: Number of documents for that day
  - `chunks`: Number of chunks for that day
  - `sentiment`: Average sentiment score for that day (-1.0 to 1.0)

### Helper Functions

- **`print_volume_results()`**: Pretty print volume API results
- **`get_volume_dataframe()`**: Fetch volume data as a pandas DataFrame
- **`get_volume_totals()`**: Get only the total counts
- **`plot_volume_evolution()`**: Create multi-panel visualization with 7-day moving averages

## Examples

### Basic Volume Query

```python
volume_query = {
    "query": {
        "text": "Global semiconductor shortage impacts",
        "auto_enrich_filters": False,
        "filters": {
            "timestamp": {
                "start": "2021-01-01T00:00:00Z",
                "end": "2021-12-30T23:59:59Z"
            }
        }
    }
}

response = session.post(VOLUME_ENDPOINT, json=volume_query)
volume_data = response.json()
```

### Volume with Entity Filter

```python
volume_entity_query = {
    "query": {
        "text": "Global semiconductor shortage impacts",
        "auto_enrich_filters": False,
        "filters": {
            "timestamp": {
                "start": "2021-01-01T00:00:00Z",
                "end": "2021-12-30T23:59:59Z"
            },
            "entity": {"any_of": ["D8442A"]}  # Apple Inc
        }
    }
}
```

### Visualization

```python
from api_helpers import plot_volume_evolution

fig = plot_volume_evolution(
    volume_data=volume_data,
    text="Your query text",
    start_date="2021-01-01",
    end_date="2021-12-30"
)
fig.show()
```

## Notes

- The Volume API does NOT return actual documents, only aggregated counts
- Use the Volume API to understand data distribution before running expensive Search API queries
- The `freshness_boost` parameter should be set to `0` for historical data (2021-2022)
- Entity filters support `any_of`, `all_of`, and `none_of` modes
- The visualization includes 7-day moving averages to smooth out daily fluctuations

## License

This tutorial is part of the WorldQuant Brain / Bigdata.com API documentation.
