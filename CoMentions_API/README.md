# Co-mentions API Tutorial

A focused tutorial for the WorldQuant Brain / Bigdata.com Co-mentions API. This notebook demonstrates how to discover entities that are frequently mentioned together with your search query and visualize their relationships as network graphs.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
5. [API Endpoints Covered](#api-endpoints-covered)
6. [Tutorial Sections](#tutorial-sections)
7. [File Structure](#file-structure)

## Features

- **Co-mentions API**: Discover entities frequently mentioned together with your search query
- **Network Graph Visualization**: Interactive Plotly network graphs showing entity relationships
- **Entity Name Resolution**: Automatically resolve entity IDs to human-readable names using the Knowledge Graph API
- **Category-based Analysis**: Explore co-mentions by entity type (companies, places, people, products, organizations, concepts)

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
jupyter notebook CoMentions_API_Tutorial.ipynb
```

4. Run cells sequentially from the top

## API Endpoints Covered

| Endpoint | Description |
|----------|-------------|
| `/authentication` | Cookie-based authentication |
| `/bigdata/v1/search/co-mentions/entities` | Co-mentioned entities discovery |
| `/bigdata/v1/knowledge-graph/entities/id` | Entity details by ID (for name resolution) |

## Tutorial Sections

### Section 3: Co-mentions API

Learn how to query the Co-mentions API to discover entities that are frequently mentioned together with your search query. The API returns entities organized by type:
- Companies
- Places
- People
- Products
- Organizations
- Concepts

### Section 13: Co-mentions Network Graph

Visualize entity relationships as interactive network graphs. This section demonstrates:
- How to filter co-mentions by a specific entity
- Creating network graphs for different entity categories
- Resolving entity IDs to human-readable names
- Customizing graph visualization parameters

## File Structure

```
CoMentions_API/
├── CoMentions_API_Tutorial.ipynb  # Main tutorial notebook
├── print_helpers.py                # Helper function for printing results
├── api_helpers.py                   # Helper function for network graphs
├── requirements.txt                 # Python dependencies
├── .env.example                    # Example environment file
└── README.md                       # This file
```

## Key Parameters

### Co-mentions API Query Parameters

- `query.text`: The search query text
- `query.auto_enrich_filters`: Set to `False` for full control (recommended)
- `query.filters.timestamp`: Date range filter
- `query.filters.entity`: Optional entity filter (any_of, all_of, none_of)
- `limit`: Maximum number of entities to return per category

### Network Graph Parameters

- `max_nodes`: Maximum number of connected nodes to display (default: 20)
- `category_name`: Entity category to visualize (companies, places, etc.)
- `center_name`: Name of the center entity
- `center_id`: ID of the center entity

## Examples

### Basic Co-mentions Query

```python
comention_query = {
    "query": {
        "text": "Global semiconductor shortage impacts",
        "auto_enrich_filters": False,
        "filters": {
            "timestamp": {
                "start": "2021-01-01T00:00:00Z",
                "end": "2021-12-30T23:59:59Z"
            }
        }
    },
    "limit": 10
}
```

### Co-mentions with Entity Filter

```python
comention_query = {
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
    },
    "limit": 100
}
```

## Documentation

- **Co-mentions API**: https://docs.bigdata.com/api-reference/search/get-co-mentions
- **Knowledge Graph API**: https://docs.bigdata.com/api-reference/knowledge-graph

## Notes

- The Co-mentions API requires the Knowledge Graph Entities endpoint to resolve entity IDs to names for network graph visualizations
- Network graphs are interactive Plotly visualizations that can be zoomed and panned
- Entity relationships are shown as edges connecting the center entity to co-mentioned entities
- Node sizes in network graphs are proportional to the number of chunks where entities appear together
