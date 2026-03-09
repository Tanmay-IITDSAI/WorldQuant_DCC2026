# Knowledge Graph API Tutorial

A focused tutorial for the WorldQuant Brain / Bigdata.com Knowledge Graph API. This notebook demonstrates how to use the Knowledge Graph API for company lookup, entity resolution, and source discovery.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
5. [API Endpoints Covered](#api-endpoints-covered)
6. [Tutorial Sections](#tutorial-sections)
7. [File Structure](#file-structure)
8. [Key Concepts](#key-concepts)

## Features

- **Company Lookup**: Find company entity IDs by name
- **Entity Resolution**: Translate entity IDs to detailed entity information
- **Source Discovery**: Find news sources by name with quality filters
- **Entity Discovery Workflow**: Complete end-to-end workflow for entity discovery

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
jupyter notebook Knowledge_Graph_API_Tutorial.ipynb
```

4. Run cells sequentially from the top

## API Endpoints Covered

| Endpoint | Description |
|----------|-------------|
| `/authentication` | Cookie-based authentication |
| `/bigdata/v1/knowledge-graph/companies` | Company lookup by name |
| `/bigdata/v1/knowledge-graph/entities/id` | Entity details by ID |
| `/bigdata/v1/knowledge-graph/sources` | News source lookup |

## Tutorial Sections

### Basic Usage
1. **Setup & Configuration** - Authentication and endpoint setup
2. **Searching with Company IDs** - Finding company entity IDs
3. **Translating Entity IDs to Names** - Resolving entity IDs to detailed information

### Advanced Workflows
4. **Entity Discovery Workflow** - Complete end-to-end example:
   - Finding a company entity ID
   - Using it in search queries
   - Extracting entity IDs from results
   - Resolving entity IDs to names

5. **Finding Sources** - Discovering news sources with quality filters

## File Structure

```
Knowledge_Graph_API/
├── Knowledge_Graph_API_Tutorial.ipynb    # Main tutorial notebook
├── print_helpers.py                      # Pretty printing utilities
├── api_helpers.py                        # Helper functions (minimal for KG API)
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variables template
└── README.md                             # This file
```

## Key Concepts

### Entity IDs

- Entity IDs are 6-character hexadecimal identifiers (e.g., `D8442A`)
- They uniquely identify companies, people, places, products, and concepts
- Entity IDs are used to filter search results and resolve entity names

### Company Lookup

- The Companies API returns a **list** of matching results
- You must review results and select the most correct element
- For well-known companies with correct spelling, the first result is typically correct

### Entity Resolution

- The Entities API accepts multiple entity IDs at once
- Returns a dictionary mapping entity IDs to entity objects
- Entity objects contain: name, ID, type, and optionally ticker (for companies)

### Source Discovery

- Sources can be filtered by:
  - **Rank**: Quality ranking (RANK_1 = highest quality)
  - **Category**: news, transcripts, research, podcasts, filings, expert_interviews
  - **Country**: ISO 3166-1 A-2 country codes
  - **Packages**: Data packages (e.g., sec_filings)

### Common Workflow

1. **Find Company ID**: Use Companies API to get entity ID
2. **Search with Entity**: Use entity ID in Search API filters
3. **Extract Entity IDs**: Get entity IDs from search result detections
4. **Resolve to Names**: Use Entities API to get entity names

## Integration with Other APIs

The Knowledge Graph API is often used in combination with:

- **Search API**: Use entity IDs to filter search results
- **Volume API**: Use entity IDs to filter volume queries
- **Co-mentions API**: Use entity IDs to discover relationships

For complete examples using Knowledge Graph with other APIs, see:
- `Search_API/` - Using entity IDs in search filters
- `Volume_API/` - Using entity IDs in volume queries
- `CoMentions_API/` - Using entity IDs in co-mentions analysis

## Documentation

- **Official API Documentation**: https://docs.bigdata.com/api-reference/knowledge-graph
- **Companies API**: https://docs.bigdata.com/api-reference/knowledge-graph/find-companies
- **Entities API**: https://docs.bigdata.com/api-reference/knowledge-graph/get-entities-by-id
- **Sources API**: https://docs.bigdata.com/api-reference/knowledge-graph/find-sources

## Notes

- The provider APIs do not have full coverage for parameter validation
- Always verify entity IDs match your intended entities
- Entity IDs are case-sensitive
- The API returns lists - always review and select the correct result
