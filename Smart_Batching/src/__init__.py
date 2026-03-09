"""
Smart Batching module for BigData search operations.

This module provides:
- plan_search, execute_search: High-level search functions
- save_plan, load_plan, load_universe_from_csv, deduplicate_documents
- convert_to_dataframe: Result conversion
"""

from .search_function import (
    plan_search,
    execute_search,
    deduplicate_documents,
    save_plan,
    load_plan,
    load_universe_from_csv,
)
from .output_converter import convert_to_dataframe

__all__ = [
    'plan_search',
    'execute_search',
    'deduplicate_documents',
    'save_plan',
    'load_plan',
    'load_universe_from_csv',
    'convert_to_dataframe',
]
