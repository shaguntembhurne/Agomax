"""
Utility functions for data loading and validation.

This module provides helper functions for loading telemetry data from
various sources.
"""

import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Any

from .exceptions import InvalidDataError


def load_data(
    source: Union[str, Path, pd.DataFrame, Dict, List[Dict]]
) -> pd.DataFrame:
    """
    Load telemetry data from various sources.
    
    Parameters
    ----------
    source : str, Path, DataFrame, dict, or list of dict
        Data source:
        - str/Path: File path (CSV, JSON, Parquet, Excel, Pickle)
        - DataFrame: Returns copy
        - dict: Single telemetry record
        - list of dict: Multiple telemetry records
        
    Returns
    -------
    df : pd.DataFrame
        Loaded data
        
    Raises
    ------
    InvalidDataError
        If source is invalid or file doesn't exist
        
    Examples
    --------
    >>> df = load_data("flight_data.csv")
    >>> df = load_data({"altitude": 100, "velocity": 5})
    >>> df = load_data([{"altitude": 100}, {"altitude": 101}])
    """
    if source is None:
        raise InvalidDataError("Data source is None")
    
    # DataFrame
    if isinstance(source, pd.DataFrame):
        return source.copy()
    
    # File path
    if isinstance(source, (str, Path)):
        return _load_file(Path(source))
    
    # Single record
    if isinstance(source, dict):
        return pd.DataFrame([source])
    
    # Multiple records
    if isinstance(source, list) and source and isinstance(source[0], dict):
        return pd.DataFrame(source)
    
    raise InvalidDataError(
        f"Unsupported data source type: {type(source)}. "
        f"Expected str, Path, DataFrame, dict, or list of dict."
    )


def _load_file(path: Path) -> pd.DataFrame:
    """Load data from file based on extension."""
    if not path.exists():
        raise InvalidDataError(f"File not found: {path}")
    
    ext = path.suffix.lower()
    
    loaders = {
        ".csv": lambda p: pd.read_csv(p),
        ".json": _load_json,
        ".parquet": lambda p: pd.read_parquet(p),
        ".xlsx": lambda p: pd.read_excel(p),
        ".xls": lambda p: pd.read_excel(p),
        ".pkl": lambda p: pd.read_pickle(p),
        ".pickle": lambda p: pd.read_pickle(p),
    }
    
    if ext in loaders:
        try:
            return loaders[ext](path)
        except Exception as e:
            raise InvalidDataError(f"Failed to load {path}: {e}")
    
    # Try CSV as fallback
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise InvalidDataError(
            f"Unknown file type '{ext}' and CSV parsing failed: {e}"
        )


def _load_json(path: Path) -> pd.DataFrame:
    """Load JSON with fallback to line-delimited JSON."""
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def save_data(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Save DataFrame to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    path : str or Path
        Output file path
    **kwargs
        Additional arguments passed to pandas save function
        
    Examples
    --------
    >>> save_data(df, "output.csv")
    >>> save_data(df, "output.parquet", compression="gzip")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = path.suffix.lower()
    
    if ext == ".csv":
        df.to_csv(path, index=False, **kwargs)
    elif ext == ".json":
        df.to_json(path, orient="records", **kwargs)
    elif ext == ".parquet":
        df.to_parquet(path, **kwargs)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False, **kwargs)
    elif ext in (".pkl", ".pickle"):
        df.to_pickle(path, **kwargs)
    else:
        # Default to CSV
        df.to_csv(path, index=False, **kwargs)
