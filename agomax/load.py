# ==============================================================================
# load.py
# PURPOSE:
#   Any file -> CSV
#   No preprocessing, no ML, no feature logic
# ==============================================================================

import pandas as pd
from pathlib import Path


# ==============================================================================
# 1. READ FILE (format-specific)
# ==============================================================================

def read_file(input_path: str) -> pd.DataFrame:
    """
    Read file into a pandas DataFrame based on extension.
    Raises exception on failure (FAIL FAST).
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = input_path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(input_path)

    if ext == ".json":
        try:
            return pd.read_json(input_path)
        except ValueError:
            return pd.read_json(input_path, lines=True)

    if ext == ".parquet":
        return pd.read_parquet(input_path)

    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(input_path)

    if ext in [".pkl", ".pickle"]:
        return pd.read_pickle(input_path)

    # Fallback: try generic CSV parsing
    return pd.read_csv(input_path, sep=None, engine="python")


# ==============================================================================
# 2. NORMALIZE (STRUCTURE ONLY)
# ==============================================================================

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize structure only:
    - Flatten nested JSON if present
    - Do NOT drop or rename columns
    """
    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    # Detect nested dict-like columns
    if any(isinstance(v, dict) for v in df.iloc[0].values):
        df = pd.json_normalize(df.to_dict(orient="records"))

    return df


# ==============================================================================
# 3. WRITE CSV
# ==============================================================================

def write_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Write DataFrame to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


# ==============================================================================
# 4. ORCHESTRATOR
# ==============================================================================

def convert_to_csv(input_path: str, output_path: str) -> None:
    """
    End-to-end:
    input file -> normalized dataframe -> csv
    """
    df = read_file(input_path)
    df = normalize(df)
    write_csv(df, output_path)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python load.py <input_file> <output_csv>"
        )

    convert_to_csv(sys.argv[1], sys.argv[2])
    print(f"[OK] Converted {sys.argv[1]} -> {sys.argv[2]}")
