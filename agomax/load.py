# ==============================================================================
# load.py
# PURPOSE:
#   Any telemetry source -> pandas DataFrame
# ==============================================================================

from pathlib import Path
import pandas as pd
from agomax.logger import get_logger

logger = get_logger("agomax.load")


# ==============================================================================
# MAIN ENTRY
# ==============================================================================

def load(source) -> pd.DataFrame:
    logger.info("Loading telemetry source")

    if source is None:
        logger.error("Source is None")
        raise ValueError("Source is None")

    if isinstance(source, pd.DataFrame):
        logger.info("Source type: DataFrame")
        df = source.copy()

    elif isinstance(source, (str, Path)):
        logger.info(f"Source type: file ({source})")
        df = _read_file(Path(source))

    elif isinstance(source, dict):
        logger.info("Source type: single telemetry snapshot")
        df = pd.DataFrame([source])

    elif isinstance(source, list) and source and isinstance(source[0], dict):
        logger.info(f"Source type: batch telemetry ({len(source)} records)")
        df = pd.DataFrame(source)

    else:
        logger.error(f"Unsupported source type: {type(source)}")
        raise TypeError(f"Unsupported source type: {type(source)}")

    df = _normalize(df)
    _validate(df)

    logger.info(f"Loaded DataFrame shape: {df.shape}")
    return df


# ==============================================================================
# FILE READERS
# ==============================================================================

def _read_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    logger.info(f"Reading file type: {ext}")

    if ext == ".csv":
        return pd.read_csv(path)

    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)

    if ext == ".parquet":
        return pd.read_parquet(path)

    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)

    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)

    logger.warning("Unknown extension, attempting generic CSV parse")
    return pd.read_csv(path, sep=None, engine="python")


# ==============================================================================
# NORMALIZATION
# ==============================================================================

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Normalizing structure")

    if df.empty:
        logger.error("Empty DataFrame")
        raise ValueError("Empty DataFrame")

    if any(isinstance(v, dict) for v in df.iloc[0].values):
        logger.info("Flattening nested telemetry fields")
        df = pd.json_normalize(df.to_dict(orient="records"))

    return df


# ==============================================================================
# VALIDATION
# ==============================================================================

def _validate(df: pd.DataFrame) -> None:
    logger.info("Validating telemetry")

    if df.empty:
        logger.error("Validation failed: empty DataFrame")
        raise ValueError("Empty DataFrame")

    if df.select_dtypes(include="number").shape[1] == 0:
        logger.error("No numeric columns found")
        raise ValueError("No numeric columns found")

    null_cols = df.columns[df.isna().all()]
    if len(null_cols) > 0:
        logger.warning(f"Dropping fully-null columns: {list(null_cols)}")
        df.drop(columns=null_cols, inplace=True)


# ==============================================================================
# CSV UTILITY
# ==============================================================================

def to_csv(df: pd.DataFrame, output_path: str) -> None:
    logger.info(f"Writing CSV -> {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python load.py <input> <output_csv>")

    df = load(sys.argv[1])
    to_csv(df, sys.argv[2])
    logger.info("Conversion complete")
