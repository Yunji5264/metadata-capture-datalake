import os
import json
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from reference import PERF_DIR


def _safe_get(d: Dict[str, Any], key: str, default=None):
    """
    Safely read one key from a dict.

    This is used to make the perf-table construction more robust in case
    some perf.json files are missing optional fields.
    """
    return d.get(key, default) if isinstance(d, dict) else default


def _flatten_perf_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten one perf.json entry into a single row.

    Input structure:
        {
            "datasetId": ...,
            "rawzonePath": ...,
            "fileSizeBytes": ...,
            "executedAt": ...,
            "timings": {
                "read_df_sec": ...,
                ...
            }
        }

    Output structure:
        One flat dictionary suitable for building a pandas DataFrame.
    """
    timings = _safe_get(entry, "timings", {}) or {}

    return {
        # identification
        "datasetId": _safe_get(entry, "datasetId"),
        "rawzonePath": _safe_get(entry, "rawzonePath"),
        "executedAt": _safe_get(entry, "executedAt"),

        # dataset size
        "fileSizeBytes": _safe_get(entry, "fileSizeBytes"),

        # detailed timing fields
        "read_df_sec": timings.get("read_df_sec"),
        "semantic_helper_sec": timings.get("semantic_helper_sec"),
        "classify_attributes_sec": timings.get("classify_attributes_sec"),
        "scopes_granularities_sec": timings.get("scopes_granularities_sec"),
        "find_common_theme_sec": timings.get("find_common_theme_sec"),
        "transform_result_sec": timings.get("transform_result_sec"),
        "total_sec": timings.get("total_sec"),
    }


def load_perf_entries(perf_dir: str = PERF_DIR) -> List[Dict[str, Any]]:
    """
    Read all perf log files under PERF_DIR and return flattened rows.

    Each file is expected to be a JSON file ending with '.perf.json'.
    Files that cannot be parsed are skipped with a warning.
    """
    perf_dir = str(perf_dir)
    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(perf_dir, "*.perf.json")

    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            rows.append(_flatten_perf_entry(entry))
        except Exception as ex:
            print(f"[WARN] Failed to parse {path}: {ex}")

    return rows


def build_perf_table(
    perf_dir: str = PERF_DIR,
    out_csv: Optional[str] = os.path.join(str(PERF_DIR), "perf_summary.csv"),
    out_parquet: Optional[str] = os.path.join(str(PERF_DIR), "perf_summary.parquet"),
) -> pd.DataFrame:
    """
    Aggregate all perf logs into a single table.

    Parameters
    ----------
    perf_dir:
        Directory containing '*.perf.json' files.

    out_csv:
        Output CSV path. If None, no CSV is written.

    out_parquet:
        Output Parquet path. If None, no Parquet file is written.

    Returns
    -------
    pandas.DataFrame
        A table where each row corresponds to one perf log entry.
    """
    perf_dir = str(perf_dir)
    rows = load_perf_entries(perf_dir)

    if not rows:
        print(f"No perf files found under {perf_dir}.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert executedAt to datetime when possible so that sorting is reliable.
    if "executedAt" in df.columns:
        df["executedAt"] = pd.to_datetime(df["executedAt"], errors="coerce")
        df = df.sort_values(["executedAt", "datasetId"], ascending=[False, True])

    # Save CSV summary if requested.
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] Wrote CSV: {out_csv}")

    # Save Parquet summary if requested.
    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        print(f"[OK] Wrote Parquet: {out_parquet}")

    return df