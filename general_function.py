import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
import zipfile

import geopandas as gpd
import numpy as np
import openpyxl
import pandas as pd

from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional, Iterable, Union

from reference import *

ColT = Union[str, int]


# =========================================================
# MinIO helpers
# =========================================================

def minio_bytes(path: str | os.PathLike) -> bytes:
    """
    Read one object from MinIO and return its raw bytes.
    """
    key = minio_key(path)
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=key)
    return obj["Body"].read()


def minio_text(path: str | os.PathLike, encoding: str = "utf-8") -> str:
    """
    Read one text object from MinIO.
    """
    return minio_bytes(path).decode(encoding)


def object_exists(bucket: str, key: str) -> bool:
    """
    Check whether an object already exists in the bucket.
    """
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def object_size_bytes(path: str | os.PathLike, bucket: str = LAKE_BUCKET) -> Optional[int]:
    """
    Return the object size in bytes from MinIO metadata.
    """
    try:
        resp = s3.head_object(Bucket=bucket, Key=minio_key(path))
        return resp.get("ContentLength")
    except Exception:
        return None


def uncompressed_zip_size(path: str | os.PathLike) -> Optional[int]:
    """
    Return the total uncompressed size of a ZIP stored in MinIO.
    """
    key = minio_key(path)
    if not key.lower().endswith(".zip"):
        return None

    try:
        raw = minio_bytes(path)
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            return sum(info.file_size for info in zf.infolist())
    except Exception:
        return None


# =========================================================
# Small binary/text guessing helpers
# =========================================================

def _sample_bytes(path: str, size: int = 256_000) -> bytes:
    """
    Read only the first bytes of a MinIO object.
    """
    key = minio_key(path)
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=key, Range=f"bytes=0-{size-1}")
    return obj["Body"].read()


def _guess_encoding(sample: bytes) -> str:
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"


def _guess_sep(sample_text: str):
    cands = [",", ";", "\t", "|"]
    counts = {sep: sample_text.count(sep) for sep in cands}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


# =========================================================
# Cache path
# =========================================================

def _cache_path(src_path: str | os.PathLike, ext: str = ".parquet") -> str:
    """
    Return a stable local cache file path.
    This cache is only for temporary runtime acceleration.
    """
    src_key = minio_key(src_path)
    os.makedirs(CACHE_DIR, exist_ok=True)

    size = object_size_bytes(src_key) or 0
    key = f"{src_key}|{size}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(str(CACHE_DIR), f"{h}{ext}")


def _is_cache_fresh(src_path: str | os.PathLike, cache: Path) -> bool:
    """
    Cache freshness check based on object size only.
    """
    try:
        return cache.exists() and cache.stat().st_size > 0
    except Exception:
        return False


# =========================================================
# Normalisation helpers
# =========================================================

def strip_accents(s: str):
    """Remove accents; normalise spaces/hyphens/apostrophes; fix common encoding issues."""
    if not isinstance(s, str):
        return s

    replacements = {
        "Ã©": "é",
        "Æ": "AE",
        "æ": "ae",
        "œ": "oe",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.replace("’", "'").replace("`", "'").replace("´", "'")
    s = re.sub(r"[-_\s]+", " ", s)

    return s.strip()


def norm_name(v):
    """Normalise a display name for robust comparison."""
    if pd.isna(v):
        return ""
    return strip_accents(str(v)).lower()


def norm_code(v):
    """Normalise a code for robust comparison (uppercase, trimmed)."""
    if pd.isna(v):
        return ""
    return str(v).strip().upper()


def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)


def is_string_series(s):
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def not_null_ratio(s):
    n = len(s)
    return float(s.notna().sum()) / n if n else 0.0


def normalise_colname(name: str) -> str:
    """Lightweight column-name normaliser for pattern matching and series alignment."""
    s = str(name).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# =========================================================
# Temporary local staging for geospatial readers
# =========================================================

def _suffix_from_key(path: str | os.PathLike) -> str:
    return Path(str(path)).suffix.lower()


def _stage_minio_object_to_tempfile(path: str | os.PathLike, suffix: Optional[str] = None) -> str:
    """
    Stage a MinIO object to a temporary local file for libraries that require a real file path.
    """
    raw = minio_bytes(path)
    suf = suffix or _suffix_from_key(path) or ".tmp"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name


# =========================================================
# Helpers for zip + /vsizip/
# =========================================================

def _first_member_with_suffix(zf: zipfile.ZipFile, *suffixes: str) -> str | None:
    """Return the first entry inside the zip matching any of the given suffixes (case-insensitive)."""
    suf = tuple(s.lower() for s in suffixes)
    for n in zf.namelist():
        ln = n.lower()
        if ln.startswith("__macosx/"):
            continue
        if ln.endswith(suf):
            return n
    return None


def _vsizip_path(zip_path: Path, inner_path: str) -> str:
    """Build a GDAL /vsizip/ path with forward slashes."""
    return f"/vsizip/{zip_path.as_posix()}/{inner_path}"


def _maybe_drop_geometry(gdf: gpd.GeoDataFrame, keep_geometry: bool = True):
    """Optionally drop active geometry to return a plain DataFrame."""
    if keep_geometry:
        return gdf
    return gdf.drop(columns=[gdf.geometry.name])


# =========================================================
# Excel helpers
# =========================================================

def _engine_for_excel(path: str) -> tuple[str | None, str | None]:
    ext = Path(path).suffix.lower()
    if ext in {".xlsx", ".xlsm"}:
        return "openpyxl", "pip install openpyxl"
    if ext == ".xls":
        return "xlrd", "pip install xlrd"
    if ext == ".xlsb":
        return "pyxlsb", "pip install pyxlsb"
    if ext == ".ods":
        return "odf", "pip install odfpy"
    return None, None


def _peek_block(path: str, sheet, engine: str | None, nrows: int = 1000) -> pd.DataFrame:
    raw = minio_bytes(path)
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None, nrows=nrows, engine=engine)


def _detect_header_row(block: pd.DataFrame) -> int | None:
    rc = block.count(axis=1)
    if rc.empty or rc.max() == 0:
        return None
    return int(rc.idxmax())


def _norm(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_description_sheet(sheet_name: str, header_row_values: list) -> bool:
    name_pat = re.compile(
        r"(description\s+(des\s+)?(données|variables)|"
        r"data\s+dictionary|dictionary|liste\s+des\s+variables|variables?)",
        re.IGNORECASE
    )
    if name_pat.search(str(sheet_name) or ""):
        return True

    headers = [_norm(x) for x in header_row_values]
    key_tokens = {"description", "détails", "définition", "definition", "explication"}
    var_tokens = {"variable", "champ", "field", "colonne", "column", "attribut"}

    if any(h in key_tokens for h in headers) and any(h in var_tokens for h in headers):
        return True

    if len(headers) in (2, 3) and any(h in var_tokens for h in headers) and any(h in key_tokens for h in headers):
        return True

    return False


def read_excel_workbook_with_desc_detection(file_path: str):
    """
    Read all non-empty sheets from an Excel workbook stored in MinIO.
    """
    engine, hint = _engine_for_excel(file_path)
    raw = minio_bytes(file_path)

    try:
        with pd.ExcelFile(io.BytesIO(raw), engine=engine) as xls:
            inspection = [classify_excel_sheet(raw, s, engine=engine, nrows=1000) for s in xls.sheet_names]
            results = []
            for i, info in enumerate(inspection):
                if info["sheet_role"] == "empty":
                    continue

                df = _read_excel_sheet_from_bytes(
                    raw,
                    sheet_name=info["sheet_name"],
                    header_row=info["header_row"],
                    engine=engine,
                )

                results.append({
                    "sheet_index": i,
                    "sheet_name": info["sheet_name"],
                    "df": df,
                    "header_row": info["header_row"],
                    "is_description": info["sheet_role"] == "dictionary",
                    "sheet_role": info["sheet_role"],
                })
            return results
    except ImportError as e:
        raise ImportError(f"Missing Excel engine for {Path(file_path).suffix.lower()}. Try: {hint}") from e


def count_str(df, num_row):
    row = df.iloc[num_row]
    return row.apply(lambda x: isinstance(x, str)).sum()


def _is_empty(x):
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip() == ""
    return False


def test_title_with_null(df, num_row, max_num):
    nb_null = max_num - count_str(df, num_row)
    row_first = df.iloc[num_row, :nb_null]
    empty_mask = row_first.apply(_is_empty)
    return empty_mask.all()


def _engine_for_excel(path: str) -> tuple[str | None, str | None]:
    ext = Path(path).suffix.lower()
    if ext in {".xlsx", ".xlsm"}:
        return "openpyxl", "pip install openpyxl"
    if ext == ".xls":
        return "xlrd", "pip install xlrd"
    if ext == ".xlsb":
        return "pyxlsb", "pip install pyxlsb"
    if ext == ".ods":
        return "odf", "pip install odfpy"
    return None, None


def _peek_block_from_bytes(raw: bytes, sheet, engine: str | None, nrows: int = 1000) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None, nrows=nrows, engine=engine)


def _detect_header_row(block: pd.DataFrame) -> int | None:
    block = block.dropna(how="all")
    if block.empty:
        return None

    best_row = None
    best_score = -1.0
    sample = block.head(min(len(block), 50))
    for idx, row in sample.iterrows():
        values = [_norm_text(v) for v in row.tolist()]
        non_empty = [v for v in values if v]
        if not non_empty:
            continue

        unique_ratio = len(set(non_empty)) / max(1, len(non_empty))
        text_ratio = sum(not _looks_numeric(v) for v in non_empty) / max(1, len(non_empty))
        next_rows = block.loc[idx:].iloc[1:6]
        next_fill = float(next_rows.notna().sum(axis=1).mean()) if not next_rows.empty else 0.0
        score = len(non_empty) + unique_ratio + text_ratio + min(next_fill, len(non_empty))

        if score > best_score:
            best_score = score
            best_row = idx

    if best_row is None:
        return None
    return int(best_row)


def _norm_text(x) -> str:
    if x is None or pd.isna(x):
        return ""
    s = strip_accents(str(x)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_numeric(value) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, float, np.number)):
        return True
    try:
        float(str(value).replace(",", ".").strip())
        return True
    except Exception:
        return False


def _excel_cache_path(src_path: str | os.PathLike, sheet, usecols=None) -> str:
    local_path = Path(src_path)
    if local_path.exists():
        src_key = str(local_path.resolve())
        size = local_path.stat().st_size
    else:
        src_key = minio_key(src_path)
        size = object_size_bytes(src_key) or 0

    os.makedirs(CACHE_DIR, exist_ok=True)

    token = json.dumps(
        {"src": src_key, "size": size, "sheet": sheet, "usecols": usecols},
        sort_keys=True,
        default=str,
    )
    h = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return os.path.join(str(CACHE_DIR), f"{h}.parquet")


def _parquet_columns_arg(usecols):
    return usecols if isinstance(usecols, list) else None


def _excel_source_bytes(file_path) -> bytes:
    local_path = Path(file_path)
    if local_path.exists():
        return local_path.read_bytes()
    return minio_bytes(file_path)


def _xlsx_with_safe_stylesheet(raw: bytes) -> bytes:
    safe_styles = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>
  <fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium9" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>"""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw), "r") as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        style_written = False
        for item in src.infolist():
            if item.filename == "xl/styles.xml":
                dst.writestr(item, safe_styles)
                style_written = True
            else:
                dst.writestr(item, src.read(item.filename))
        if not style_written:
            dst.writestr("xl/styles.xml", safe_styles)
    return out.getvalue()


def _open_excel_file(raw: bytes, engine: str | None):
    try:
        return pd.ExcelFile(io.BytesIO(raw), engine=engine), raw
    except ValueError as e:
        msg = str(e).lower()
        if engine == "openpyxl" and ("stylesheet" in msg or "style" in msg):
            fixed_raw = _xlsx_with_safe_stylesheet(raw)
            return pd.ExcelFile(io.BytesIO(fixed_raw), engine=engine), fixed_raw
        raise


def _sheet_name_signal(sheet_name: str) -> str | None:
    s = _norm_text(sheet_name)

    dictionary_patterns = [
        r"description",
        r"description des",
        r"dictionary",
        r"data dictionary",
        r"metadata",
        r"variables",
        r"liste des variables",
        r"dictionnaire",
        r"dictionnaire de données",
        r"nomenclature",
        r"lexique",
        r"glossaire",
    ]

    dashboard_patterns = [
        r"dashboard",
        r"tableau de bord",
        r"synthese",
        r"synthèse",
        r"resume",
        r"résumé",
        r"indicateurs clés",
        r"kpi",
        r"overview",
    ]

    for pat in dictionary_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return "dictionary"

    for pat in dashboard_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return "dashboard"

    return None


def _classify_from_headers_and_content(block: pd.DataFrame, header_row: int | None) -> str:
    """
    Return one of:
    - data
    - dictionary
    - dashboard
    - empty
    - unknown
    """
    block = block.dropna(how="all")
    if block.empty:
        return "empty"

    if header_row is None or header_row not in block.index:
        return "unknown"

    header_values = [_norm_text(x) for x in block.loc[header_row].tolist()]
    non_empty_headers = [x for x in header_values if x]

    # --- dictionary signal ---
    dict_key_tokens = {
        "description", "definition", "définition", "explication", "details", "détails",
        "modalites", "modalités", "format", "type", "unite", "unité"
    }
    dict_var_tokens = {
        "variable", "variables", "champ", "field", "column", "colonne", "attribut", "code", "libelle", "libellé"
    }

    if non_empty_headers:
        has_dict_key = any(h in dict_key_tokens for h in non_empty_headers)
        has_dict_var = any(h in dict_var_tokens for h in non_empty_headers)
        if has_dict_key and has_dict_var:
            return "dictionary"

        if len(non_empty_headers) <= 4 and has_dict_var:
            joined = " ".join(non_empty_headers)
            if any(tok in joined for tok in dict_key_tokens):
                return "dictionary"

    # --- dashboard signal ---
    # 特征：有效列少、上方标题很多、数字摘要多、不是规则表头
    n_cols = block.shape[1]
    n_rows = block.shape[0]

    preview = block.head(min(15, n_rows)).copy()

    non_null_ratio = preview.notna().sum().sum() / max(1, preview.shape[0] * preview.shape[1])

    # header row 附近若大多为空而前几行像标题块，常见于 dashboard
    if header_row is not None and header_row > 3:
        title_zone = block.iloc[:header_row]
        if not title_zone.empty:
            text_cells = sum(
                isinstance(v, str) and str(v).strip() != ""
                for v in title_zone.to_numpy().flatten()
            )
            if text_cells >= 3 and len(non_empty_headers) <= 6:
                return "dashboard"

    # --- data signal ---
    # 特征：header 比较结构化，列数较多，后续行较规则
    if len(non_empty_headers) >= 3:
        header_pos = block.index.get_loc(header_row)
        data_part = block.iloc[header_pos + 1: header_pos + 11] if header_pos + 1 < len(block) else pd.DataFrame()
        if not data_part.empty:
            row_fill = data_part.notna().sum(axis=1)
            if len(row_fill) > 0:
                avg_fill = row_fill.mean()
                if avg_fill >= max(2, len(non_empty_headers) * 0.4):
                    return "data"

    # 小而规则的两列表，优先视为 dictionary
    if len(non_empty_headers) in {2, 3}:
        joined = " ".join(non_empty_headers)
        if any(tok in joined for tok in ["variable", "champ", "field", "column", "colonne", "description", "definition"]):
            return "dictionary"

    # 否则如果整体比较表格化，给 data
    if n_cols >= 3 and non_null_ratio >= 0.2:
        return "data"

    return "unknown"


def _sheet_name_signal(sheet_name: str) -> str | None:
    s = _norm_text(sheet_name)

    dictionary_patterns = [
        r"description",
        r"description des",
        r"dictionary",
        r"data dictionary",
        r"metadata",
        r"variables",
        r"liste des variables",
        r"dictionnaire",
        r"dictionnaire de donnees",
        r"nomenclature",
        r"lexique",
        r"glossaire",
    ]

    dashboard_patterns = [
        r"dashboard",
        r"tableau de bord",
        r"synthese",
        r"resume",
        r"indicateurs cles",
        r"kpi",
        r"overview",
    ]

    for pat in dictionary_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return "dictionary"

    for pat in dashboard_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return "dashboard"

    return None


def _classify_from_headers_and_content(block: pd.DataFrame, header_row: int | None) -> str:
    """
    Return one of: data, dictionary, dashboard, empty, unknown.
    """
    block = block.dropna(how="all")
    if block.empty:
        return "empty"

    if header_row is None or header_row not in block.index:
        return "unknown"

    header_values = [_norm_text(x) for x in block.loc[header_row].tolist()]
    non_empty_headers = [x for x in header_values if x]

    dict_key_tokens = {
        "description", "definition", "explication", "details",
        "modalites", "format", "type", "unite"
    }
    dict_var_tokens = {
        "variable", "variables", "champ", "field", "column",
        "colonne", "attribut", "code", "libelle"
    }

    if non_empty_headers:
        has_dict_key = any(h in dict_key_tokens for h in non_empty_headers)
        has_dict_var = any(h in dict_var_tokens for h in non_empty_headers)
        if has_dict_key and has_dict_var:
            return "dictionary"

        if len(non_empty_headers) <= 4 and has_dict_var:
            joined = " ".join(non_empty_headers)
            if any(tok in joined for tok in dict_key_tokens):
                return "dictionary"

    n_cols = block.shape[1]
    n_rows = block.shape[0]
    preview = block.head(min(15, n_rows)).copy()
    non_null_ratio = preview.notna().sum().sum() / max(1, preview.shape[0] * preview.shape[1])

    if header_row is not None and header_row > 3:
        title_zone = block.loc[:header_row].iloc[:-1]
        if not title_zone.empty:
            text_cells = sum(
                isinstance(v, str) and str(v).strip() != ""
                for v in title_zone.to_numpy().flatten()
            )
            if text_cells >= 3 and len(non_empty_headers) <= 6:
                return "dashboard"

    if len(non_empty_headers) >= 3:
        header_pos = block.index.get_loc(header_row)
        data_part = block.iloc[header_pos + 1: header_pos + 11] if header_pos + 1 < len(block) else pd.DataFrame()
        if not data_part.empty:
            row_fill = data_part.notna().sum(axis=1)
            if len(row_fill) > 0 and row_fill.mean() >= max(2, len(non_empty_headers) * 0.4):
                return "data"

    if len(non_empty_headers) in {2, 3}:
        joined = " ".join(non_empty_headers)
        if any(tok in joined for tok in ["variable", "champ", "field", "column", "colonne", "description", "definition"]):
            return "dictionary"

    if n_cols >= 3 and non_null_ratio >= 0.2:
        return "data"

    return "unknown"


def classify_excel_sheet(raw: bytes, sheet_name: str, engine: str | None, nrows: int = 1000) -> dict:
    """
    Inspect one sheet and return its classification.
    """
    block = _peek_block_from_bytes(raw, sheet_name, engine=engine, nrows=nrows)
    non_empty_block = block.dropna(how="all")
    if non_empty_block.empty:
        return {
            "sheet_name": sheet_name,
            "sheet_role": "empty",
            "header_row": None,
            "nrows_preview": 0,
            "ncols_preview": 0,
        }

    header_row = _detect_header_row(block)

    name_role = _sheet_name_signal(sheet_name)
    if name_role is not None:
        content_role = _classify_from_headers_and_content(block, header_row)
        role = content_role if content_role == "data" else name_role
    else:
        role = _classify_from_headers_and_content(block, header_row)

    return {
        "sheet_name": sheet_name,
        "sheet_role": role,
        "header_row": header_row,
        "nrows_preview": int(non_empty_block.shape[0]),
        "ncols_preview": int(non_empty_block.shape[1]),
    }


def inspect_excel_workbook(file_path: str, nrows: int = 1000) -> list[dict]:
    """
    Return sheet inspection results for the workbook.
    """
    engine, hint = _engine_for_excel(file_path)
    raw = _excel_source_bytes(file_path)

    try:
        xls, raw = _open_excel_file(raw, engine)
        with xls:
            results = []
            for sheet_name in xls.sheet_names:
                info = classify_excel_sheet(raw, sheet_name, engine=engine, nrows=nrows)
                results.append(info)
            return results
    except ImportError as e:
        raise ImportError(f"Missing Excel engine for {Path(file_path).suffix.lower()}. Try: {hint}") from e


def _read_excel_sheet_from_bytes(
    raw: bytes,
    sheet_name,
    header_row: int | None,
    engine: str | None,
    usecols=None,
) -> pd.DataFrame:
    return pd.read_excel(
        io.BytesIO(raw),
        sheet_name=sheet_name,
        header=header_row if header_row is not None else 0,
        engine=engine,
        usecols=usecols,
    )


def _read_data_sheets_from_bytes(
    raw: bytes,
    data_sheets: list[dict],
    engine: str | None,
    usecols=None,
    cache_parquet: bool = False,
    file_path=None,
) -> dict[str, pd.DataFrame]:
    out = {}
    for info in data_sheets:
        current_sheet = info["sheet_name"]
        pq = _excel_cache_path(file_path, current_sheet, usecols) if cache_parquet and file_path else None

        if pq and _is_cache_fresh(file_path, Path(pq)):
            try:
                out[current_sheet] = pd.read_parquet(pq, columns=_parquet_columns_arg(usecols))
                continue
            except Exception:
                pass

        df = _read_excel_sheet_from_bytes(
            raw,
            sheet_name=current_sheet,
            header_row=info["header_row"],
            engine=engine,
            usecols=usecols,
        )
        out[current_sheet] = df

        if pq:
            try:
                df.to_parquet(pq, index=False)
            except Exception:
                pass

    return out


def excel_EL(
    file_path,
    sheet_name="auto",
    usecols=None,
    cache_parquet: bool = True,
    return_sheet_info: bool = False
):
    """
    Read an Excel workbook from MinIO with automatic data-sheet detection.

    Parameters
    ----------
    file_path : str
        MinIO object key or logical input path.
    sheet_name : str | int | None
        - "auto": return the only data sheet, or a dict when several data sheets are detected
        - "first_data": choose the first detected data sheet
        - "all_data": return dict of all detected data sheets
        - int / str: read one specific sheet
    return_sheet_info : bool
        If True, also return workbook sheet inspection results.
    """
    engine, hint = _engine_for_excel(file_path)
    raw = _excel_source_bytes(file_path)

    try:
        xls, raw = _open_excel_file(raw, engine)
        with xls:
            sheet_names = xls.sheet_names

            inspection = [classify_excel_sheet(raw, s, engine=engine, nrows=1000) for s in sheet_names]
            data_sheets = [x for x in inspection if x["sheet_role"] == "data"]

            # Resolve target sheet(s)
            if sheet_name == "auto":
                if not data_sheets:
                    raise ValueError(f"No data sheet detected in workbook: {file_path}")
                if len(data_sheets) > 1:
                    out = _read_data_sheets_from_bytes(
                        raw,
                        data_sheets,
                        engine=engine,
                        usecols=usecols,
                        cache_parquet=cache_parquet,
                        file_path=file_path,
                    )
                    if return_sheet_info:
                        return out, inspection
                    return out
                target_info = data_sheets[0]

            elif sheet_name == "all_data":
                if not data_sheets:
                    raise ValueError(f"No data sheet detected in workbook: {file_path}")
                out = _read_data_sheets_from_bytes(
                    raw,
                    data_sheets,
                    engine=engine,
                    usecols=usecols,
                    cache_parquet=cache_parquet,
                    file_path=file_path,
                )
                if return_sheet_info:
                    return out, inspection
                return out

            elif sheet_name == "first_data":
                if not data_sheets:
                    raise ValueError(f"No data sheet detected in workbook: {file_path}")
                target_info = data_sheets[0]

            elif isinstance(sheet_name, int):
                if not sheet_names:
                    raise ValueError("Workbook has no sheets.")
                idx = max(0, min(sheet_name, len(sheet_names) - 1))
                target_sheet = sheet_names[idx]
                info = next((x for x in inspection if x["sheet_name"] == target_sheet), None)
                target_info = info if info else {"sheet_name": target_sheet, "header_row": 0}

            else:
                target_sheet = sheet_name
                info = next((x for x in inspection if x["sheet_name"] == target_sheet), None)
                if info is None:
                    raise ValueError(f"Sheet '{target_sheet}' not found in workbook.")
                target_info = info

            target_sheet = target_info["sheet_name"]
            header_row = target_info["header_row"]

    except ImportError as e:
        raise ImportError(f"Missing Excel engine for {Path(file_path).suffix.lower()}. Try: {hint}") from e

    pq = _excel_cache_path(file_path, target_sheet, usecols)
    if cache_parquet and _is_cache_fresh(file_path, Path(pq)):
        try:
            df = pd.read_parquet(pq, columns=_parquet_columns_arg(usecols))
            if return_sheet_info:
                return df, inspection
            return df
        except Exception:
            pass

    df = _read_excel_sheet_from_bytes(
        raw,
        sheet_name=target_sheet,
        header_row=header_row,
        engine=engine,
        usecols=usecols,
    )

    if cache_parquet:
        try:
            df.to_parquet(pq, index=False)
        except Exception:
            pass

    if return_sheet_info:
        return df, inspection
    return df


# =========================================================
# Structured readers
# =========================================================

def csv_EL(file_path):
    raw = minio_bytes(file_path)
    encs = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    seps = [None, ";", ",", "\t", "|"]

    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, low_memory=False, on_bad_lines="skip")
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    raise ValueError(f"Could not parse {file_path} as CSV/TSV.")


def geojson_EL(file_path):
    tmp = _stage_minio_object_to_tempfile(file_path, suffix=".geojson")
    try:
        return gpd.read_file(tmp)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def json_EL(file_path):
    raw = minio_bytes(file_path)
    try:
        return pd.read_json(io.BytesIO(raw), lines=True)
    except ValueError:
        data = json.loads(raw.decode("utf-8-sig"))
        if isinstance(data, dict):
            data = [data]
        return pd.json_normalize(data)


def parquet_EL(file_path):
    raw = minio_bytes(file_path)
    return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")


def detect_json_type(filepath):
    raw = minio_bytes(filepath)
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, dict) and "type" in data:
        geo_types = {"FeatureCollection", "Feature", "Point", "Polygon", "MultiPolygon", "LineString"}
        if data["type"] in geo_types:
            return geojson_EL(filepath)
    return json_EL(filepath)


# =========================================================
# Geospatial readers
# =========================================================

def shapefile_EL(zip_file_path: str):
    tmp = _stage_minio_object_to_tempfile(zip_file_path, suffix=Path(zip_file_path).suffix.lower() or ".zip")
    try:
        p = Path(tmp).resolve()

        if p.suffix.lower() == ".shp":
            return gpd.read_file(p.as_posix())

        if p.suffix.lower() != ".zip":
            raise ValueError("shapefile_EL expects a .zip (or .shp) path.")

        with zipfile.ZipFile(p, "r") as zf:
            shp_inside = _first_member_with_suffix(zf, ".shp")
            if not shp_inside:
                raise FileNotFoundError("No .shp file found inside the zip.")

        vsip = _vsizip_path(p, shp_inside)
        return gpd.read_file(vsip)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def fgb_EL(path: str):
    tmp = _stage_minio_object_to_tempfile(path, suffix=Path(path).suffix.lower() or ".fgb")
    try:
        p = Path(tmp).resolve()

        if p.suffix.lower() == ".fgb":
            return gpd.read_file(p.as_posix())

        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as zf:
                fgb_inside = _first_member_with_suffix(zf, ".fgb")
                if not fgb_inside:
                    raise FileNotFoundError("No .fgb file found inside the zip.")
            vsip = _vsizip_path(p, fgb_inside)
            return gpd.read_file(vsip)

        raise ValueError("fgb_EL expects a .fgb or .zip path.")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def _list_layers(pathlike: str) -> list[str]:
    try:
        from pyogrio import list_layers
        return [l[0] for l in list_layers(pathlike)]
    except Exception:
        try:
            import fiona
            return fiona.listlayers(pathlike)
        except Exception as e:
            raise RuntimeError(f"Cannot list layers from: {pathlike}. Error: {e}")


def _concat_layers(path_like: str, layers: list[str]) -> gpd.GeoDataFrame:
    frames = []
    for lyr in layers:
        sub = gpd.read_file(path_like, layer=lyr)
        sub["src_layer"] = lyr
        frames.append(sub)
    if not frames:
        raise ValueError("No readable layers found.")
    return pd.concat(frames, ignore_index=True)


def kml_EL(path: str, layer: str | None = None):
    tmp = _stage_minio_object_to_tempfile(path, suffix=Path(path).suffix.lower() or ".kml")
    try:
        p = Path(tmp).resolve()

        if p.suffix.lower() == ".kml":
            layers = _list_layers(p.as_posix())
            if layer is None:
                return _concat_layers(p.as_posix(), layers)
            if layer not in layers:
                raise ValueError(f"Layer '{layer}' not found. Available: {layers}")
            return gpd.read_file(p.as_posix(), layer=layer)

        if p.suffix.lower() == ".kmz":
            with zipfile.ZipFile(p, "r") as zf:
                kml_inside = _first_member_with_suffix(zf, ".kml")
                if not kml_inside:
                    raise FileNotFoundError("No .kml file found inside the KMZ.")
            vsip = _vsizip_path(p, kml_inside)
            layers = _list_layers(vsip)
            if layer is None:
                return _concat_layers(vsip, layers)
            if layer not in layers:
                raise ValueError(f"Layer '{layer}' not found. Available: {layers}")
            return gpd.read_file(vsip, layer=layer)

        raise ValueError("kml_EL expects a .kml or .kmz path.")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def gpx_EL(path: str, layer: str | None = None):
    tmp = _stage_minio_object_to_tempfile(path, suffix=Path(path).suffix.lower() or ".gpx")
    try:
        p = Path(tmp).resolve()

        def _read_from_pathlike(pathlike: str) -> gpd.GeoDataFrame:
            layers = _list_layers(pathlike)
            if layer is None:
                wanted = [l for l in ("tracks", "routes", "waypoints") if l in layers]
                if not wanted:
                    wanted = layers
                return _concat_layers(pathlike, wanted)
            if layer not in layers:
                raise ValueError(f"Layer '{layer}' not found. Available: {layers}")
            return gpd.read_file(pathlike, layer=layer)

        if p.suffix.lower() == ".gpx":
            return _read_from_pathlike(p.as_posix())

        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as zf:
                gpx_inside = _first_member_with_suffix(zf, ".gpx")
                if not gpx_inside:
                    raise FileNotFoundError("No .gpx file found inside the zip.")
            vsip = _vsizip_path(p, gpx_inside)
            return _read_from_pathlike(vsip)

        raise ValueError("gpx_EL expects a .gpx or .zip path.")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def zip_geodata_EL(zip_path: str):
    tmp = _stage_minio_object_to_tempfile(zip_path, suffix=".zip")
    try:
        p = Path(tmp).resolve()
        if p.suffix.lower() != ".zip":
            raise ValueError("zip_geodata_EL expects a .zip path.")

        with zipfile.ZipFile(p, "r") as zf:
            inner = _first_member_with_suffix(zf, ".shp")
            if inner:
                return gpd.read_file(_vsizip_path(p, inner))

            inner = _first_member_with_suffix(zf, ".fgb")
            if inner:
                return gpd.read_file(_vsizip_path(p, inner))

            inner = _first_member_with_suffix(zf, ".kml")
            if inner:
                kml_vsip = _vsizip_path(p, inner)
                layers = _list_layers(kml_vsip)
                return _concat_layers(kml_vsip, layers)

            inner = _first_member_with_suffix(zf, ".gpx")
            if inner:
                gpx_vsip = _vsizip_path(p, inner)
                layers = _list_layers(gpx_vsip)
                wanted = [l for l in ("tracks", "routes", "waypoints") if l in layers] or layers
                return _concat_layers(gpx_vsip, wanted)

        raise FileNotFoundError("No supported geodata found inside the zip (.shp/.fgb/.kml/.gpx).")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# =========================================================
# Extension maps
# =========================================================

dict_EL = {
    ".xlsx": excel_EL,
    ".xls": excel_EL,
    ".csv": csv_EL,
    ".tsv": csv_EL,
    ".parquet": parquet_EL,
    ".geojson": geojson_EL,
    ".json": detect_json_type,
    ".shp": shapefile_EL,
    ".zip": zip_geodata_EL,
    ".kml": kml_EL,
    ".kmz": kml_EL,
    ".fgb": fgb_EL,
    ".gpx": gpx_EL,
}

dict_category = {
    ".xlsx": "structured",
    ".xls": "structured",
    ".csv": "structured",
    ".tsv": "structured",
    ".parquet": "structured",
    ".geojson": "semi-structured",
    ".json": "semi-structured",
    ".shp": "semi-structured",
    ".zip": "semi-structured",
    ".kml": "semi-structured",
    ".kmz": "semi-structured",
    ".fgb": "semi-structured",
    ".gpx": "semi-structured",
}


# =========================================================
# Unified readers
# =========================================================

def get_df(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in dict_EL:
        df = dict_EL[ext](str(path))
        return df, ext, dict_category[ext]
    else:
        return None, "unknown", "unknown"


def _resolve_excel_target_sheet(file_path: str, sheet) -> tuple[object, str]:
    engine, _ = _engine_for_excel(file_path)
    raw = minio_bytes(file_path)
    with pd.ExcelFile(io.BytesIO(raw), engine=engine) as xls:
        if sheet in (None, "first", "auto"):
            target = xls.sheet_names[0] if xls.sheet_names else 0
        elif isinstance(sheet, int):
            if not xls.sheet_names:
                raise ValueError("Workbook has no sheets.")
            idx = max(0, min(sheet, len(xls.sheet_names) - 1))
            target = xls.sheet_names[idx]
        else:
            target = sheet
    return target, engine


def _detect_excel_header(file_path: str, target_sheet, engine: str, preview_n: int = 300) -> int:
    raw = minio_bytes(file_path)
    block = pd.read_excel(io.BytesIO(raw), sheet_name=target_sheet, header=None, nrows=preview_n, engine=engine)
    block = block.dropna(how="all")
    if block.empty:
        return 0
    rc = block.count(axis=1)
    return int(rc.idxmax()) if not rc.empty else 0


def read_head_any(
    path: str,
    *,
    sheet: int | str | None = "auto",
    nrows: int = 20,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read the first nrows from a MinIO object.
    """
    ext = Path(path).suffix.lower()

    if ext in {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"}:
        raw = minio_bytes(path)
        target, engine = _resolve_excel_target_sheet(path, sheet)
        header = _detect_excel_header(path, target, engine, preview_n=max(50, nrows))
        return pd.read_excel(io.BytesIO(raw), sheet_name=target, header=header, nrows=nrows, usecols=columns, engine=engine)

    if ext in {".csv", ".tsv"}:
        raw = minio_bytes(path)
        encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        seps = [None, ",", ";", "\t", "|"]
        for enc in encs:
            for sep in seps:
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw),
                        usecols=columns,
                        sep=sep,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines="skip"
                    )
                    if df.shape[1] > 1:
                        return df.head(nrows)
                except Exception:
                    pass
        raise ValueError(f"Could not parse {path} as CSV/TSV.")

    if ext == ".parquet":
        raw = minio_bytes(path)
        df = pd.read_parquet(io.BytesIO(raw), engine="pyarrow", columns=columns)
        return df.head(nrows)

    if ext == ".json":
        raw = minio_bytes(path)
        try:
            df = pd.read_json(io.BytesIO(raw), lines=True)
        except ValueError:
            data = json.loads(raw.decode("utf-8-sig"))
            if isinstance(data, dict):
                data = [data]
            df = pd.json_normalize(data[:nrows])
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return df.head(nrows)

    if ext in {".geojson", ".shp", ".kml", ".kmz", ".gpx", ".fgb", ".zip"}:
        df, _, _ = get_df(path)
        if hasattr(df, "geometry"):
            df = pd.DataFrame(df.drop(columns=[df.geometry.name], errors="ignore"))
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return df.head(nrows)

    df, _, _ = get_df(path)
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return df.head(nrows)


def _map_pos_to_names(names: List[str], cols: List[ColT]) -> List[str]:
    out: List[str] = []
    for c in cols:
        if isinstance(c, int):
            if 0 <= c < len(names):
                out.append(names[c])
        else:
            out.append(str(c))
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def read_cols_full(
    path: str,
    *,
    sheet: int | str | None = "auto",
    columns: Optional[List[ColT]] = None,
) -> pd.DataFrame:
    """
    Read the full MinIO object but only the requested columns where possible.
    """
    if columns is not None and len(columns) == 0:
        return pd.DataFrame()

    ext = Path(path).suffix.lower()

    if ext in {".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"}:
        raw = minio_bytes(path)
        target, engine = _resolve_excel_target_sheet(path, sheet)
        header = _detect_excel_header(path, target, engine, preview_n=300)
        return pd.read_excel(io.BytesIO(raw), sheet_name=target, header=header, usecols=columns, engine=engine)

    if ext in {".csv", ".tsv"}:
        raw = minio_bytes(path)
        encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        seps = [None, ",", ";", "\t", "|"]
        for enc in encs:
            for sep in seps:
                try:
                    return pd.read_csv(
                        io.BytesIO(raw),
                        usecols=columns,
                        sep=sep,
                        encoding=enc,
                        low_memory=False,
                        on_bad_lines="skip"
                    )
                except Exception:
                    pass
        raise ValueError(f"Could not parse {path} as CSV/TSV.")

    if ext == ".parquet":
        raw = minio_bytes(path)
        if columns is None:
            return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
        if any(isinstance(c, int) for c in columns):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(io.BytesIO(raw))
            schema_names = pf.schema_arrow.names
            colnames = _map_pos_to_names(schema_names, list(columns))
            return pd.read_parquet(io.BytesIO(raw), engine="pyarrow", columns=colnames)
        else:
            return pd.read_parquet(io.BytesIO(raw), engine="pyarrow", columns=list(columns))

    if ext == ".json":
        raw = minio_bytes(path)
        try:
            df = pd.read_json(io.BytesIO(raw), lines=True)
        except ValueError:
            data = json.loads(raw.decode("utf-8-sig"))
            if isinstance(data, dict):
                data = [data]
            df = pd.json_normalize(data)
        if columns is None:
            return df
        names = list(df.columns)
        keep = _map_pos_to_names(names, list(columns))
        keep = [c for c in keep if c in df.columns]
        return df[keep]

    if ext in {".geojson", ".shp", ".kml", ".kmz", ".gpx", ".fgb", ".zip"}:
        df, _, _ = get_df(path)
        if hasattr(df, "geometry"):
            df = pd.DataFrame(df.drop(columns=[df.geometry.name], errors="ignore"))
        if columns is None:
            return df
        names = list(df.columns)
        sel = _map_pos_to_names(names, list(columns))
        keep = [c for c in sel if c in df.columns]
        return df[keep]

    df, _, _ = get_df(path)
    if columns is None:
        return df
    names = list(df.columns)
    sel = _map_pos_to_names(names, list(columns))
    keep = [c for c in sel if c in df.columns]
    return df[keep]


def read_head_with_meta(
    path: str,
    *,
    sheet: int | str | None = "auto",
    nrows: int = 5,
    columns: list[str] | None = None,
):
    """
    Read first nrows and also return ext/category.
    Returns: (df, ext, category)
    """
    df = read_head_any(path, sheet=sheet, nrows=nrows, columns=columns)
    ext = os.path.splitext(str(path))[1].lower()
    cat = dict_category.get(ext, "unknown")
    return df, ext, cat


# =========================================================
# Hierarchy helpers
# =========================================================

def level_matches(level, granularities) -> bool:
    if isinstance(level, tuple):
        return any(alias in granularities for alias in level)
    return level in granularities


def level_name(level) -> str:
    return level[0] if isinstance(level, tuple) else level


def get_most_general_in_path(granularities, path) -> Union[str, None]:
    for level in path:
        if level_matches(level, granularities):
            return level_name(level)
    return None


def get_most_specific_in_path(granularities, path) -> Union[str, None]:
    for level in reversed(path):
        if level_matches(level, granularities):
            return level_name(level)
    return None


# =========================================================
# Size / MinIO object movement helpers
# =========================================================

def human_readable_size(num_bytes: Optional[int]) -> Optional[str]:
    if num_bytes is None:
        return None
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def upload_local_file_to_rawzone(local_path: str, raw_zone_path: str, bucket: str = LAKE_BUCKET) -> None:
    """
    Upload one local file to the final raw-zone key in MinIO.
    """
    s3.upload_file(local_path, bucket, raw_zone_path)


def copy_input_object_to_rawzone(
    input_relative_path: str,
    raw_zone_path: str,
    bucket: str = LAKE_BUCKET
) -> bool:
    """
    Copy one staged object from input_data/ to its final raw-zone location.

    Parameters
    ----------
    input_relative_path:
        Relative path under the input area, for example
        'traffic/SIREDO_2010_1.csv' or 'SIREDO_2010_1.csv'.

    raw_zone_path:
        Final bucket-internal object key.

    Returns
    -------
    bool
        True if copied during this call, False if target already existed.
    """
    if object_exists(bucket, raw_zone_path):
        return False

    rel = str(input_relative_path).replace("\\", "/").lstrip("/")
    source_key = f"{INPUT_PREFIX}/{rel}"
    copy_source = {"Bucket": bucket, "Key": source_key}
    s3.copy_object(Bucket=bucket, CopySource=copy_source, Key=raw_zone_path)
    return True


def delete_input_object(
    input_relative_path: str,
    bucket: str = LAKE_BUCKET
) -> None:
    """
    Delete the staged object from input_data after a successful copy.
    """
    rel = str(input_relative_path).replace("\\", "/").lstrip("/")
    source_key = f"{INPUT_PREFIX}/{rel}"
    s3.delete_object(Bucket=bucket, Key=source_key)
