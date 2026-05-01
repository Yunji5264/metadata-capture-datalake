import os
import io
import json
import re
import time
import zipfile
import pandas as pd

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from collections.abc import Iterable

from uml_class import *
from scope_detector import *
from granularity_detector import *
from theme_detector import collect_all_themes_set
from semantic_helper import semantic_helper
from attribute_classifier import classify_attributes_with_semantic_helper, find_geometry_columns
from general_function import get_df, human_readable_size

from reference import (
    HIER,
    REF_SEMANTIC,
    PERF_DIR,
    DATASET_SOURCE_REGISTRY,
    read_excel_from_minio,
    read_json_from_minio,
    object_exists,
    minio_key,
    s3,
    LAKE_BUCKET
)

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


# =========================================================
# MinIO object helpers
# =========================================================

def _get_object_head(path_obj) -> Optional[dict]:
    """
    Return the MinIO object metadata if the object exists, else None.
    """
    try:
        return s3.head_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
    except Exception:
        return None


def _get_object_size_bytes(path_obj) -> Optional[int]:
    """
    Return object size in bytes from MinIO metadata.
    """
    head = _get_object_head(path_obj)
    if not head:
        return None
    return head.get("ContentLength")


def _uncompressed_zip_size_from_minio(path_obj) -> Optional[int]:
    """
    Return the total uncompressed size of a ZIP object stored in MinIO.

    This avoids assuming that the ZIP file exists on the local filesystem.
    """
    key = minio_key(path_obj)
    if not key.lower().endswith(".zip"):
        return None

    try:
        obj = s3.get_object(Bucket=LAKE_BUCKET, Key=key)
        raw = obj["Body"].read()
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            return sum(info.file_size for info in zf.infolist())
    except Exception:
        return None


# =========================================================
# Geometry helpers
# =========================================================

def _collect_geoms_from_geometry_column(df: pd.DataFrame, col: str) -> List[BaseGeometry]:
    """Collect shapely geometries from a geometry-like column."""
    s = df[col]
    return [g for g in s if g is not None]


def _collect_points_from_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> List[Point]:
    """Build shapely Points from numeric latitude/longitude columns."""
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    pts = []
    for x, y in zip(lon, lat):
        if pd.notna(x) and pd.notna(y):
            pts.append(Point(float(x), float(y)))
    return pts


def _aggregate_geometry(
    geoms: Iterable[BaseGeometry],
    *,
    method: str = "envelope",
    buffer_m: float = 0.0
) -> Optional[BaseGeometry]:
    """
    Aggregate a set of geometries into one geometry extent.

    Supported methods:
      - envelope
      - convex_hull
      - union
    """
    geoms = [g for g in geoms if g is not None]
    if not geoms:
        return None

    merged = unary_union(geoms)

    if method == "envelope":
        agg = merged.envelope
    elif method == "convex_hull":
        agg = merged.convex_hull
    elif method == "union":
        agg = merged
    else:
        agg = merged.envelope

    if buffer_m and buffer_m > 0:
        try:
            agg = agg.buffer(buffer_m)
        except Exception:
            pass

    return agg


def _geometry_values_as_wkt(geom: Optional[BaseGeometry]) -> List[str]:
    """Convert an aggregated geometry into the WKT list expected by DSSpatialScope."""
    return [geom.wkt] if geom is not None else []


# =========================================================
# Result-to-Dataset transformation
# =========================================================

def transform_result(
    dataset: Dataset,
    atts_spatial,
    atts_temporal,
    atts_indicator,
    atts_other
) -> Dataset:
    """
    Map classified attributes into the Dataset object.

    Design choices:
    - multi-column attributes keep their column list
    - themeName keeps the full hierarchical path
    - 'other' attributes are stored as ComplementaryInformation
    """

    def _name_preserve_list(cols: Union[str, List[str], None]) -> Union[str, List[str]]:
        if cols is None:
            return ""
        if isinstance(cols, list):
            return [str(c) for c in cols]
        return str(cols)

    def _dtype_from_type(tp: Any) -> Union[str, List[str]]:
        if isinstance(tp, list) and tp:
            return [str(t) for t in tp]
        return str(tp) if tp is not None else "object"

    def _text(val: Any, default: str = "") -> str:
        return str(val) if val is not None else default

    def _granularity(entry: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
        g = entry.get("granularity", default)
        return str(g) if g is not None else None

    def _indicator_type(entry: Dict[str, Any]) -> Optional[str]:
        it = entry.get("indicatorType", entry.get("indicator_type"))
        return str(it) if it is not None else None

    def _normalise_path(name: str) -> str:
        """Normalize a hierarchical theme path to the form 'A > B > C'."""
        parts = re.split(r'>|/|\\|→|»', str(name))
        tokens = [p.strip() for p in parts if p and p.strip()]
        return " > ".join(tokens) if tokens else str(name).strip()

    def _theme(entry: Dict[str, Any]) -> Optional[Theme]:
        """Convert theme payloads from multiple possible formats into a Theme object."""
        th = entry.get("theme")
        if th is None:
            return None

        if isinstance(th, Theme):
            normalised = _normalise_path(th.themeName)
            return Theme(normalised, th.themeDescription)

        if isinstance(th, dict):
            name = th.get("themeName") or th.get("name") or th.get("theme_name") or th.get("title")
            desc = th.get("themeDescription") or th.get("description") or th.get("desc") or th.get("theme_description")
            if isinstance(name, str):
                normalised = _normalise_path(name)
                final_desc = desc if desc is not None else str(name)
                return Theme(normalised, str(final_desc))
            return Theme("Theme", str(desc) if desc is not None else "")

        if isinstance(th, str):
            normalised = _normalise_path(th)
            return Theme(normalised, th)

        return None

    # Spatial attributes
    for att in atts_spatial:
        dataset.add_spatial_parameter(
            _name_preserve_list(att.get("columns")),
            _text(att.get("description")),
            _dtype_from_type(att.get("type")),
            _granularity(att, default="geocode")
        )

    # Temporal attributes
    for att in atts_temporal:
        dataset.add_temporal_parameter(
            _name_preserve_list(att.get("columns")),
            _text(att.get("description")),
            _dtype_from_type(att.get("type")),
            _granularity(att, default="unknown")
        )

    # Indicators
    for att in atts_indicator:
        dataset.add_existing_indicator(
            _name_preserve_list(att.get("columns")),
            _text(att.get("description")),
            _dtype_from_type(att.get("type")),
            _indicator_type(att),
            _theme(att)
        )

    # Complementary information
    for att in atts_other:
        dataset.add_complementary_information(
            _name_preserve_list(att.get("columns")),
            _text(att.get("description")),
            _dtype_from_type(att.get("type")),
            _granularity(att),
            _theme(att)
        )

    return dataset


# =========================================================
# Value collection and scope computation
# =========================================================

def _collect_values(df, cols):
    """
    Return distinct values for a column specification.

    - str column name -> set of scalar values
    - list of one column -> same as scalar
    - list of multiple columns -> set of row tuples
    """
    def normalize_val(v):
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except Exception:
            return str(v)

    if isinstance(cols, str):
        return set(df[cols].dropna().map(normalize_val))

    elif isinstance(cols, (list, tuple)):
        if len(cols) == 0:
            return set()
        if len(cols) == 1:
            return set(df[cols[0]].dropna().map(normalize_val))
        sub = df[list(cols)].dropna()
        return set(
            map(
                tuple,
                sub.applymap(normalize_val).itertuples(index=False, name=None)
            )
        )

    else:
        return set()


def get_dataset_scopes_gras(df, atts_spatial, atts_temporal):
    """
    Compute spatial and temporal scopes and final granularities for a dataset.
    """
    spatial_gras = [att["granularity"] for att in atts_spatial if att.get("granularity")]
    temporal_gras = [att["granularity"] for att in atts_temporal if att.get("granularity")]

    spatial_scope_level = get_scope(spatial_gras, HIER["spatial"])
    temporal_scope_level = get_scope(temporal_gras, HIER["temporal"])
    spatial_granularity = get_granularity(spatial_gras, HIER["spatial"])
    temporal_granularity = get_granularity(temporal_gras, HIER["temporal"])

    spatial_scope = []
    temporal_scope = []

    # Build spatial scope
    for att in atts_spatial:
        gra = att.get("granularity")

        if gra in spatial_scope_level:
            if "scope_values" in att:
                spatial_scope.append(DSSpatialScope(gra, list(att.get("scope_values") or [])))
                continue

            # Standard label-based spatial scope
            if gra not in {"geometry", "geopoint", "latlon_pair"}:
                values = list(_collect_values(df, att["columns"]))
                spatial_scope.append(DSSpatialScope(gra, values))
                continue

            # Geometry or coordinate-based spatial scope
            cols = att["columns"]
            geoms: List[BaseGeometry] = []

            # Explicit geometry column
            try:
                present_cols = [
                    c for c in (cols if isinstance(cols, list) else [cols])
                    if c in df.columns
                ]
                geom_col = None
                for c in present_cols:
                    sample = df[c].dropna().head(1)
                    if not sample.empty and hasattr(sample.iloc[0], "geom_type"):
                        geom_col = c
                        break
                if geom_col:
                    geoms = _collect_geoms_from_geometry_column(df, geom_col)
            except Exception:
                pass

            # Latitude/longitude pair
            if not geoms and isinstance(cols, list) and len(cols) == 2:
                la, lo = cols[0], cols[1]
                if la in df.columns and lo in df.columns:
                    la_l, lo_l = la.lower(), lo.lower()
                    if any(k in la_l for k in ("lon", "lng", "x")) and \
                       any(k in lo_l for k in ("lat", "y")):
                        la, lo = lo, la
                    try:
                        geoms = _collect_points_from_latlon(df, la, lo)
                    except Exception:
                        pass

            agg = _aggregate_geometry(geoms, method="union", buffer_m=0.0)
            values = _geometry_values_as_wkt(agg)
            spatial_scope.append(DSSpatialScope("geometry_extent", values))

    # Build temporal scope
    for att in atts_temporal:
        gra = att.get("granularity")
        if gra in temporal_scope_level:
            tokens = _collect_values(df, att["columns"])
            ranges = extract_label_ranges(gra, tokens)
            temporal_scope.append(DSTemporalScope(gra, ranges))

    return spatial_scope, temporal_scope, spatial_granularity, temporal_granularity


# =========================================================
# Source registry
# =========================================================

@lru_cache(maxsize=1)
def load_source_registry() -> pd.DataFrame:
    """
    Load dataset&source.xlsx once from MinIO and standardize the key column.
    """
    df = read_excel_from_minio(DATASET_SOURCE_REGISTRY, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")

    if "file_name" not in df.columns:
        raise ValueError("dataset&source.xlsx must contain a 'file_name' column")

    df["file_name"] = df["file_name"].astype(str).str.strip()
    return df


def get_source_info_for_file(file_name: str) -> dict:
    """
    Return source metadata for a file from dataset&source.xlsx.
    """
    reg = load_source_registry()
    row = reg.loc[reg["file_name"].astype(str).str.strip() == str(file_name).strip()]

    if row.empty:
        return {
            "sourceAddress": "",
            "sourceName": file_name,
            "sourceType": "local",
            "filename_series_hint": ""
        }

    rec = row.iloc[0].to_dict()
    return {
        "sourceAddress": rec.get("sourceAddress", "") or "",
        "sourceName": rec.get("sourceName", "") or file_name,
        "sourceType": rec.get("sourceType", "") or "external_source",
        "filename_series_hint": rec.get("filename_series_hint", "") or ""
    }


# =========================================================
# Theme-path handling for raw zone storage
# =========================================================

def normalize_theme_path(theme_path: str) -> list[str]:
    """Split a theme path like 'A > B > C' into cleaned tokens."""
    if not theme_path:
        return []
    return [p.strip() for p in str(theme_path).split(">") if p and p.strip()]


def find_longest_common_theme_path_from_theme_objects(themes) -> str | None:
    """
    Find the longest common prefix theme path from Theme objects or theme dicts.
    """
    if not themes:
        return None

    paths = []
    for th in themes:
        if isinstance(th, dict):
            name = th.get("themeName")
        else:
            name = getattr(th, "themeName", None)
        if name:
            paths.append(normalize_theme_path(name))

    if not paths:
        return None

    common = []
    min_len = min(len(p) for p in paths)

    for i in range(min_len):
        token = paths[0][i]
        if all(p[i] == token for p in paths[1:]):
            common.append(token)
        else:
            break

    return " > ".join(common) if common else None


def slugify_theme(label: str) -> str:
    """Convert a theme label into a folder-safe slug."""
    text = str(label).strip().lower()
    text = text.replace("&", "and").replace("/", "-")
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text


def theme_path_to_rawzone_path(theme_path: str, filename: str) -> str:
    """
    Convert the common theme path into the target raw-zone storage path.
    """
    if not theme_path:
        return f"well-being-and-age-friendliness/raw/unclassified/{filename}"

    levels = [slugify_theme(x) for x in normalize_theme_path(theme_path)]
    return "well-being-and-age-friendliness/raw/" + "/".join(levels) + f"/{filename}"


# =========================================================
# Main dataset construction pipeline
# =========================================================

def construct_dataset(path: str, measure: bool = True):
    """
    Build a Dataset object from one MinIO input object.

    Parameters
    ----------
    path:
        MinIO object key, for example:
        'input_data/fr-esr-parcoursup_2021.csv'

    Notes
    -----
    - The input dataset is read from MinIO via general_function.get_df().
    - Governance resources such as registry, semantic cache and perf cache
      are also read from / written to MinIO.
    """
    title = path.split("/")[-1]
    source_info = get_source_info_for_file(title)
    series_hint = source_info.get("filename_series_hint", "") or None

    # Semantic cache stored in MinIO
    semantic_cache_path = REF_SEMANTIC / f"{title}.json"
    cache_hit = object_exists(semantic_cache_path)

    # Perf cache stored in MinIO
    perf_cache_path = PERF_DIR / f"{title}.perf.json" if cache_hit else None
    sec_timing = None

    if perf_cache_path and object_exists(perf_cache_path):
        perf_obj = json.loads(read_json_from_minio(perf_cache_path).decode("utf-8"))
        timings_obj = perf_obj.get("timings") or {}
        sec_timing = timings_obj.get("semantic_helper_sec")

    timings = {}
    t0_total = time.perf_counter()

    # Read the full dataset from MinIO
    t0 = time.perf_counter()
    df, ext, data_format = get_df(path)
    timings["read_df_sec"] = time.perf_counter() - t0

    # Load semantic cache from MinIO if available.
    # Otherwise compute semantic annotations from a sample and store them back to MinIO.
    if cache_hit and sec_timing is not None:
        semantic_res = pd.read_json(
            io.BytesIO(read_json_from_minio(semantic_cache_path)),
            orient="records"
        )
        timings["semantic_helper_sec"] = sec_timing
    else:
        samples = df.head(5)
        geom_cols = find_geometry_columns(samples)[0]

        if geom_cols:
            samples_for_sem = samples.drop(columns=geom_cols)
        else:
            samples_for_sem = samples

        t0 = time.perf_counter()
        semantic_res = semantic_helper(
            samples_for_sem,
            series_hint=series_hint,
            update_backbone=False
        )
        timings["semantic_helper_sec"] = time.perf_counter() - t0

        payload = semantic_res.to_json(
            orient="records",
            force_ascii=False,
            indent=2
        )
        s3.put_object(
            Bucket=LAKE_BUCKET,
            Key=minio_key(semantic_cache_path),
            Body=payload.encode("utf-8"),
            ContentType="application/json"
        )

    # Attribute classification
    t0 = time.perf_counter()
    results = classify_attributes_with_semantic_helper(df, semantic_res)
    timings["classify_attributes_sec"] = time.perf_counter() - t0

    atts_spatial = results.get("spatial", []) or []
    atts_temporal = results.get("temporal", []) or []
    atts_indicator = results.get("indicators", results.get("indicator", [])) or []
    atts_other = results.get("other", []) or []
    atts_theme = atts_indicator + atts_other

    # Scope and granularity detection
    t0 = time.perf_counter()
    ss, ts, sg, tg = get_dataset_scopes_gras(df, atts_spatial, atts_temporal)
    timings["scopes_granularities_sec"] = time.perf_counter() - t0

    # Theme aggregation and raw-zone path derivation
    t0 = time.perf_counter()
    all_themes = collect_all_themes_set(atts_theme)
    theme_list = [Theme(t, t) for t in all_themes] if all_themes else []
    theme_obj = set(theme_list) if theme_list else None

    common_theme_path = find_longest_common_theme_path_from_theme_objects(theme_list)
    raw_zone_path = theme_path_to_rawzone_path(common_theme_path, title)
    timings["find_common_theme_sec"] = time.perf_counter() - t0

    # File-level metrics from MinIO object metadata
    file_size_bytes = _get_object_size_bytes(path)
    file_size_human = human_readable_size(file_size_bytes)
    n_rows, n_cols = (df.shape if df is not None else (None, None))
    n_records = int(len(df)) if df is not None else None
    n_features = n_records if ext in {".geojson", ".zip"} else None
    unzipped_size = _uncompressed_zip_size_from_minio(path) if ext == ".zip" else None

    # Dataset assembly
    dataset = Dataset(
        title=title,
        description="",
        dataFormat=data_format,
        fileType=ext,
        rawzonePath=raw_zone_path,
        updateFrequency="",
        sourceName=source_info["sourceName"],
        sourceType=source_info["sourceType"],
        sourceAddress=source_info["sourceAddress"],
        spatialGranularity=sg,
        spatialScope=ss,
        temporalGranularity=tg,
        temporalScope=ts,
        themes=theme_obj,
        attributes=[],
        fileSizeBytes=file_size_bytes,
        fileSizeHuman=file_size_human,
        nRows=n_rows,
        nCols=n_cols,
        nRecords=n_records,
        nFeatures=n_features,
        uncompressedSizeBytes=unzipped_size
    )

    # Transform classified attributes into Dataset attributes
    t0 = time.perf_counter()
    dataset = transform_result(dataset, atts_spatial, atts_temporal, atts_indicator, atts_other)
    timings["transform_result_sec"] = time.perf_counter() - t0

    timings["total_sec"] = time.perf_counter() - t0_total

    return (dataset, timings) if measure else dataset
