import os
import io
import json
import re
import time
import zipfile
import pandas as pd

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections.abc import Iterable

from uml_class import *
from scope_detector import *
from granularity_detector import *
from theme_detector import collect_all_themes_set
from attribute_classifier import classify_attributes_with_semantic_helper, find_geometry_columns
from general_function import get_df, human_readable_size, normalise_colname, norm_code, norm_name
from spatial_detector import build_ref_sets, match_series_to_ref_levels

from reference import (
    HIER,
    SPATIAL_NAME_MAP,
    ref_dict,
    REF_SEMANTIC,
    PERF_DIR,
    DATASET_SOURCE_REGISTRY,
    read_csv_from_minio,
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

    def _copy_spatial_extras(param: SpatialParameter, entry: Dict[str, Any]) -> None:
        for key in (
            "contains_aggregate_values",
            "mixed_levels",
            "aggregate_values",
            "values_by_level",
            "insee_geo_object",
            "code_labels",
            "source_code_column",
            "value_level_map",
            "unmatched_values",
        ):
            if key in entry:
                setattr(param, key, entry[key])

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
        if _default_country_spatial_attribute(att):
            continue

        param = dataset.add_spatial_parameter(
            _name_preserve_list(att.get("columns")),
            _text(att.get("description")),
            _dtype_from_type(att.get("type")),
            _granularity(att, default="geocode")
        )
        _copy_spatial_extras(param, att)

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


def _spatial_levels_from_attributes(atts_spatial) -> List[str]:
    """Return detected spatial levels, including mixed levels in one column."""
    levels: List[str] = []
    for att in atts_spatial:
        mixed_levels = att.get("mixed_levels") or []
        if mixed_levels:
            levels.extend(str(level) for level in mixed_levels if level)
            continue
        gra = att.get("granularity")
        if gra:
            levels.append(gra)
    return levels


def _collect_spatial_scope_values(df, att: dict, level: str):
    """
    Collect values for a spatial scope level.

    For mixed-level columns, prefer the pre-classified values for the exact
    scope level so aggregate labels like 'France' are not emitted as region
    or departement values.
    """
    values_by_level = att.get("values_by_level") or {}
    if level in values_by_level:
        return list(values_by_level.get(level) or [])
    return list(_collect_values(df, att["columns"]))


def _dedupe_spatial_scopes(scopes: List[DSSpatialScope]) -> List[DSSpatialScope]:
    """Merge duplicate scope entries at the same level while preserving order."""
    merged: Dict[str, List[Any]] = {}
    for scope in scopes:
        level = scope.spatialScopeLevel
        merged.setdefault(level, [])
        for value in scope.spatialScope or []:
            if value not in merged[level]:
                merged[level].append(value)
    return [DSSpatialScope(level, values) for level, values in merged.items()]


_FILENAME_SPATIAL_LEVEL_ALIASES = [
    ("arr_dep", ("arr_dep", "arrondissement_departemental", "arrondissement_dep")),
    ("com_arr", ("com_arr", "arrondissement_communal", "arrondissement_com")),
    ("dep", ("dep", "dpt", "departement", "department")),
    ("reg", ("reg", "region")),
    ("academie", ("academie", "acad")),
    ("epci", ("epci",)),
    ("canton", ("canton",)),
    ("com", ("com", "commune", "insee")),
    ("iris", ("iris",)),
]


def _default_country_spatial_attribute(att: dict) -> bool:
    """Return True for the synthetic France fallback spatial attribute."""
    return (
        att.get("granularity") == "country"
        and att.get("scope_values") == ["France"]
        and str(att.get("type", "")).lower() == "constant"
    )


def _validated_filename_spatial_hints(filename: Optional[str]) -> List[dict]:
    """
    Extract spatial hints from a filename using explicit level aliases.

    Examples:
      - 2020_detail_IDF_dep_75.csv -> departement 75
      - indicateur_region_11.csv -> region 11
    """
    if not filename or not ref_dict:
        return []

    text = normalise_colname(Path(str(filename)).stem)
    ref_sets = build_ref_sets(ref_dict)
    hits: List[dict] = []

    for raw_level, aliases in _FILENAME_SPATIAL_LEVEL_ALIASES:
        if raw_level not in ref_sets:
            continue

        valid_codes = ref_sets[raw_level].get("codes", set()) or set()
        code_lengths = sorted({len(c) for c in valid_codes}, reverse=True)

        for alias in aliases:
            pat = re.compile(
                rf"(?:^|_){re.escape(alias)}_?(?P<code>2a|2b|[0-9a-z]{{1,9}})(?:_|$)",
                re.I,
            )
            for match in pat.finditer(text):
                raw_code = norm_code(match.group("code"))
                candidates = [raw_code]
                if raw_code.isdigit():
                    candidates.extend(
                        raw_code.zfill(length)
                        for length in code_lengths
                        if length >= len(raw_code)
                    )

                value = next((c for c in candidates if c in valid_codes), None)
                if not value:
                    continue

                level = SPATIAL_NAME_MAP.get(raw_level, raw_level)
                hits.append({
                    "granularity": level,
                    "value": value,
                    "confidence": 0.96,
                    "evidence": f"Filename fallback: detected {level} code '{value}'.",
                })

    uniq = {(h["granularity"], h["value"]): h for h in hits}
    return list(uniq.values())


def _unique_metadata_column_name(df: pd.DataFrame, base: str) -> str:
    """Return a stable metadata-derived column name that does not collide."""
    name = base
    i = 2
    while name in df.columns:
        name = f"{base}_{i}"
        i += 1
    return name


def add_filename_scope_columns(
    df: pd.DataFrame,
    filename: Optional[str],
    atts_spatial: List[dict],
    atts_temporal: List[dict],
) -> tuple[pd.DataFrame, List[dict], List[dict], Dict[str, Any]]:
    """
    Materialize filename-derived temporal/spatial hints as constant raw-zone columns.

    This is used only when the dataset has no real parameter for the corresponding
    dimension. It makes downstream extraction faster because Step 2 can filter real
    columns instead of rebuilding virtual parameters from dataset scope.
    """
    enrichment: Dict[str, Any] = {"temporal": [], "spatial": []}
    if df is None or not filename:
        return df, atts_spatial, atts_temporal, enrichment

    has_temporal_parameter = any(att.get("granularity") for att in atts_temporal)
    has_real_spatial_parameter = any(
        att.get("granularity") and not _default_country_spatial_attribute(att)
        for att in atts_spatial
    )

    if not has_temporal_parameter:
        temporal_hints = extract_temporal_hints_from_text(filename)
        hint_gras = [h["granularity"] for h in temporal_hints]
        temporal_level = get_granularity(hint_gras, HIER["temporal"])

        if temporal_level:
            values = list(dict.fromkeys(
                h["value"]
                for h in temporal_hints
                if h["granularity"] == temporal_level and h.get("value")
            ))
            if values:
                col = _unique_metadata_column_name(
                    df,
                    f"METADATA_TEMPORAL_{str(temporal_level).upper()}",
                )
                value = values[0]
                df[col] = value
                entry = {
                    "columns": col,
                    "description": f"Temporal parameter derived from dataset filename '{filename}'.",
                    "type": "constant",
                    "granularity": temporal_level,
                    "scope_values": values,
                    "confidence": 0.95,
                    "matched_by": "filename_temporal_hint_materialized",
                    "evidence": f"Filename contains {temporal_level} value '{value}'.",
                }
                atts_temporal.append(entry)
                enrichment["temporal"].append(entry)

    if not has_real_spatial_parameter:
        spatial_hints = _validated_filename_spatial_hints(filename)
        hint_gras = [h["granularity"] for h in spatial_hints]
        spatial_level = get_granularity(hint_gras, HIER["spatial"])

        if spatial_level:
            values = list(dict.fromkeys(
                h["value"]
                for h in spatial_hints
                if h["granularity"] == spatial_level and h.get("value")
            ))
            if values:
                col = _unique_metadata_column_name(
                    df,
                    f"METADATA_SPATIAL_{str(spatial_level).upper()}",
                )
                value = values[0]
                df[col] = value
                entry = {
                    "columns": col,
                    "description": f"Spatial parameter derived from dataset filename '{filename}'.",
                    "type": "constant",
                    "granularity": spatial_level,
                    "scope_values": values,
                    "confidence": 0.96,
                    "matched_by": "filename_spatial_hint_materialized",
                    "evidence": f"Filename contains {spatial_level} value '{value}'.",
                }
                atts_spatial = [
                    att
                    for att in atts_spatial
                    if not _default_country_spatial_attribute(att)
                ]
                atts_spatial.append(entry)
                enrichment["spatial"].append(entry)

    return df, atts_spatial, atts_temporal, enrichment


def _apply_filename_scope_fallbacks(
    filename: Optional[str],
    spatial_scope: List[DSSpatialScope],
    temporal_scope: List[DSTemporalScope],
    spatial_granularity: Optional[str],
    temporal_granularity: Optional[str],
    atts_spatial,
    atts_temporal,
):
    """Use filename hints only when real dataset parameters are unavailable."""
    has_real_spatial_parameter = any(
        att.get("granularity") and not _default_country_spatial_attribute(att)
        for att in atts_spatial
    )
    has_temporal_parameter = any(att.get("granularity") for att in atts_temporal)

    if filename and not has_temporal_parameter:
        temporal_hints = extract_temporal_hints_from_text(filename)
        hint_gras = [h["granularity"] for h in temporal_hints]
        fallback_temporal_granularity = get_granularity(hint_gras, HIER["temporal"])

        if fallback_temporal_granularity:
            values = [
                h["value"]
                for h in temporal_hints
                if h["granularity"] == fallback_temporal_granularity
            ]
            ranges = extract_label_ranges(fallback_temporal_granularity, values)
            if ranges:
                temporal_scope = [DSTemporalScope(fallback_temporal_granularity, ranges)]
                temporal_granularity = fallback_temporal_granularity

    if filename and not has_real_spatial_parameter:
        spatial_hints = _validated_filename_spatial_hints(filename)
        hint_gras = [h["granularity"] for h in spatial_hints]
        fallback_spatial_granularity = get_granularity(hint_gras, HIER["spatial"])
        fallback_scope_levels = get_scope(hint_gras, HIER["spatial"])

        if fallback_spatial_granularity:
            spatial_granularity = fallback_spatial_granularity

        if fallback_scope_levels:
            spatial_scope = [
                sc
                for sc in spatial_scope
                if getattr(sc, "spatialScopeLevel", None) != "country"
            ]
            for level in fallback_scope_levels:
                values = [
                    h["value"]
                    for h in spatial_hints
                    if h["granularity"] == level
                ]
                if values:
                    spatial_scope.append(DSSpatialScope(level, list(dict.fromkeys(values))))

    return spatial_scope, temporal_scope, spatial_granularity, temporal_granularity


def get_dataset_scopes_gras(df, atts_spatial, atts_temporal, filename: Optional[str] = None):
    """
    Compute spatial and temporal scopes and final granularities for a dataset.
    """
    spatial_gras = _spatial_levels_from_attributes(atts_spatial)
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
                values = _collect_spatial_scope_values(df, att, gra)
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

        for mixed_level in att.get("mixed_levels") or []:
            if mixed_level in spatial_scope_level and mixed_level != gra:
                values = _collect_spatial_scope_values(df, att, mixed_level)
                spatial_scope.append(DSSpatialScope(mixed_level, values))

    spatial_scope = _dedupe_spatial_scopes(spatial_scope)

    # Build temporal scope
    for att in atts_temporal:
        gra = att.get("granularity")
        if gra in temporal_scope_level:
            tokens = _collect_values(df, att["columns"])
            ranges = extract_label_ranges(gra, tokens)
            temporal_scope.append(DSTemporalScope(gra, ranges))

    return _apply_filename_scope_fallbacks(
        filename,
        spatial_scope,
        temporal_scope,
        spatial_granularity,
        temporal_granularity,
        atts_spatial,
        atts_temporal,
    )


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
# INSEE sidecar code-list helpers
# =========================================================

def _insee_metadata_path_for_data_path(path: str) -> Optional[str]:
    """Return the expected INSEE *_metadata.csv sidecar path for a *_data.csv file."""
    key = minio_key(path)
    dirname, filename = key.rsplit("/", 1) if "/" in key else ("", key)

    candidates = []
    if filename.endswith("_data.csv"):
        candidates.append(filename[:-len("_data.csv")] + "_metadata.csv")
    if filename.endswith("_data.CSV"):
        candidates.append(filename[:-len("_data.CSV")] + "_metadata.CSV")

    for candidate in candidates:
        sidecar = f"{dirname}/{candidate}" if dirname else candidate
        if object_exists(sidecar):
            return sidecar

    return None


def _read_insee_metadata_sidecar(path: str) -> Optional[pd.DataFrame]:
    """Read an INSEE code-list sidecar if it exists and has the expected schema."""
    sidecar = _insee_metadata_path_for_data_path(path)
    if not sidecar:
        return None

    try:
        meta = read_csv_from_minio(sidecar, dtype=str, sep=";")
    except Exception:
        return None

    meta.columns = [str(c).strip().strip('"') for c in meta.columns]
    required = {"COD_VAR", "LIB_VAR", "COD_MOD", "LIB_MOD"}
    if not required.issubset(set(meta.columns)):
        return None

    return meta.fillna("")


def _normalise_insee_code(value: Any) -> str:
    text = str(value).strip().strip('"')
    if not text:
        return ""
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text.upper()


def _code_label_map_for_var(meta: pd.DataFrame, var: str) -> Dict[str, str]:
    sub = meta.loc[meta["COD_VAR"].astype(str).str.strip() == var]
    out: Dict[str, str] = {}
    for _, row in sub.iterrows():
        code = str(row.get("COD_MOD", "")).strip()
        label = str(row.get("LIB_MOD", "")).strip()
        if code:
            out[_normalise_insee_code(code)] = label
    return out


def _insee_geo_object_values(df: pd.DataFrame) -> List[str]:
    if "GEO_OBJECT" not in df.columns:
        return []
    values = [
        str(v).strip()
        for v in df["GEO_OBJECT"].dropna().drop_duplicates().tolist()
        if str(v).strip()
    ]
    return values[:20]


def _unique_column_name(df: pd.DataFrame, base: str) -> str:
    """Return a column name that does not already exist in the dataframe."""
    if base not in df.columns:
        return base
    i = 2
    while f"{base}_{i}" in df.columns:
        i += 1
    return f"{base}_{i}"


def add_insee_geo_label_column_from_sidecar(
    df: pd.DataFrame,
    path: str,
) -> tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Add a real label column for INSEE dataset-specific GEO codes.

    The original GEO code column is kept unchanged. The new GEO_LABEL column is
    the value that should be used by downstream filtering when the numeric GEO
    code has a dataset-specific meaning declared in *_metadata.csv.
    """
    if df is None or "GEO" not in df.columns:
        return df, None

    meta = _read_insee_metadata_sidecar(path)
    if meta is None:
        return df, None

    label_by_code = _code_label_map_for_var(meta, "GEO")
    if not label_by_code:
        return df, None

    values = [
        str(v).strip().strip('"')
        for v in df["GEO"].dropna().drop_duplicates().tolist()
        if str(v).strip()
    ]
    if not values:
        return df, None

    matched = [
        value
        for value in values
        if _normalise_insee_code(value) in label_by_code
    ]
    if len(matched) / max(1, len(values)) < 0.8:
        return df, None

    out = df.copy()
    label_col = _unique_column_name(out, "GEO_LABEL")
    out[label_col] = out["GEO"].map(
        lambda v: label_by_code.get(_normalise_insee_code(v), "")
        if pd.notna(v) and str(v).strip()
        else ""
    )

    code_labels = {
        value: label_by_code.get(_normalise_insee_code(value), "")
        for value in values
    }
    info = {
        "code_column": "GEO",
        "label_column": label_col,
        "code_labels": code_labels,
        "insee_geo_object": _insee_geo_object_values(out),
        "sidecar_path": _insee_metadata_path_for_data_path(path),
    }
    return out, info


def _infer_spatial_level_for_insee_geo_label(
    df: pd.DataFrame,
    label_col: str,
    *,
    min_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Infer the real reference spatial level for an INSEE GEO_LABEL column.

    Once GEO codes are resolved to labels, the labels can often be matched to
    ref_spatial names. In that case the column is no longer an opaque insee_geo
    code-list; it is a normal region/departement/commune/etc. parameter.
    """
    if label_col not in df.columns:
        return {}

    try:
        ref_sets = build_ref_sets(ref_dict)
        best = match_series_to_ref_levels(df[label_col], ref_sets)
    except Exception:
        return {}

    if not best or float(best.get("ratio") or 0.0) < min_ratio:
        return {}

    return best


def _infer_mixed_spatial_levels_for_insee_geo_label(
    df: pd.DataFrame,
    label_col: str,
) -> Dict[str, Any]:
    """
    Match each GEO_LABEL value to ref_spatial levels.

    Some INSEE GEO code-lists mix communes, EPCI and other administrative
    levels in the same GEO column. This returns per-level values so dataset
    scope and downstream filtering can choose the useful level later.
    """
    if label_col not in df.columns:
        return {}

    try:
        ref_sets = build_ref_sets(ref_dict)
    except Exception:
        return {}

    values = [
        str(v).strip()
        for v in df[label_col].dropna().drop_duplicates().tolist()
        if str(v).strip()
    ]
    if not values:
        return {}

    level_order = [
        "country",
        "region",
        "academie",
        "departement",
        "arrondissement_departemental",
        "epci",
        "canton",
        "commune",
        "arrondissement_communal",
        "iris",
    ]
    level_rank = {level: idx for idx, level in enumerate(level_order)}

    values_by_level: Dict[str, List[str]] = {}
    unmatched: List[str] = []
    value_level_map: Dict[str, str] = {}

    for value in values:
        value_name = norm_name(value)
        hits: List[str] = []
        for raw_level, sets in ref_sets.items():
            level = SPATIAL_NAME_MAP.get(raw_level, raw_level)
            if value_name and value_name in (sets.get("names", set()) or set()):
                hits.append(level)

        if not hits:
            unmatched.append(value)
            continue

        # If one label exists at several levels, keep the most specific one.
        # The original label remains unchanged in values_by_level.
        chosen = sorted(hits, key=lambda x: level_rank.get(x, -1), reverse=True)[0]
        values_by_level.setdefault(chosen, []).append(value)
        value_level_map[value] = chosen

    mixed_levels = [
        level
        for level in level_order
        if values_by_level.get(level)
    ]

    if not mixed_levels:
        return {}

    total = len(values)
    matched = total - len(unmatched)
    return {
        "mixed_levels": mixed_levels,
        "values_by_level": values_by_level,
        "value_level_map": value_level_map,
        "unmatched_values": unmatched[:200],
        "match_ratio": matched / total if total else 0.0,
    }


def enrich_insee_spatial_attributes_from_sidecar(
    df: pd.DataFrame,
    path: str,
    atts_spatial: List[dict],
    atts_other: List[dict],
    enrichment_info: Optional[Dict[str, Any]] = None,
) -> tuple[List[dict], List[dict]]:
    """
    Detect INSEE dataset-specific GEO code-list columns.

    INSEE CSV packages often encode geography as GEO codes whose meanings are
    declared in the sibling *_metadata.csv file. Those codes are not
    administrative codes, so reference-spatial matching must not reinterpret
    them as region/departement/commune.
    """
    if "GEO" not in df.columns:
        return atts_spatial, atts_other

    if enrichment_info is None:
        df, enrichment_info = add_insee_geo_label_column_from_sidecar(df, path)
    if not enrichment_info:
        return atts_spatial, atts_other

    code_col = enrichment_info.get("code_column", "GEO")
    label_col = enrichment_info.get("label_column", "GEO_LABEL")
    if label_col not in df.columns:
        return atts_spatial, atts_other

    values = [
        str(v).strip()
        for v in df[label_col].dropna().drop_duplicates().tolist()
        if str(v).strip()
    ]
    mixed = _infer_mixed_spatial_levels_for_insee_geo_label(df, label_col)
    inferred = _infer_spatial_level_for_insee_geo_label(df, label_col) if not mixed else {}

    if mixed and len(mixed.get("mixed_levels") or []) > 1:
        spatial_level = "mixed"
        matched_by = "insee_metadata_sidecar_label_mixed_ref_spatial"
        confidence = min(0.99, max(0.80, float(mixed.get("match_ratio") or 0.0)))
        scope_values = values
    elif mixed and len(mixed.get("mixed_levels") or []) == 1:
        spatial_level = mixed["mixed_levels"][0]
        matched_by = "insee_metadata_sidecar_label_ref_spatial"
        confidence = min(0.99, max(0.80, float(mixed.get("match_ratio") or 0.0)))
        scope_values = mixed.get("values_by_level", {}).get(spatial_level, values)
    else:
        spatial_level = inferred.get("level") or "insee_geo"
        matched_by = (
            f"insee_metadata_sidecar_label_{inferred.get('by')}"
            if inferred
            else "insee_metadata_sidecar"
        )
        confidence = min(0.99, max(0.80, float(inferred.get("ratio") or 0.98))) if inferred else 0.98
        scope_values = values

    entry = {
        "columns": label_col,
        "description": "INSEE dataset-specific geography label resolved from GEO and *_metadata.csv",
        "type": str(df[label_col].dtype),
        "granularity": spatial_level,
        "confidence": confidence,
        "matched_by": matched_by,
        "scope_values": scope_values,
        "insee_geo_object": enrichment_info.get("insee_geo_object", []),
        "code_labels": enrichment_info.get("code_labels", {}),
        "source_code_column": code_col,
        "contains_aggregate_values": spatial_level == "mixed",
        "mixed_levels": mixed.get("mixed_levels", []) if mixed else [],
        "values_by_level": mixed.get("values_by_level", {}) if mixed else {},
        "value_level_map": mixed.get("value_level_map", {}) if mixed else {},
        "unmatched_values": mixed.get("unmatched_values", []) if mixed else [],
        "evidence": (
            f"GEO values were resolved to {label_col} using the sibling INSEE *_metadata.csv code-list; "
            f"{label_col} contains mixed ref_spatial levels: {', '.join(mixed.get('mixed_levels', []))}."
            if mixed and spatial_level == "mixed"
            else (
                f"GEO values were resolved to {label_col} using the sibling INSEE *_metadata.csv code-list; "
                f"{label_col} matched ref_spatial level '{spatial_level}'."
                if inferred or mixed
                else "GEO values were resolved to labels using the sibling INSEE *_metadata.csv code-list, but labels did not match a ref_spatial level."
            )
        ),
    }

    updated_spatial = []
    replaced = False
    for att in atts_spatial:
        cols = att.get("columns")
        cols_list = cols if isinstance(cols, list) else [cols]
        if code_col in cols_list or label_col in cols_list:
            updated_spatial.append({**att, **entry})
            replaced = True
        else:
            updated_spatial.append(att)

    if not replaced:
        updated_spatial.append(entry)

    updated_other = []
    for att in atts_other:
        cols = att.get("columns")
        cols_list = cols if isinstance(cols, list) else [cols]
        if code_col not in cols_list and label_col not in cols_list:
            updated_other.append(att)

    return updated_spatial, updated_other


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
    from semantic_helper import semantic_helper

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
    df, insee_geo_enrichment = add_insee_geo_label_column_from_sidecar(df, path)
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
    atts_spatial, atts_other = enrich_insee_spatial_attributes_from_sidecar(
        df,
        path,
        atts_spatial,
        atts_other,
        insee_geo_enrichment,
    )
    df, atts_spatial, atts_temporal, filename_scope_enrichment = add_filename_scope_columns(
        df,
        title,
        atts_spatial,
        atts_temporal,
    )
    atts_theme = atts_indicator + atts_other

    # Scope and granularity detection
    t0 = time.perf_counter()
    ss, ts, sg, tg = get_dataset_scopes_gras(df, atts_spatial, atts_temporal, filename=title)
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
    if insee_geo_enrichment or filename_scope_enrichment.get("temporal") or filename_scope_enrichment.get("spatial"):
        dataset._rawzone_enriched_df = df
        dataset._rawzone_enrichment = {
            "insee_geo": insee_geo_enrichment,
            "filename_scope": filename_scope_enrichment,
        }

    # Transform classified attributes into Dataset attributes
    t0 = time.perf_counter()
    dataset = transform_result(dataset, atts_spatial, atts_temporal, atts_indicator, atts_other)
    timings["transform_result_sec"] = time.perf_counter() - t0

    timings["total_sec"] = time.perf_counter() - t0_total

    return (dataset, timings) if measure else dataset
