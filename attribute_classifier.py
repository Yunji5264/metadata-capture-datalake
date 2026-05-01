import binascii

from pandas.api.types import is_object_dtype, is_string_dtype
from shapely import wkb

from spatial_detector import *
from temporal_detector import *
from indicator_detector import *
from reference import *

DEFAULT_COUNTRY_SPATIAL_ATTRIBUTE = {
    "columns": "country",
    "description": "dataset country scope",
    "type": "constant",
    "granularity": "country",
    "confidence": 1.0,
    "scope_values": ["France"],
    "evidence": "Default country-level spatial scope for the French data lake.",
}


def find_geometry_columns(df: pd.DataFrame) -> list[str]:
    """
    Return a list of geometry-like columns in the DataFrame:
    - If GeoDataFrame: use the true geometry column name
    - Otherwise: check if cell values are shapely geometries (have .geom_type)
    - Fallback: check common column names like "geometry", "geom", "the_geom"
    """
    geom_cols = set()
    wkb_cols = set()

    # 1) GeoDataFrame case
    try:
        if isinstance(df, gpd.GeoDataFrame):
            geom_cols.add(df.geometry.name)
    except Exception:
        pass

    # 2) Value type inspection (shapely geometries usually have .geom_type)
    for c in df.columns:
        s = df[c]
        sample = s.dropna().head(20)
        if sample.empty:
            continue
        v = sample.iloc[0]
        if hasattr(v, "geom_type"):
            geom_cols.add(c)
        # 4) Value
        elif detect_geometry_object(s) :
            geom_cols.add(c)
        if as_wkb_bytes(v):
            wkb_cols.add(c)

    # 3) Common name fallback (only if nothing found)
    if not geom_cols:
        common_names = {"geometry", "geom", "the_geom"}
        geom_cols |= {c for c in df.columns if c.lower() in common_names}

    return list(geom_cols), list(wkb_cols)


# def classify_attributes(
#     df: pd.DataFrame,
#     *,
#     min_ratio: float = 0.7,
#     sample_size: int = 300,
#     include_city_zip_in_address: bool = False,
#     address_colname: str = "__address__",
#     filename: str | None = None,
#     sheet_names: List[str] | None = None,
#     require_name_hint_for_geoformats: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Heuristic + reference-based classifier (no semantic_res required).
#     All entries consistently use "columns": [...] (never "column").
#     """
#
#     def _to_columns(entry: Dict[str, Any], name: str) -> List[str]:
#         """Ensure a list-of-string for the 'columns' field."""
#         if "columns" in entry and entry["columns"] is not None:
#             cols = entry["columns"]
#             if isinstance(cols, list):
#                 return [str(c) for c in cols]
#             return [str(cols)]
#         if "column" in entry and entry["column"] is not None:
#             return [str(entry["column"])]
#         return [str(name)]
#
#     results = {
#         "spatial": [],
#         "temporal": [],
#         "unknown": [],
#         "indicators": [],
#         "other": [],
#         "meta": {}
#     }
#
#     # 0) Address aggregation
#     df2, addr_cols = add_combined_address_column(
#         df, colname=address_colname, include_city_zip=include_city_zip_in_address
#     )
#     results["meta"]["address_columns_used"] = addr_cols
#     results["meta"]["config"] = {
#         "min_ratio": min_ratio,
#         "sample_size": sample_size,
#         "include_city_zip_in_address": include_city_zip_in_address,
#         "address_colname": address_colname,
#         "filename": filename,
#         "sheet_names": sheet_names or [],
#         "require_name_hint_for_geoformats": require_name_hint_for_geoformats,
#     }
#
#     # 1) Lat/Lon pair
#     consumed: set = set()
#     latlon = detect_latlon_pair(df2)
#     if latlon:
#         la, lo = latlon
#         consumed.update([la, lo])
#         try:
#             df2[la] = pd.to_numeric(df2[la], errors="coerce")
#             df2[lo] = pd.to_numeric(df2[lo], errors="coerce")
#         except Exception:
#             pass
#
#         results["spatial"].append({
#             "columns": [la, lo],
#             "description": "latlon pair geopoint",
#             "type": [str(df2[la].dtype), str(df2[lo].dtype)],
#             "granularity": "latlon_pair",
#             "confidence": 0.98,
#             "evidence": "Both columns numeric and within valid lat/lon ranges."
#         })
#
#     for gc in find_geometry_columns(df):
#         consumed.update(gc)
#         results["spatial"].append({
#             "columns": [gc],
#             "description": "geometry column (excluded from semantic analysis)",
#             "type": str(df[gc].dtype),
#             "granularity": "geometry",
#             "confidence": 0.99,
#             "evidence": "Detected geometry column and skipped for LLM input."
#         })
#
#     # 2) Reference sets
#     ref_sets = build_ref_sets(ref_dict) if ref_dict else {}
#
#     # 3) Column-wise classification
#     for col in df2.columns:
#         if col in consumed:
#             continue
#
#         s = df2[col]
#         fmt_hints = geoformat_hints_from_colname(col) or {}
#
#         # # Geometry objects
#         # if detect_geometry_object(s):
#         #     conf = 0.99 if fmt_hints.get("geometry") else 0.98
#         #     evd = "Objects/dtype appear to be geometry." + (" Column name suggests geometry." if fmt_hints.get("geometry") else "")
#         #     results["spatial"].append({
#         #         "columns": col,
#         #         "description": "geometry",
#         #         "type": str(s.dtype),
#         #         "granularity": "geometry",
#         #         "confidence": conf,
#         #         "evidence": evd
#         #     })
#         #     continue
#         #
#         # # WKT / GeoJSON strings
#         # if detect_wkt_geojson_string(s):
#         #     name_gate_ok = fmt_hints.get("wkt") or fmt_hints.get("geojson") or fmt_hints.get("geometry")
#         #     if (not require_name_hint_for_geoformats) or name_gate_ok:
#         #         conf = 0.96 if name_gate_ok else 0.93
#         #         evd = "Values match WKT/GeoJSON textual patterns." + (" Column name suggests WKT/GeoJSON." if name_gate_ok else "")
#         #         results["spatial"].append({
#         #             "columns": col,
#         #             "description": "geometry",
#         #             "type": str(s.dtype),
#         #             "granularity": "geometry",
#         #             "confidence": conf,
#         #             "evidence": evd
#         #         })
#         #         continue
#
#         # Address
#         if col == address_colname and detect_address(s):
#             results["spatial"].append({
#                 "columns": addr_cols,
#                 "description": "complete address",
#                 "type": str(s.dtype),
#                 "granularity": "address",
#                 "confidence": 0.8,
#                 "evidence": "Aggregated address lines with street tokens and numbers."
#             })
#             continue
#         if not addr_cols and detect_address(s):
#             results["spatial"].append({
#                 "columns": col,
#                 "description": "complete address",
#                 "type": str(s.dtype),
#                 "granularity": "address",
#                 "confidence": 0.75,
#                 "evidence": "Address-like strings detected."
#             })
#             continue
#
#         # Reference matching (gated)
#         gate = spatial_gate_from_colname(col) or {"has_gate": False, "levels": [], "generic": False}
#         if ref_sets and not_null_ratio(s) > 0.2 and gate.get("has_gate", False):
#             if gate.get("levels"):
#                 filtered = {lvl: ref_sets[lvl] for lvl in gate["levels"] if lvl in ref_sets}
#                 if filtered:
#                     best = match_series_to_ref_levels(s, filtered, sample_size=sample_size)
#                     if best and best["ratio"] >= min_ratio:
#                         results["spatial"].append({
#                             "columns": col,
#                             "description": f"geographic attribute hinted by name ({', '.join(gate['levels'])})",
#                             "type": str(s.dtype),
#                             "granularity": best["level"],
#                             "matched_by": best["by"],
#                             "confidence": min(0.99, 0.7 + 0.3 * best["ratio"]),
#                             "evidence": f"Name gate [{', '.join(gate['levels'])}] → ref match by {best['by']} ({round(best['ratio'] * 100)}%)."
#                         })
#                         continue
#                     elif best:
#                         results["other"].append({
#                             "columns": col,
#                             "description": f"geographic attribute hinted by name ({', '.join(gate['levels'])})",
#                             "type": str(s.dtype),
#                             "granularity": best["level"],
#                             "matched_by": best["by"]
#                         })
#                         continue
#
#             elif gate.get("generic", False):
#                 best = match_series_to_ref_levels(s, ref_sets, sample_size=sample_size) if ref_sets else None
#                 generic_min = max(min_ratio, 0.80)
#                 if best and best["ratio"] >= generic_min:
#                     results["spatial"].append({
#                         "columns": col,
#                         "description": "generic geographic attribute (name-based gate)",
#                         "type": str(s.dtype),
#                         "granularity": best["level"],
#                         "matched_by": best["by"],
#                         "confidence": min(0.99, 0.68 + 0.32 * best["ratio"]),
#                         "evidence": f"Generic geo gate → ref match by {best['by']} ({round(best['ratio'] * 100)}%)."
#                     })
#                     continue
#                 elif best:
#                     results["other"].append({
#                         "columns": col,
#                         "description": "generic geographic attribute (name-based gate)",
#                         "type": str(s.dtype),
#                         "granularity": best["level"],
#                         "matched_by": best["by"]
#                     })
#                     continue
#
#         # Temporal granularity
#         gran, conf = detect_temporal_granularity(s)
#         if gran:
#             results["temporal"].append({
#                 "columns": col,
#                 "description": "temporal attribute",
#                 "type": str(s.dtype),
#                 "granularity": gran,
#                 "confidence": conf
#             })
#             continue
#
#         # Generic geocode fallback
#         nunique_ratio = s.nunique(dropna=True) / max(1, len(s))
#         if nunique_ratio > 0.9 and (is_string_series(s) or is_numeric_series(s)):
#             results["spatial"].append({
#                 "columns": col,
#                 "description": "high-uniqueness identifier (treated as geocode)",
#                 "type": str(s.dtype),
#                 "granularity": "geocode",
#                 "confidence": 0.6,
#                 "evidence": "High-uniqueness identifier; treated as generic geocode."
#             })
#             continue
#
#         # Indicators (quantitative / qualitative) — normalised to "columns"
#         is_quant, payload_q = detect_quantitative_indicator(s, col)
#         if is_quant:
#             payload_q = dict(payload_q)  # copy
#             payload_q["columns"] = _to_columns(payload_q, col)
#             payload_q.pop("column", None)
#             results["indicators"].append({
#                 "columns": payload_q["columns"],
#                 "type": str(s.dtype),
#                 **{k: v for k, v in payload_q.items() if k not in {"columns"}}
#             })
#             continue
#
#         is_qual, payload_ql = detect_qualitative_indicator(s)
#         if is_qual:
#             payload_ql = dict(payload_ql)
#             payload_ql["columns"] = _to_columns(payload_ql, col)
#             payload_ql.pop("column", None)
#             results["indicators"].append({
#                 "columns": payload_ql["columns"],
#                 "type": str(s.dtype),
#                 **{k: v for k, v in payload_ql.items() if k not in {"columns"}}
#             })
#             continue
#
#         # Other
#         avg_len = None
#         try:
#             avg_len = float(s.dropna().astype(str).str.len().mean())
#         except Exception:
#             pass
#         results["other"].append({
#             "columns": col,
#             "description": "unclassified attribute",
#             "type": str(s.dtype),
#             "reason": "does not fit spatial/temporal or indicator heuristics",
#             "avg_text_len": avg_len
#         })
#
#     return results



# ---------- Main function with type guards ----------

def classify_attributes_with_semantic_helper(
    df: pd.DataFrame,
    semantic_res: pd.DataFrame,
    *,
    min_ratio: float = 0.7,
    sample_size: int = 300,
    include_city_zip_in_address: bool = False,
    address_colname: str = "__address__",
    filename: str | None = None,
    sheet_names: List[str] | None = None,
    require_name_hint_for_geoformats: bool = True,
) -> Dict[str, Any]:
    """
    Hybrid classifier with semantic guidance.
    All entries consistently use "columns": [...] (never "column").
    """

    def _ensure_columns_name_list(name: str | List[str]) -> List[str]:
        if isinstance(name, list):
            return [str(x) for x in name]
        return [str(name)]

    def _col_list_from_sem(sem_df: pd.DataFrame, mask) -> List[str]:
        cols = sem_df.loc[mask, "column_name"].tolist()
        return [c for c in cols if c in df.columns]

    def _desc(sem_df: pd.DataFrame, col: str) -> str:
        vals = sem_df.loc[sem_df["column_name"] == col, "meaning"].tolist()
        return vals[0] if vals else ""

    results = {
        "spatial": [],
        "temporal": [],
        "unknown": [],
        "indicators": [],
        "other": [],
        "meta": {}
    }

    consumed: set = set()
    spatial_cols=[]

    # Detect geometry-like columns in the ORIGINAL df and record them
    geom_cols, wkb_cols = find_geometry_columns(df)
    if geom_cols:
        for gc in geom_cols:
            results["spatial"].append({
                "columns": [gc],
                "description": "geometry column",
                "type": str(df[gc].dtype),
                "granularity": "geometry",
                "confidence": 0.99,
                "evidence": "Detected geometry/shapely objects; excluded from semantic sampling."
            })
            consumed.add(gc)
            spatial_cols.append(gc)
    if wkb_cols:
        for wkb in wkb_cols:
            results["spatial"].append({
                "columns": [wkb],
                "description": "geometry column in form wkb",
                "type": str(df[wkb].dtype),
                "granularity": "geometry",
                "confidence": 0.99,
                "evidence": "Detected geometry/shapely objects; excluded from semantic sampling."
            })
            consumed.add(wkb)
            spatial_cols.append(wkb)

    # 0) Resolve columns by semantic hints


    spatial_cols = spatial_cols + _col_list_from_sem(semantic_res, semantic_res["is_spatial"] == True)
    temporal_cols = _col_list_from_sem(semantic_res, semantic_res["is_temporal"] == True)
    indicator_qual_cols = _col_list_from_sem(
        semantic_res, (semantic_res["is_indicator"] == True) & (semantic_res["indicator_type"] == "Qualitative")
    )
    indicator_quant_cols = _col_list_from_sem(
        semantic_res, (semantic_res["is_indicator"] == True) & (semantic_res["indicator_type"] == "Quantitative")
    )

    used_cols = set(spatial_cols + temporal_cols + indicator_qual_cols + indicator_quant_cols)
    other_cols = [c for c in df.columns if c not in used_cols]

    df_spatial = df[spatial_cols] if spatial_cols else pd.DataFrame(index=df.index)
    df_temporal = df[temporal_cols] if temporal_cols else pd.DataFrame(index=df.index)
    df_indicator_qual = df[indicator_qual_cols] if indicator_qual_cols else pd.DataFrame(index=df.index)
    df_indicator_quant = df[indicator_quant_cols] if indicator_quant_cols else pd.DataFrame(index=df.index)
    df_other = df[other_cols] if other_cols else pd.DataFrame(index=df.index)

    # 1) Spatial refinement (address composition)
    df2, addr_cols = add_combined_address_column(
        df_spatial, colname=address_colname, include_city_zip=include_city_zip_in_address
    )
    results["meta"]["address_columns_used"] = addr_cols

    # Config snapshot
    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname,
        "filename": filename,
        "sheet_names": sheet_names or [],
        "require_name_hint_for_geoformats": require_name_hint_for_geoformats,
    }



    # 1a) Detect and record lat/lon pairs
    latlon = detect_latlon_pair(df2)
    if latlon:
        la, lo = latlon
        consumed.update([lo, la])
        try:
            df2[lo] = pd.to_numeric(df2[lo], errors="coerce")
            df2[la] = pd.to_numeric(df2[la], errors="coerce")
        except Exception:
            pass
        results["spatial"].append({
            "columns": [la, lo],  # keep [lat, lon]
            "description": "latlon pair geopoint",
            "granularity": "latlon_pair",
            "type": [str(df2[la].dtype), str(df2[lo].dtype)],
            "confidence": 0.98,
            "evidence": "Both columns numeric and within valid lat/lon ranges."
        })


    # 1c) Reference sets (if available externally)
    ref_sets = build_ref_sets(ref_dict) if ref_dict else {}

    # 1d) Column-wise spatial inference with type guards
    for col in df2.columns:
        if col in consumed:
            continue

        s = df2[col]
        fmt_hints = geoformat_hints_from_colname(col) or {}

        # 用于兜底：若整条规则链未命中，则强制进入 other
        assigned = False

        # Guard 0: skip columns that are known geometry-like
        # Intentionally excluded from both spatial and other buckets
        if col in geom_cols:
            continue

        # Guard 1: shapely/object geometry detection only for object-like series
        if is_object_dtype(s):
            if detect_geometry_object(s):
                conf = 0.99 if fmt_hints.get("geometry") else 0.98
                evd = "Objects/dtype appear to be geometry." + (
                    " Column name suggests geometry." if fmt_hints.get("geometry") else ""
                )
                results["spatial"].append({
                    "columns": col,
                    "description": "geometry",
                    "type": str(s.dtype),
                    "granularity": "geometry",
                    "confidence": conf,
                    "evidence": evd
                })
                consumed.add(col)
                assigned = True
                continue

        # Guard 2: WKT/GeoJSON textual detection only for text-like series
        if detect_wkt_geojson_string(s):
            name_gate_ok = (
                    fmt_hints.get("wkt")
                    or fmt_hints.get("geojson")
                    or fmt_hints.get("geometry")
            )
            if (not require_name_hint_for_geoformats) or name_gate_ok:
                conf = 0.96 if name_gate_ok else 0.93
                evd = "Values match WKT/GeoJSON textual patterns." + (
                    " Column name suggests WKT/GeoJSON." if name_gate_ok else ""
                )
                results["spatial"].append({
                    "columns": col,
                    "description": "geometry",
                    "type": str(s.dtype),
                    "granularity": "geometry",
                    "confidence": conf,
                    "evidence": evd
                })
                consumed.add(col)
                assigned = True
                continue

        # Address detection
        if col == address_colname and detect_address(s):
            results["spatial"].append({
                "columns": addr_cols,
                "description": "complete address",
                "type": str(s.dtype),
                "granularity": "address",
                "confidence": 0.8,
                "evidence": "Aggregated address lines with street tokens and numbers."
            })
            assigned = True
            continue

        if not addr_cols and detect_address(s):
            results["spatial"].append({
                "columns": col,
                "description": "complete address",
                "type": str(s.dtype),
                "granularity": "address",
                "confidence": 0.75,
                "evidence": "Address-like strings detected."
            })
            assigned = True
            continue

        # Gate-based reference matching
        gate = spatial_gate_from_colname(col) or {
            "has_gate": False,
            "levels": [],
            "generic": False,
        }
        des = _desc(semantic_res, col)

        if ref_sets and not_null_ratio(s) > 0.2 and gate.get("has_gate", False):

            # A) Level-gated matching
            if gate.get("levels"):
                filtered = {
                    lvl: ref_sets[lvl]
                    for lvl in gate["levels"]
                    if lvl in ref_sets
                }

                if filtered:
                    if is_numeric_series(s):
                        s = s.apply(
                            lambda x: str(int(x))
                            if pd.notna(x) and float(x).is_integer()
                            else str(x)
                            if pd.notna(x)
                            else pd.NA
                        ).astype("object")

                    best = match_series_to_ref_levels(
                        s, filtered, sample_size=sample_size
                    )

                    if best and best["ratio"] >= min_ratio:
                        results["spatial"].append({
                            "columns": col,
                            "description": des,
                            "type": str(s.dtype),
                            "granularity": best["level"],
                            "matched_by": best["by"],
                            "confidence": min(
                                0.99, 0.7 + 0.3 * best["ratio"]
                            ),
                            "evidence": (
                                f"Name gate [{', '.join(gate['levels'])}] → "
                                f"ref match by {best['by']} "
                                f"({round(best['ratio'] * 100)}%)."
                            ),
                        })
                        assigned = True
                        continue

                    elif best:
                        results["other"].append({
                            "columns": col,
                            "description": des,
                            "type": str(s.dtype),
                            "granularity": best["level"],
                            "matched_by": best["by"],
                        })
                        assigned = True
                        continue
                    # best is None → fall through to other

            # B) Generic fallback matching
            elif gate.get("generic", False):
                best = (
                    match_series_to_ref_levels(
                        s, ref_sets, sample_size=sample_size
                    )
                    if ref_sets
                    else None
                )
                generic_min = max(min_ratio, 0.80)

                if best and best["ratio"] >= generic_min:
                    results["spatial"].append({
                        "columns": col,
                        "description": des,
                        "type": str(s.dtype),
                        "granularity": best["level"],
                        "matched_by": best["by"],
                        "confidence": min(
                            0.99, 0.68 + 0.32 * best["ratio"]
                        ),
                        "evidence": (
                            f"Generic geo gate → ref match by {best['by']} "
                            f"({round(best['ratio'] * 100)}%)."
                        ),
                    })
                    assigned = True
                    continue

                elif best:
                    results["other"].append({
                        "columns": col,
                        "description": des,
                        "type": str(s.dtype),
                        "granularity": best["level"],
                        "matched_by": best["by"],
                    })
                    assigned = True
                    continue
                # best is None → fall through to other

        # ---- Unified fallback ----
        if not assigned:
            results["other"].append({
                "columns": col,
                "description": des,
                "type": str(s.dtype),
                "granularity": "",
                "matched_by": "",
            })

    # 2) Temporal (always "columns":[col])
    for col in df_temporal.columns:
        des = _desc(semantic_res, col)
        s = df_temporal[col]
        gran, conf = detect_temporal_granularity(s)
        if gran:
            results["temporal"].append({
                "columns": col,
                "description": des,
                "type": str(s.dtype),
                "granularity": gran,
                "confidence": conf
            })
        else:
            results["other"].append({
                "columns": col,
                "description": des,
                "type": str(s.dtype),
                "granularity": None,
                "confidence": None
            })

    # 3) Indicators — builders may return 'column'; normalize to 'columns'
    for col in df_indicator_quant.columns:
        s = df_indicator_quant[col]
        entry = build_quantitative_entry(col, s, semantic_res)
        if entry:
            entry = dict(entry)
            entry["columns"] = _ensure_columns_name_list(entry.get("columns", col))
            entry.pop("column", None)
            results["indicators"].append(entry)

    for col in df_indicator_qual.columns:
        s = df_indicator_qual[col]
        entry = build_qualitative_entry(col, s, semantic_res)
        if entry:
            entry = dict(entry)
            entry["columns"] = _ensure_columns_name_list(entry.get("columns", col))
            entry.pop("column", None)
            results["indicators"].append(entry)

    # 4) Other (pure semantic 'other')
    for col in df_other.columns:
        s = df_other[col]
        des = _desc(semantic_res, col)
        avg_len = None
        try:
            avg_len = float(s.dropna().astype(str).str.len().mean())
        except Exception:
            pass
        results["other"].append({
            "columns": col,
            "description": des,
            "type": str(s.dtype),
            "theme": get_semantic_value(semantic_res, col, "thematic_path", default=None),
            "reason": "Semantically detected as other information",
            "avg_text_len": avg_len
        })

    # 5) Meta (final snapshot)
    results["meta"]["config"] = {
        "min_ratio": min_ratio,
        "sample_size": sample_size,
        "include_city_zip_in_address": include_city_zip_in_address,
        "address_colname": address_colname,
        "filename": filename,
        "sheet_names": sheet_names or [],
        "require_name_hint_for_geoformats": require_name_hint_for_geoformats,
    }

    if not any(att.get("granularity") == "country" for att in results["spatial"]):
        results["spatial"].insert(0, dict(DEFAULT_COUNTRY_SPATIAL_ATTRIBUTE))

    return results
