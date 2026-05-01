import os
import json
import io
import pandas as pd
from openai import OpenAI

from reference import (
    THEME_FOLDER_STRUCTURE,
    REF_SEMANTIC,
    minio_key,
    object_exists,
    read_json_from_minio,
    s3,
    LAKE_BUCKET,
)
from general_function import normalise_colname


# ---------------------------------------------------------
# Lazy OpenAI client
# ---------------------------------------------------------

_client = None


def get_openai_client() -> OpenAI:
    """
    Lazily create the OpenAI client only when it is actually needed.

    This avoids import-time failure when OPENAI_API_KEY is not yet set.
    """
    global _client

    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please set it before calling semantic_helper()."
            )
        _client = OpenAI(api_key=api_key)

    return _client


# ---------------------------------------------------------
# Backbone storage paths
# ---------------------------------------------------------

def backbone_path_for_series(series_hint: str):
    """
    Return the MinIO object key path for one dataset-series semantic backbone.
    """
    return REF_SEMANTIC / f"{series_hint}__backbone.json"


def load_semantic_backbone(series_hint: str) -> dict:
    """
    Load a series semantic backbone from MinIO if it exists.

    The expected structure is:
    {
      "series_hint": "...",
      "columns": {
        "normalized_col_name": {...}
      }
    }
    """
    if not series_hint:
        return {}

    path = backbone_path_for_series(series_hint)

    if not object_exists(path):
        return {}

    raw = read_json_from_minio(path)
    data = json.loads(raw.decode("utf-8"))

    if isinstance(data, dict) and "columns" in data and isinstance(data["columns"], dict):
        return data["columns"]

    return {}


def save_semantic_backbone(series_hint: str, backbone: dict) -> None:
    """
    Save a series semantic backbone to MinIO.
    """
    if not series_hint:
        return

    path = backbone_path_for_series(series_hint)

    payload = {
        "series_hint": series_hint,
        "columns": backbone
    }

    s3.put_object(
        Bucket=LAKE_BUCKET,
        Key=minio_key(path),
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json"
    )


# ---------------------------------------------------------
# Prompt building
# ---------------------------------------------------------

def build_semantic_prompt(samples: dict) -> str:
    """
    Build the GPT prompt for semantic classification of one batch of columns.
    """
    return f"""
You are a rigorous data steward. Classify each column using header names and sample values.

Allowed classes (mutually exclusive; exactly one must be True):
- Spatial: encodes a location (latitude/longitude, X/Y, address, admin code, region/department names or codes).
- Temporal: encodes time or period (year, date, month, quarter, week, YYYY, YYYY-MM, timestamps).
- Indicator: a measured variable used for analysis/monitoring (counts, rates, ratios, scores, categories, yes/no flags).
- Other information: ONLY if it is NOT spatial, NOT temporal, and NOT an indicator (e.g., labels, IDs, free text, entity attributes).

Tasks per column:
1) Explain the meaning (one short sentence).
2) Set is_spatial (bool).
3) Set is_temporal (bool).
4) Set is_indicator (bool).
5) Set is_other_information (bool) = True only if the first three are all False.
6) If is_indicator is True, set indicator_type to "Quantitative" or "Qualitative"; otherwise null.
7) If is_indicator is True, assign a theme from the given hierarchy.
8) If is_other_information is True:
   - If it is a supplementary field for Spatial or Temporal, set thematic_path = null.
   - Otherwise, assign a full thematic path from the hierarchy.

Thematic hierarchy (use exactly as provided; do NOT invent nodes):
{THEME_FOLDER_STRUCTURE}

Column names and sample data:
{samples}

STRICT formatting of thematic_path:
    - MUST start with "Well-being" (exact spelling and casing).
    - MUST be the FULL PATH from the root to the most specific applicable leaf.
    - Join levels with " > " and NO trailing/leading spaces.
    - Use ONLY nodes that exist in the provided hierarchy, preserving order and spelling.
    - If you truly cannot assign a valid path beginning with "Well-being", return null. This null is NOT allowed when is_other_information is True for a non-spatial/non-temporal auxiliary field.

Decision steps (apply in order for each column):
A) Detect class:
   - If matches location patterns (geo codes/names, lon/lat, address): Spatial.
   - Else if matches time patterns (year, date, month, etc.): Temporal.
   - Else if looks like a measure (numeric or categorical outcome used for analysis): Indicator.
   - Else: Other information.
B) Thematic mapping:
   - If Indicator → REQUIRED to map to the deepest applicable leaf in "Well-being ...".
   - If Other information:
       * If clearly auxiliary to Spatial/Temporal (e.g., geocoding quality, date parsing source), thematic_path = null.
       * Else REQUIRED to map to the deepest applicable leaf in "Well-being ...".
   - If Spatial or Temporal → thematic_path = null.

Common mappings (examples; use only if relevant and the node exists in the provided hierarchy):
- Healthcare professional specialty → Well-being > Current Well-being > Health > Health systems & services > Workforce & resources
- Hospital admission date → Well-being > Current Well-being > Health > Access to care
- Chronic disease flag → Well-being > Current Well-being > Health > Physical health > Disease burden
- Self-reported anxiety/depression score → Well-being > Current Well-being > Health > Mental health > Mental state

- Education program type → Well-being > Current Well-being > Education & Skills > Skills & learning
- Literacy test result → Well-being > Current Well-being > Education & Skills > Educational outcomes > Performance > Literacy

- Household disposable income → Well-being > Current Well-being > Income & Wealth > Income > Household income
- Net wealth of household → Well-being > Current Well-being > Income & Wealth > Wealth > Net wealth

- Employment contract type → Well-being > Current Well-being > Jobs & Earnings > Job quality > Stability
- Weekly working hours → Well-being > Current Well-being > Jobs & Earnings > Job quality > Working conditions

- Housing occupancy status → Well-being > Current Well-being > Housing > Housing conditions
- Housing cost burden → Well-being > Current Well-being > Housing > Housing affordability > Cost burden

- PM2.5 concentration → Well-being > Current Well-being > Environment Quality > Environmental exposure > Air quality
- Green space availability → Well-being > Current Well-being > Environment Quality > Perceptions & access > Green space accessibility

- Recorded crime rate → Well-being > Current Well-being > Safety > Personal safety > Crime incidence
- Road traffic injury counts → Well-being > Current Well-being > Safety > Road safety > Traffic injuries

- Voter turnout percentage → Well-being > Current Well-being > Civic Engagement & Governance > Participation > Voter turnout
- Trust in government score → Well-being > Current Well-being > Civic Engagement & Governance > Trust & satisfaction > Institutional trust

- Number of close friends reported → Well-being > Current Well-being > Social Connections > Social support > Reliance network
- Participation in community events → Well-being > Current Well-being > Social Connections > Social participation > Community participation

- Life satisfaction score → Well-being > Current Well-being > Subjective Well-being > Life satisfaction
- Positive affect index → Well-being > Current Well-being > Subjective Well-being > Affective balance > Positive affect

- Average commuting time → Well-being > Current Well-being > Work-life Balance > Commuting time
- Hours spent on unpaid work → Well-being > Current Well-being > Work-life Balance > Unpaid work

- Religious affiliation → Well-being > Current Well-being > Spirituality / Religion / Personal Beliefs

---

- Protected forest area share → Well-being > Resources for Future Well-being > Natural Capital > Ecosystems & biodiversity > Forest cover
- Renewable energy share of total → Well-being > Resources for Future Well-being > Natural Capital > Climate & sustainability > Renewable energy

- Child development index → Well-being > Resources for Future Well-being > Human Capital > Health stock > Child development
- Adult skills survey result → Well-being > Resources for Future Well-being > Human Capital > Education & skills stock > Adult skills

- Interpersonal trust index → Well-being > Resources for Future Well-being > Social Capital > Trust & norms > Interpersonal trust
- Gender equality measure → Well-being > Resources for Future Well-being > Social Capital > Inclusion & cohesion > Gender equality

- Public infrastructure investment → Well-being > Resources for Future Well-being > Economic & Produced Capital > Infrastructure & innovation > Fixed capital
- Adjusted net savings → Well-being > Resources for Future Well-being > Economic & Produced Capital > Wealth sustainability > Adjusted savings


Quality checks (HARD FAIL if violated):
- Exactly ONE of [is_spatial, is_temporal, is_indicator, is_other_information] is True.
- If is_indicator is False → indicator_type must be null.
- thematic_path must either be a valid full path starting with "Well-being" or null.
- If is_indicator is True → thematic_path is REQUIRED (not null).
- If is_other_information is True and the field is NOT a Spatial/Temporal auxiliary → thematic_path is REQUIRED (not null).

Return JSON strictly as:
{{
  "columns": [
    {{
      "column_name": "str",
      "meaning": "str",
      "is_spatial": true/false,
      "is_temporal": true/false,
      "is_indicator": true/false,
      "is_other_information": true/false,
      "indicator_type": "Quantitative" | "Qualitative" | null,
      "thematic_path": "str" | null
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------
# GPT call
# ---------------------------------------------------------

def call_gpt_for_semantic(batch_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Call GPT on one batch of columns and return semantic classification as a DataFrame.
    """
    client = get_openai_client()

    samples = batch_df.to_dict(orient="list")
    prompt = build_semantic_prompt(samples)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    content = resp.choices[0].message.content
    result = json.loads(content)
    return pd.DataFrame(result["columns"])


# ---------------------------------------------------------
# Lightweight consistency check for backbone reuse
# ---------------------------------------------------------

def _infer_sample_profile(series: pd.Series) -> str:
    """
    Infer a coarse data profile from sample values.

    Returns one of:
    - numeric
    - datetime_like
    - mostly_text
    - empty
    """
    s = series.dropna()
    if s.empty:
        return "empty"

    numeric_ratio = pd.to_numeric(s, errors="coerce").notna().mean()
    if numeric_ratio >= 0.8:
        return "numeric"

    dt_ratio = pd.to_datetime(s, errors="coerce").notna().mean()
    if dt_ratio >= 0.8:
        return "datetime_like"

    return "mostly_text"


def _backbone_entry_conflicts_with_sample(entry: dict, sample_series: pd.Series) -> bool:
    """
    Lightweight consistency check before inheriting semantic info from backbone.

    Reject inheritance if the current file sample strongly conflicts with the
    semantic class stored in the backbone.
    """
    profile = _infer_sample_profile(sample_series)

    is_temporal = bool(entry.get("is_temporal"))
    is_spatial = bool(entry.get("is_spatial"))
    is_indicator = bool(entry.get("is_indicator"))
    is_other = bool(entry.get("is_other_information"))

    if is_temporal and profile == "empty":
        return True

    if is_indicator and profile == "empty":
        return True

    if is_other and profile == "numeric":
        return True

    if is_spatial and profile == "empty":
        return True

    return False


# ---------------------------------------------------------
# Main semantic helper
# ---------------------------------------------------------

def semantic_helper(
    df: pd.DataFrame,
    model: str = "gpt-5-mini",
    sample_rows: int = 10,
    max_cols_per_batch: int = 30,
    series_hint: str | None = None,
    update_backbone: bool = False
) -> pd.DataFrame:
    """
    Semantic classification with optional series-level backbone reuse.

    Strategy
    --------
    1. Normalize current column names.
    2. If a backbone exists for the series, inherit semantic information for matched columns.
    3. Before inheritance, run a lightweight consistency check against current sample values.
    4. Only send unmatched or conflicting columns to GPT.
    5. Merge inherited and newly inferred semantic information.
    6. Optionally update the backbone with newly inferred columns.
    """
    df_sample = df.head(sample_rows).copy()
    original_columns = list(df_sample.columns)

    norm_map = {col: normalise_colname(col) for col in original_columns}
    backbone = load_semantic_backbone(series_hint) if series_hint else {}

    inherited_rows = []
    unmatched_cols = []

    for col in original_columns:
        norm_col = norm_map[col]

        if norm_col in backbone:
            entry = backbone[norm_col]
            sample_series = df_sample[col]

            if _backbone_entry_conflicts_with_sample(entry, sample_series):
                unmatched_cols.append(col)
                continue

            row = entry.copy()
            row["column_name"] = col
            row["semantic_source"] = "backbone"
            inherited_rows.append(row)
        else:
            unmatched_cols.append(col)

    gpt_rows = []
    if unmatched_cols:
        for i in range(0, len(unmatched_cols), max_cols_per_batch):
            batch_cols = unmatched_cols[i:i + max_cols_per_batch]
            batch_df = df_sample[batch_cols]
            batch_res = call_gpt_for_semantic(batch_df, model=model)
            batch_res["semantic_source"] = "file_inferred"
            gpt_rows.append(batch_res)

    inherited_df = pd.DataFrame(inherited_rows) if inherited_rows else pd.DataFrame()
    gpt_df = pd.concat(gpt_rows, ignore_index=True) if gpt_rows else pd.DataFrame()

    if not inherited_df.empty and not gpt_df.empty:
        out_df = pd.concat([inherited_df, gpt_df], ignore_index=True)
    elif not inherited_df.empty:
        out_df = inherited_df
    elif not gpt_df.empty:
        out_df = gpt_df
    else:
        out_df = pd.DataFrame(columns=[
            "column_name",
            "meaning",
            "is_spatial",
            "is_temporal",
            "is_indicator",
            "is_other_information",
            "indicator_type",
            "thematic_path",
            "semantic_source"
        ])

    if not out_df.empty:
        out_df["column_name_normalized"] = out_df["column_name"].apply(normalise_colname)

    # Optional backbone update
    if series_hint and update_backbone and not gpt_df.empty:
        for _, row in gpt_df.iterrows():
            norm_col = normalise_colname(row["column_name"])
            backbone[norm_col] = {
                "column_name": row["column_name"],
                "column_name_normalized": norm_col,
                "meaning": row["meaning"],
                "is_spatial": bool(row["is_spatial"]),
                "is_temporal": bool(row["is_temporal"]),
                "is_indicator": bool(row["is_indicator"]),
                "is_other_information": bool(row["is_other_information"]),
                "indicator_type": row["indicator_type"],
                "thematic_path": row["thematic_path"]
            }

        save_semantic_backbone(series_hint, backbone)

    return out_df