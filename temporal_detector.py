from general_function import *
from reference import TEMP_NAME_PATTERNS

def is_datetime_column(s: pd.Series, threshold: float = 0.9) -> bool:
    """
    判断 Series 是否为时间列。
    threshold: 至少有多少比例的值能成功解析为时间，才算时间列。
    """
    parsed = pd.to_datetime(s, errors="coerce", utc=True)
    success_ratio = parsed.notna().mean()   # 成功解析的比例
    return success_ratio >= threshold


def detect_temporal_granularity(col: pd.Series) -> Tuple[str | None, float]:
    """Return ('year'|'quarter'|'month'|'week'|'day'|None, confidence)."""
    if pd.api.types.is_datetime64_any_dtype(col):
        s = pd.to_datetime(col.dropna(), errors="coerce")
        if s.empty:
            return (None, 0.0)
        s = s.dropna()
        if len(s) == 0:
            return (None, 0.0)
        if (s.dt.month.eq(1) & s.dt.day.eq(1)).mean() >= 0.95:
            return ("year", 0.95)
        if s.dt.day.eq(1).mean() >= 0.95:
            return ("month", 0.95)
        try:
            if (s.dt.weekday.eq(0)).mean() >= 0.95:
                return ("week", 0.8)
        except Exception:
            pass
        return ("date", 0.8)

    if is_numeric_series(col):
        s = col.dropna().astype(int)
        if s.empty:
            return (None, 0.0)
        if ((s >= 1800) & (s <= 2100)).mean() >= 0.95:
            return ("year", 0.95)
        if ((s >= 180001) & (s <= 210012)).mean() >= 0.95:
            months = (s % 100).between(1, 12).mean()
            if months >= 0.95:
                return ("month", 0.9)
        return (None, 0.0)

    if is_string_series(col):
        sample = col.dropna().astype(str).str.strip().head(300)

        if sample.apply(lambda x: bool(re.fullmatch(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$", x))).mean() >= 0.9:
            return ("date", 0.95)
        if sample.apply(lambda x: bool(re.fullmatch(r"\d{4}-\d{2}", x))).mean() >= 0.9:
            return ("month", 0.95)
        if sample.apply(lambda x: bool(re.fullmatch(r"\d{4}", x)) and 1800 <= int(x) <= 2100).mean() >= 0.9:
            return ("year", 0.95)
        if sample.apply(lambda x: bool(re.fullmatch(r"\d{4}-W\d{2}", x))).mean() >= 0.8:
            return ("week", 0.9)
        if sample.apply(lambda x: bool(re.fullmatch(r"\d{4}[-]?Q[1-4]", x))).mean() >= 0.8:
            return ("quarter", 0.9)

    return (None, 0.0)

# ---------- Temporal name-based extraction (filename/colname/sheetname) -----

def _extract_temporal_from_text(text: str) -> List[dict]:
    """Extract temporal hints from arbitrary text (filename/colname/sheetname)."""
    t = normalise_colname(text)
    hits = []
    for gran, pat in TEMP_NAME_PATTERNS:
        for m in pat.finditer(t):
            d = {"granularity": gran}
            gd = m.groupdict()
            if gd.get("y"): d["year"] = int(gd["y"])
            if gd.get("m"): d["month"] = int(gd["m"])
            if gd.get("q"): d["quarter"] = int(gd["q"])
            if gd.get("s"): d["semester"] = int(gd["s"])
            if gd.get("w"): d["week"] = int(gd["w"])
            if gd.get("d"): d["date"] = int(gd["d"])
            hits.append(d)
    return hits

def _best_temporal_from_text(text: str) -> Tuple[str | None, dict, float]:
    """Return the most specific temporal granularity from text."""
    matches = _extract_temporal_from_text(text)
    if not matches:
        return (None, {}, 0.0)
    rank = {"date": 4, "month": 3, "quarter": 3, "semester": 3, "week": 2, "year": 1}
    best = max(matches, key=lambda d: rank[d["granularity"]])
    conf_map = {"date": 0.98, "month": 0.96, "quarter": 0.95, "semester": 0.95, "week": 0.93, "year": 0.9}
    return (best["granularity"], best, conf_map[best["granularity"]])

def detect_temporal_from_colname(name: str) -> Tuple[str | None, float, dict]:
    """
    Infer temporal granularity purely from a column's *name*.
    Returns (granularity, confidence, payload).
    """
    g, payload, conf = _best_temporal_from_text(name)
    return (g, conf, payload)