from reference import TEMP_NAME_PATTERNS
from datetime import date, timedelta
from general_function import *
import calendar


def get_scope(granularities, hierarchies) -> List[str]:
    """Determine the highest (most general) matched level(s) for the given dataset.

    Logic:
    1) For each hierarchy path, pick the most general level that appears in `granularities`.
    2) If ALL picked levels belong to a single path, collapse to that path's single
       most general match (a one-element list).
    3) Otherwise, return the per-path picks (deduplicated, order-preserving).

    Returns:
        List[str]: Either a single most-general level (if all picks lie on one path),
                   or multiple levels (one per path) when matches span multiple paths.
    """
    # Step 1: pick most-general match per path
    picks: List[str] = []
    for path in hierarchies:
        top = get_most_general_in_path(granularities, path)
        if top:
            picks.append(top)

    if not picks:
        return []

    # Step 2: check if all picks lie within a single path
    pickset: Set[str] = set(picks)
    for path in hierarchies:
        path_names = {level_name(lvl) for lvl in path}
        if pickset.issubset(path_names):
            # Collapse to that single path's most-general match
            collapsed = get_most_general_in_path(granularities, path)
            return [collapsed] if collapsed else []

    # Step 3: return per-path picks (deduplicated, order-preserving)
    seen: Set[str] = set()
    out: List[str] = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _bounds(granularity: str, y: int, m=None, d=None, q=None, s=None, w=None):
    """Return the start and end date of a given period for the specified granularity."""
    if granularity == "year":
        return date(y,1,1), date(y,12,31)
    if granularity == "quarter":
        sm = (q-1)*3 + 1
        em = sm + 2
        return date(y,sm,1), date(y,em,calendar.monthrange(y,em)[1])
    if granularity == "semester":
        sm, em = (1,6) if s == 1 else (7,12)
        return date(y,sm,1), date(y,em,calendar.monthrange(y,em)[1])
    if granularity == "month":
        return date(y,m,1), date(y,m,calendar.monthrange(y,m)[1])
    if granularity == "week":
        return date.fromisocalendar(y,w,1), date.fromisocalendar(y,w,7)
    if granularity == "date":
        d0 = date(y,m,d)
        return d0, d0
    raise ValueError(granularity)

def _label(granularity: str, y: int, m=None, d=None, q=None, s=None, w=None) -> str:
    """Return the normalised string label for a given period and granularity."""
    if granularity == "year":
        return f"{y:04d}"
    if granularity == "quarter":
        return f"{y:04d}-Q{q}"
    if granularity == "semester":
        return f"{y:04d}-S{s}"
    if granularity == "month":
        return f"{y:04d}-{m:02d}"
    if granularity == "week":
        return f"{y:04d}-W{w:02d}"
    if granularity == "date":
        return f"{y:04d}-{m:02d}-{d:02d}"
    raise ValueError(granularity)

def _parse_one(token: str, granularity: str):
    """
    Parse a single token and return:
      {
        'start': start_date,
        'end':   end_date,
        'label': normalised string label
      }
    Returns None if the token cannot be parsed.
    """
    s = str(token).strip()
    # Special case: pure year as int or numeric string
    if granularity == "year" and s.isdigit() and len(s) == 4 and s.startswith(("18","19","20")):
        y = int(s)
        st, ed = _bounds("year", y)
        return {'start': st, 'end': ed, 'label': _label("year", y)}

    for g, rx in TEMP_NAME_PATTERNS:
        if g != granularity:
            continue
        m = rx.search(s)
        if not m:
            continue
        gd = {k: (int(v) if v is not None else None) for k,v in m.groupdict().items()}
        try:
            if granularity == "year":
                y = gd["y"]
                st, ed = _bounds("year", y)
                lab = _label("year", y)
            elif granularity == "quarter":
                y, q = gd["y"], gd["q"]
                if not (1 <= q <= 4): return None
                st, ed = _bounds("quarter", y, q=q)
                lab = _label("quarter", y, q=q)
            elif granularity == "semester":
                y, s_ = gd["y"], gd["s"]
                if s_ not in (1,2): return None
                st, ed = _bounds("semester", y, s=s_)
                lab = _label("semester", y, s=s_)
            elif granularity == "month":
                y, m_ = gd["y"], gd["m"]
                if not (1 <= m_ <= 12): return None
                st, ed = _bounds("month", y, m=m_)
                lab = _label("month", y, m=m_)
            elif granularity == "week":
                y, w = gd["y"], gd["w"]
                st, ed = _bounds("week", y, w=w)  # ISO calendar check
                lab = _label("week", y, w=w)
            elif granularity == "date":
                y, m_, d_ = gd["y"], gd["m"], gd["d"]
                st, ed = _bounds("date", y, m=m_, d=d_)
                lab = _label("date", y, m=m_, d=d_)
            else:
                return None
            return {'start': st, 'end': ed, 'label': lab}
        except Exception:
            # Invalid date/week combination (e.g. 2021-02-30, 2021-W54)
            return None
    return None

def extract_label_ranges(granularity: str, tokens):
    """
    Extract consecutive ranges of the same granularity.

    Input:
      - tokens: iterable of time strings (or ints for years)
      - granularity: one of 'year' | 'quarter' | 'semester' | 'month' | 'week' | 'date'

    Output:
      - list of (start_label, end_label) tuples, sorted chronologically

    Rules:
      - Merge only *consecutive* units of the same granularity
      - Do NOT promote to a higher granularity
      - Deduplicate tokens and ignore invalid ones
    """
    granularity = granularity.lower()
    items = []
    seen = set()  # avoid duplicates based on (start, end) dates
    for t in tokens:
        rec = _parse_one(t, granularity)
        if rec:
            key = (rec['start'], rec['end'])
            if key not in seen:
                items.append(rec)
                seen.add(key)
    if not items:
        return []

    # Sort by start date
    items.sort(key=lambda r: (r['start'], r['end']))

    # Merge consecutive units
    ranges = []
    cur_start_label = items[0]['label']
    cur_end_label   = items[0]['label']
    cur_end_date    = items[0]['end']

    for r in items[1:]:
        if r['start'] == cur_end_date + timedelta(days=1):
            # Consecutive → extend right boundary
            cur_end_date  = r['end']
            cur_end_label = r['label']
        else:
            # Break in continuity → close current range
            ranges.append((cur_start_label, cur_end_label))
            cur_start_label = r['label']
            cur_end_label   = r['label']
            cur_end_date    = r['end']
    ranges.append((cur_start_label, cur_end_label))
    return ranges

