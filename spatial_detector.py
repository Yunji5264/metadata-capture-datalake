import binascii
import struct
import re

from collections import Counter
from itertools import combinations
from general_function import *
from reference import SPATIAL_NAME_MAP
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

# Adress multi-column handling

def find_address_cols(df: pd.DataFrame, include_city_zip: bool = False) -> List[str]:
    """Detect address sub-columns and return them in a canonical order for concatenation."""
    # --- Role-specific patterns (French admin datasets) ---------------------
    role_patterns = {
        # --- Thoroughfare core (street-level components) ---
        "house_number": [
            r"^(num(?:ero)?(?:_?voie)?|no_?voie|num_?voie|numvoie|n(?:o|um))$",
        ],
        "repetition_index": [  # e.g., BIS, TER, QUATER
            r"^(indrep|indice(_?de)?_?repet(?:ition)?)$",
        ],
        "way_type": [
            r"^(typ(?:e)?_?voie|type_?rue|typvoie)$",
            r"^(nature_?voie|nat_?voie|natvoie)$",
            r"^(type(_?de)?_?voie|typevoie|liblib(?:elle)?_?type_?voie|libtypevoie)$",
        ],
        "way_name": [
            r"^(nom_?voie|libelle_?voie|liblib(?:elle)?_?voie|nomvoie|nom_?rue)$",
            r"^(libelle(_?de)?_?voie|liblib(?:elle)?_?de_?voie|voie_?libelle)$",
            r"^(nom(_?de)?_?voie)$",
        ],

        # --- Building / unit details ---
        "building": [
            r"^(bat(?:iment)?|batiment|immeu(?:ble)?|immeuble|tour|bloc)$",
            r"^(bt|bt_?num|bat_?num(?:ero)?)$",
        ],
        "entrance": [
            r"^(entree|ent)$",
        ],
        "stair": [
            r"^(esc(?:alier)?)$",
        ],
        "floor": [
            r"^(etage|niveau)$",
            r"^(etg|etage_?num)$",
        ],
        "corridor": [
            r"^(couloir|coul)$",
        ],
        "unit_number": [
            r"^(num(app?t|apt|appartement|logt|logement)|app?t|apt|porte)$",
            r"^(num_?apt|apt_?num|num_?app?t|app?t_?num)$",
            r"^(num_?logt|logt_?num|num_?logement|logement_?num)$",
            r"^(porte_?num|num_?porte)$",
            r"^(lot|lot_?num)$",
        ],
        "mailbox": [
            r"^(numboite|boite(_?lettres)?|bp|cs)$",
            r"^(boite_?postale|bp_?\d*)$",
        ],

        # --- Complements / generic address lines ---
        "complement": [
            r"^(compl(ement)?(_?(adresse|ident|geo))?)$",
            r"^(lieu(_|-)?dit|residence|resid|quartier)$",
            r"^(compl(?:ement)?(?:_?\d+)?)$",  # generic complement_n
            r"^(adresse_?compl(?:ement)?|compl(?:ement)?_?adresse(?:_?\d+)?)$",
        ],
        "address_line": [
            r"^(adresse|address|addr)(_?\d+)?$",
            r"^(ligne|line)_?\d+$",
            r"^(addr(?:ess)?_?line_?\d+)$",
            r"^(address_?\d+)$",
        ],

        # --- City / ZIP (optional to include in full address) ---
        "postal_code": [
            r"^(cp|code(_|-)?postal|postal(_|-)?code|zip)$",
            r"^(post_?code|postcode|code_?post(?:al)?)$",
            r"^(cedex)$",
        ],
        "city": [
            r"^(ville|commune|localite|city|town)$",
            r"^(arrondissement|arr|arrdt)$",  # large-city subdivisions (e.g., Paris 1er)
        ],

        # --- Direct thoroughfare tokens (single-field street descriptors) ---
        "token": [
            r"^(voie|rue|avenue|av|bd|boulevard|bld|chemin|route|impasse|allee|place|quai|cours)$",
            r"^(square|sq|ruelle|villa|cite|sente|promenade|pl)$",
            r"^(quartier|lieu(_|-)?dit)$",
            r"^(bd)$",  # keep short alias explicitly (some datasets use it as the whole column name)
        ],
    }

    # Compile all regexes once
    role_res = {role: [re.compile(p, re.I) for p in pats] for role, pats in role_patterns.items()}

    # Canonical output order
    order = [
        "house_number", "repetition_index", "way_type", "way_name",          # line 1
        "building", "entrance", "stair", "floor", "corridor", "unit_number", "mailbox",  # line 2
        "complement", "address_line", "thoroughfare_token"                   # complements / generic lines
    ]
    if include_city_zip:
        order += ["postal_code", "city"]

    # Match roles while preserving the original column order per role
    matched: Dict[str, List[str]] = {k: [] for k in role_patterns.keys()}
    for col in df.columns:
        norm = normalise_colname(col)
        for role, regs in role_res.items():
            if any(r.match(norm) for r in regs):
                matched[role].append(col)
                break  # stop at the first role that matches (highest specificity wins)

    # Flatten in canonical order and deduplicate while preserving order
    cols: List[str] = []
    seen: set = set()
    for role in order:
        for c in matched.get(role, []):
            if c not in seen:
                cols.append(c)
                seen.add(c)

    return cols

def concat_columns_safe(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Concatenate multiple columns into a single string series with spaces."""
    if not cols:
        return pd.Series([None]*len(df), index=df.index, dtype="object")
    s = pd.Series([""]*len(df), index=df.index, dtype="object")
    for c in cols:
        part = df[c].astype(str).where(df[c].notna(), "")
        s = s.str.cat(part, sep=" ")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.mask(s.eq(""))
    return s

def add_combined_address_column(
    df: pd.DataFrame,
    colname: str = "__address__",
    include_city_zip: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a copy with aggregated address column; drop original address sub-columns."""
    addr_cols = find_address_cols(df, include_city_zip=include_city_zip)
    df2 = df.copy()
    if addr_cols:
        df2[colname] = concat_columns_safe(df2, addr_cols)
        df2 = df2.drop(columns=addr_cols)
    return df2, addr_cols

# --- Column-name first: spatial level hints ---------------------------------
SEP = r"[_\s\-]*"  # 统一允许下划线/空格/连字符

_SPATIAL_COLNAME_HINTS = [
    # région / region
    ("reg", re.compile(
        rf"\b(?:"
        rf"reg|région|region|régional(?:e|es)?|regional(?:e|es)?"
        rf"|state|province|area"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}(?:reg(?:ion)?|rég(?:ion)?|region)"
        rf")\b", re.I)),

    # département / department
    ("dep", re.compile(
        rf"\b(?:"
        rf"dep|dpt|département|departement|départemental(?:e|es)?|department|county"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}(?:dep(?:art(?:ement)?)?|département|departement)"
        rf")\b", re.I)),

    # arrondissement départemental / district
    ("arr_dep", re.compile(
        rf"\b(?:"
        rf"arr(?:ondiss(?:ement)?)?|arrondissement|district|subdistrict"
        rf"|arr(?:ondiss(?:ement)?)?{SEP}(?:dep|dpt|département|departement)"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}arr{SEP}(?:dep|dpt|département|departement)"
        rf"|arr{SEP}dep"
        rf")\b", re.I)),

    # canton
    ("canton", re.compile(
        rf"\b(?:"
        rf"canton|cantonal(?:e|es)?|ward|precinct"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}canton"
        rf")\b", re.I)),

    # epci / intercommunality
    ("epci", re.compile(
        rf"\b(?:"
        rf"epci|siren{SEP}epci"
        rf"|intercommunal(?:ité|ity)?|métropole|metropolitan(?:{SEP}area)?|federation"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}epci"
        rf")\b", re.I)),

    # académie / academy
    ("academie", re.compile(
        rf"\b(?:"
        rf"académie|academie|aca|academy|school{SEP}district|education{SEP}region"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}(?:aca(?:demie)?|académie|academie)"
        rf")\b", re.I)),

    # commune / municipality
    ("com", re.compile(
        rf"\b(?:"
        rf"com|commune|communal(?:e|es)?|municipality|town|village|city"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}(?:com(?:mune)?|commune)"
        rf")\b", re.I)),

    # commune et arrondissement municipal (Paris, Lyon, Marseille)
    ("com_arr", re.compile(
        rf"\b(?:"
        rf"commune|municipality|borough|city{SEP}district"
        rf"|arrondiss(?:ement)?{SEP}mun|arrondissement{SEP}municipal|com{SEP}arr"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}(?:com(?:mune)?|commune)"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}com{SEP}arr"
        rf")\b", re.I)),

    # iris / neighbourhood
    ("iris", re.compile(
        rf"\b(?:"
        rf"iris|neighbou?rhood|block"
        rf"|(?:code|insee|nom|lib(?:elle)?){SEP}iris"
        rf")\b", re.I)),
]


def spatial_level_hints_from_colname(name: str) -> list[str]:
    """Return canonical spatial levels suggested by a column name."""
    n = normalise_colname(name)
    hits_short = [key for key, pat in _SPATIAL_COLNAME_HINTS if pat.search(n)]
    # map short labels (reg/dep/com/...) to canonical keys expected in ref_sets
    # hits = [_LEVEL_ALIAS[h] for h in hits_short]
    # de-duplicate while preserving order
    seen, out = set(), []
    for h in hits_short:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

# Spatial detector without reference
def _to_text_safe(x, encodings=("utf-8", "cp1252", "latin1")) -> str | None:
    """Safely convert any scalar to text; decode bytes with fallbacks."""
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        for enc in encodings:
            try:
                return x.decode(enc)
            except UnicodeDecodeError:
                continue
        return x.decode(encodings[0], errors="replace")
    try:
        return str(x)
    except Exception:
        return None

def _to_text_series_safe(s: pd.Series) -> pd.Series:
    """Elementwise safe text conversion that preserves the index."""
    return s.map(_to_text_safe)


# Matches only clean hex (no spaces) with even length
_HEX_RE = re.compile(r'^[0-9A-Fa-f]+$')

# Matches a sequence of "\xNN" escapes, require at least 5 bytes to look like WKB header+ (cheap heuristic)
_ESCAPED_HEX_RE = re.compile(r'(?:\\x[0-9A-Fa-f]{2}){5,}')

# Quick set of base WKB geometry types (OGC); EWKB may set high bits for Z/M/SRID
_WKB_BASE_TYPES = {1, 2, 3, 4, 5, 6, 7}


def _validate_wkb_header(buf: bytes) -> bool:
    """
    Cheap sanity check for WKB/EWKB:
    - length >= 5 (endian + u32 geom type)
    - first byte is 0 or 1 (big/little endian)
    - geometry type (masked) in known set
    Note: EWKB may set high bits (Z/M/SRID flags). We mask them out.
    """
    if not buf or len(buf) < 5:
        return False
    endian = buf[0]
    if endian not in (0, 1):
        return False
    # choose endian format
    fmt = "<I" if endian == 1 else ">I"
    geom_type = struct.unpack(fmt, buf[1:5])[0]
    # EWKB flags (PostGIS): Z=0x80000000, M=0x40000000, SRID=0x20000000
    base_type = geom_type & 0xFF
    if base_type not in _WKB_BASE_TYPES:
        return False
    return True


def _looks_like_binary_str(s: str) -> bool:
    """
    Fast check: does the string already contain non-printable or high-bit chars?
    If yes, encoding to latin1 is very cheap and likely intended.
    """
    # consider printable ASCII range 32..126 plus common whitespace \t\n\r
    for ch in s:
        o = ord(ch)
        if o < 9 or (13 < o < 32) or o > 126:  # exclude \t(9), \n(10), \r(13)
            return True
    return False


def as_wkb_bytes(v) -> bytes | None:
    """
    Return bytes if value looks like WKB:
    - bytes/bytearray/memoryview → returned directly (after header sanity check)
    - clean hex string → unhexlify → validated
    - string with literal backslash escapes (e.g. "\\x01\\x01...") → unicode_escape → latin1 → validated
    - string already containing binary-ish chars (e.g. smart quotes, high-bit glyphs, control bytes) → latin1 → validated
    Else return None.

    Designed to be fast in negative cases and avoid heavy decoding unless the pattern strongly suggests WKB.
    """
    if v is None:
        return None

    # Case 0: already bytes-like
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        return b if _validate_wkb_header(b) else None

    if isinstance(v, str):
        s = v

        # Case 1: clean hex (even length)
        if len(s) % 2 == 0 and _HEX_RE.fullmatch(s):
            try:
                b = binascii.unhexlify(s)
            except binascii.Error:
                return None
            return b if _validate_wkb_header(b) else None

        # Case 2: literal "\xNN" escapes (e.g. read from CSV/JSON as text)
        # Use regex guard to avoid applying unicode_escape to arbitrary long texts
        if _ESCAPED_HEX_RE.search(s):
            try:
                # Step A: interpret backslash escapes into actual code points
                # Note: codecs.decode(s, 'unicode_escape') would also work; using encode+decode keeps types explicit.
                decoded = s.encode('latin1', errors='strict').decode('unicode_escape')
                # Step B: 1:1 map to bytes
                b = decoded.encode('latin1', errors='strict')
            except (UnicodeEncodeError, UnicodeDecodeError):
                b = None
            if b and _validate_wkb_header(b):
                return b  # success

        # Case 3: already “binary-ish” string (contains control/high-bit chars)
        if _looks_like_binary_str(s):
            try:
                b = s.encode('latin1', errors='strict')
            except UnicodeEncodeError:
                b = None
            if b and _validate_wkb_header(b):
                return b

    # Not recognized as WKB
    return None

def detect_wkt_geojson_string(s: pd.Series) -> bool:
    """Try external detect_wkt_geojson_string(s); fallback to safe text heuristics."""
    sample = s.dropna().head(1).apply(_to_text_safe).dropna()
    if sample.empty:
        return False
    for v in sample:
        t = v.strip()
        if t.startswith("{") and '"type"' in t and '"coordinates"' in t:
            return True
        U = t.upper()
        if U.startswith(("SRID=", "POINT", "LINESTRING", "POLYGON", "MULTI", "GEOMETRYCOLLECTION")):
            return True
        if as_wkb_bytes(v):
            return True
    return False

# def detect_geohash(s: pd.Series) -> bool:
#     """Detect geohash strings (base32 without a,i,l,o)."""
#     if not is_string_series(s):
#         return False
#     pat = re.compile(r"^[0123456789bcdefghjkmnpqrstuvwxyz]{5,}$")
#     sample = s.dropna().astype(str).str.strip().str.lower().head(200)
#     ok = sample.apply(lambda x: bool(pat.match(x)) and len(x) <= 12).mean()
#     return ok >= 0.7

def detect_geometry_object(s: pd.Series) -> bool:
    """Detect geometry-like Python objects (shapely/GeoSeries)."""
    if str(getattr(s, "dtype", "")).lower() == "geometry":
        return True
    sample = s.dropna().head(1)
    if sample.empty:
        return False
    v = sample.iloc[0]
    name = type(v).__name__.lower()
    if hasattr(v, "__geo_interface__") or any(k in name for k in ["polygon","point","linestring","multipolygon"]):
        return True
    return False

def detect_address(s: pd.Series) -> bool:
    """
    Heuristic address detector that is robust to bytes / mixed types.
    Returns True if at least a fraction of the sample looks address-like.
    """
    # Only proceed for object/string-like; otherwise try best-effort conversion
    if not (is_object_dtype(s) or is_string_dtype(s)):
        s = _to_text_series_safe(s)
    # Safe sample → lowercase text
    sample = _to_text_series_safe(s.dropna().head(200)).dropna()
    if sample.empty:
        return False
    txt = (sample.str.lower()
                 .str.normalize("NFKC")
                 .str.replace(r"[\u00A0\u202F]", " ", regex=True))

    # Basic address heuristics (EN/FR + generic)
    street_tokens = r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|route|rte\.?|rue|av\.?|bd|chemin|impasse|all[ée]e|place|square)\b"
    has_num   = txt.str.contains(r"\d")
    has_token = txt.str.contains(street_tokens, regex=True, na=False, flags=re.I)
    has_zip   = txt.str.contains(r"\b\d{4,5}\b")  # simple zip/postal pattern
    # Score: number+token OR token+zip
    score = ((has_num & has_token) | (has_token & has_zip)).mean()
    return bool(score >= 0.15)  # tweak threshold as needed

def _to_num(s: pd.Series) -> pd.Series:
    """
    Convert a Series to numeric robustly:
    - If already numeric dtype: coerce directly.
    - If object/string: safe-decode bytes, normalize spaces, try decimal comma/point.
    """
    if is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    # If not object/string (e.g., categorical), still try to coerce directly
    if not (is_object_dtype(s) or is_string_dtype(s)):
        return pd.to_numeric(s, errors="coerce")

    # Safe text conversion for object/string with possible bytes
    ts = _to_text_series_safe(s)

    # Normalize whitespace incl. NBSP (U+00A0) and NARROW NBSP (U+202F)
    norm = (
        ts.str.replace(r"[\u00A0\u202F]", "", regex=True)
          .str.replace(r"\s+", "", regex=True)
    )

    # Attempt 1: treat comma as decimal separator (e.g., "1,23" -> 1.23)
    nums1 = pd.to_numeric(norm.str.replace(",", "."), errors="coerce")

    # Attempt 2: handle "1.234,56" → remove thousands "." then convert comma to dot
    nums2 = pd.to_numeric(
        norm.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # Prefer nums1; fallback to nums2 where nums1 is NaN
    out = nums1.where(nums1.notna(), nums2)

    # Ensure index alignment with input
    out.index = s.index
    return out

def _frac_in_range(s: pd.Series, lo: float, hi: float) -> float:
    """Fraction of values within [lo, hi]."""
    sn = _to_num(s).dropna()
    return 0.0 if sn.empty else ((sn >= lo) & (sn <= hi)).mean()

def detect_coordinate_pairs_all(
    df: pd.DataFrame,
    min_score_geo: float = 0.7,
    min_score_proj: float = 0.7,
    enable_numeric_fallback: bool = False,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Detect ALL candidate coordinate pairs.

    Returns:
      {
        "geo":  [(lat_col, lon_col, score), ...],   # geographic degrees; score in [0,1], sorted desc
        "proj": [(y_col,   x_col,   score), ...]    # projected meters   ; score in [0,1], sorted desc
      }

    Notes:
      - Name-driven candidates first, then (optional) numeric fallback.
      - Score is mean of fractions in valid ranges.
      - Duplicates are removed keeping the highest score.
      - Column order is used as a stable tie-breaker.
    """
    cols = list(map(str, df.columns))

    # Broader name patterns for better recall
    pat_lat = re.compile(r"\b(lat|latitude)\b", re.I)
    pat_lon = re.compile(r"\b(lon|lng|long|longitude)\b", re.I)

    # Projected: include northing/easting + x/y variants
    pat_y   = re.compile(r"\b(y|y_coord|ycoord|northing|north|n)\b", re.I)
    pat_x   = re.compile(r"\b(x|x_coord|xcoord|easting|east|e)\b", re.I)

    cand_lat = [c for c in cols if pat_lat.search(c)]
    cand_lon = [c for c in cols if pat_lon.search(c)]
    cand_y   = [c for c in cols if pat_y.search(c)]
    cand_x   = [c for c in cols if pat_x.search(c)]

    # As a very weak fallback, accept bare 'x'/'y' if nothing matched for proj
    if not cand_y:
        cand_y = [c for c in cols if re.fullmatch(r"[yY]", c)]
    if not cand_x:
        cand_x = [c for c in cols if re.fullmatch(r"[xX]", c)]

    def geo_score(la: str, lo: str) -> float:
        """Score for geographic pair based on valid degree ranges."""
        return float((_frac_in_range(df[la], -90, 90) + _frac_in_range(df[lo], -180, 180)) / 2.0)

    def proj_score(y: str, x: str) -> float:
        """Score for projected pair based on coarse meter ranges (UTM/Lambert-like)."""
        return float((_frac_in_range(df[y], 1e5, 1.1e7) + _frac_in_range(df[x], 1e5, 2e7)) / 2.0)

    geo_pairs: Dict[Tuple[str, str], float] = {}
    proj_pairs: Dict[Tuple[str, str], float] = {}

    # 1) Name-driven geographic candidates
    for la in cand_lat:
        for lo in cand_lon:
            if la == lo:
                continue
            sc = geo_score(la, lo)
            if sc >= min_score_geo:
                geo_pairs[(la, lo)] = max(geo_pairs.get((la, lo), 0.0), sc)

    # 2) Name-driven projected candidates
    for y in cand_y:
        for x in cand_x:
            if y == x:
                continue
            sc = proj_score(y, x)
            if sc >= min_score_proj:
                proj_pairs[(y, x)] = max(proj_pairs.get((y, x), 0.0), sc)

    # 3) Numeric fallback (optional): try all numeric-like pairs both as geo & proj
    if enable_numeric_fallback:
        num_like = [
            c for c in cols
            if pd.api.types.is_numeric_dtype(df[c]) or _to_num(df[c]).notna().mean() > 0.7
        ]
        for a, b in combinations(num_like, 2):
            # geo (a,b) and (b,a)
            sc = geo_score(a, b)
            if sc >= min_score_geo:
                geo_pairs[(a, b)] = max(geo_pairs.get((a, b), 0.0), sc)
            sc = geo_score(b, a)
            if sc >= min_score_geo:
                geo_pairs[(b, a)] = max(geo_pairs.get((b, a), 0.0), sc)
            # proj (a,b) and (b,a)
            sc = proj_score(a, b)
            if sc >= min_score_proj:
                proj_pairs[(a, b)] = max(proj_pairs.get((a, b), 0.0), sc)
            sc = proj_score(b, a)
            if sc >= min_score_proj:
                proj_pairs[(b, a)] = max(proj_pairs.get((b, a), 0.0), sc)

    # 4) Sort by score desc, then by column order for reproducibility
    geo_list = sorted(geo_pairs.items(), key=lambda kv: (-kv[1], cols.index(kv[0][0]), cols.index(kv[0][1])))
    proj_list = sorted(proj_pairs.items(), key=lambda kv: (-kv[1], cols.index(kv[0][0]), cols.index(kv[0][1])))

    geo_out = [(a, b, s) for (a, b), s in geo_list]
    proj_out = [(y, x, s) for (y, x), s in proj_list]

    return {"geo": geo_out, "proj": proj_out}

def detect_latlon_pair(df: pd.DataFrame):
    """
    Backward-compatible single-pair detector.
    Prefers a true geographic (lat, lon) pair; if none, falls back to projected (Y, X).
    Returns:
      (lat_like_col, lon_like_col) or None
    """
    all_pairs = detect_coordinate_pairs_all(df, min_score_geo=0.7, min_score_proj=0.7)

    # Prefer the best geographic pair
    if all_pairs["geo"]:
        la, lo, _ = all_pairs["geo"][0]
        return (la, lo)

    # Fallback to best projected pair (return (Y_like, X_like) in the same order)
    if all_pairs["proj"]:
        y, x, _ = all_pairs["proj"][0]
        return (y, x)

    return None

# Spatial detector with reference
def build_ref_sets(ref_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Expect each ref table with columns ['code','nom'].
    Keys of ref_dict are the spatial levels you want reported (e.g. 'region','departement', ...).
    """
    sets = {}
    for level, df in ref_dict.items():
        if not {"code", "nom"}.issubset(df.columns):
            raise ValueError(f"Reference '{level}' must have columns ['code','nom']")
        codes = set(norm_code(x) for x in df["code"].dropna().unique())
        names = set(norm_name(x) for x in df["nom"].dropna().unique())
        sets[level] = {"codes": codes, "names": names}
    return sets

def match_ratio(sample_values, valid_set) -> float:
    if not sample_values:
        return 0.0
    hits = sum(1 for v in sample_values if v in valid_set)
    return hits / len(sample_values)

def match_series_to_ref_levels(
    s: pd.Series,
    ref_sets: Dict[str, Dict[str, Any]],
    sample_size: int = 300,
) -> Dict[str, Any]:
    """Return best {'level','by','ratio'} or {}. Robust to int/str code forms."""

    sample = s.dropna().drop_duplicates()
    if sample.empty:
        return {}
    # Work with strings consistently
    sample = sample.astype(str).str.strip().head(sample_size)

    # Precompute normalized name samples once
    sample_names = [norm_name(x) for x in sample]

    # For each level, learn plausible code lengths from its ref codes,
    # then zero-pad numeric samples accordingly before matching.
    level_len_candidates: Dict[str, list[int]] = {}
    for level, sets in ref_sets.items():
        codes_set = sets.get("codes", set()) or set()
        if codes_set:
            lens = [len(str(c)) for c in codes_set]
            if lens:
                cnt = Counter(lens)
                # Try up to 3 most common lengths (e.g., dep: 2 & 3; epci/iris: 9; commune: 5)
                level_len_candidates[level] = [L for L, _ in cnt.most_common(3)]
            else:
                level_len_candidates[level] = []
        else:
            level_len_candidates[level] = []

    best = {"level": None, "by": None, "ratio": 0.0}

    for level, sets in ref_sets.items():
        codes_set = sets.get("codes", set()) or set()
        names_set = sets.get("names", set()) or set()
        lens_to_try = level_len_candidates.get(level, [])

        # Build level-specific normalized code samples
        norm_codes_for_level: list[str] = []
        for x in sample:
            base = norm_code(x)  # your global canonicaliser (upper, strip, etc.)
            # If contains letters (e.g., '2A'), keep as-is
            if re.search(r"[A-Za-z]", base):
                norm_codes_for_level.append(base)
                continue
            # Purely digits: try zero-padding to the plausible lengths for this level
            cands = [base] + [base.zfill(L) for L in lens_to_try if L >= len(base)]
            # Choose the first candidate that exists in ref codes; otherwise keep the longest padded (or base)
            chosen = next((c for c in cands if c in codes_set), cands[-1])
            norm_codes_for_level.append(chosen)

        r_code = match_ratio(norm_codes_for_level, codes_set) if codes_set else 0.0
        r_name = match_ratio(sample_names, names_set) if names_set else 0.0

        if r_code >= r_name:
            r, by = r_code, "code"
        else:
            r, by = r_name, "nom"

        if r > best["ratio"]:
            best = {"level": SPATIAL_NAME_MAP.get(level), "by": by, "ratio": r}

    return best if best["level"] is not None else {}

# --------------- Spatial name-based extraction (filename/colname/sheetname) -

# Code patterns aligned with FR practice (incl. 2A/2B and 971–976).
# Regions (code région: 2 digits, some historic 2 digits, new INSEE list since 2016)
_REG_RE = re.compile(r"\b(0?[1-9]|1[0-3]|11|24|27|28|32|44|52|53|75|76|84|93)\b", re.I)
# Académies (codes: 2 digits, official INSEE/MENJ list; e.g. 01=Paris, 02=Créteil, 03=Versailles, etc.)
_ACADEMIE_RE = re.compile(r"\b(0?[1-9]|1\d|2[0-2])\b", re.I)
# Départements (2 digits or 2A/2B or 3 digits for overseas)
_DEP_RE = re.compile(r"\b(0[1-9]|[1-8][0-9]|9[0-5]|2a|2b|97[1-6])\b", re.I)
# Arrondissement départemental (3-digit code within département)
_ARR_DEP_RE = re.compile(r"\b\d{3}\b", re.I)
# EPCI (SIREN: 9 digits)
_EPCI_RE = re.compile(r"\b\d{9}\b", re.I)
# Cantons (INSEE: département code (2/3 chars) + 2 digits, e.g. '97412', '2A04')
# → often 4–5 chars, alphanumeric because of 2A/2B
_CANTON_RE = re.compile(r"\b(?:\d{2}|2a|2b|97[1-6])[0-9]{2}\b", re.I)
# Communes (INSEE code commune: 5 digits, or 2A/2B + 3 digits)
_COM_RE = re.compile(r"\b(\d{5}|2a\d{3}|2b\d{3})\b", re.I)
# Arrondissements municipaux (Paris, Lyon, Marseille): 5-digit commune + 2 digits
_COM_ARR_RE = re.compile(r"\b(\d{5}|2a\d{3}|2b\d{3})\d{2}\b", re.I)
# IRIS (INSEE: 9 digits)
_IRIS_RE = re.compile(r"\b\d{9}\b", re.I)

def _tokenise_for_names(text: str) -> List[str]:
    """
    Produce normalized tokens from any text. Keep the full normalized string
    plus alphanumeric splits to improve recall for multiword names.
    """
    t = normalise_colname(text)
    parts = re.split(r"[^a-z0-9]+", t)
    parts = [p for p in parts if p]
    return [t] + parts

def extract_spatial_hints_from_text_with_refs(text: str, ref_sets: Dict[str, Dict[str, Any]]) -> List[dict]:
    """
    Extract spatial hints (code or nom) from arbitrary text and validate against ref_sets.
    Returns a list of dicts with level/granularity/matched_by/value/confidence/evidence.
    """
    hits: List[dict] = []
    t_norm = normalise_colname(text)

    # 1) Code matches (low-noise) + validation using ref_sets codes
    code_trials = [
        ("region", _REG_RE.findall(t_norm)),
        ("academie", _ACADEMIE_RE.findall(t_norm)),
        ("departement", _DEP_RE.findall(t_norm)),
        ("arr_dep", _ARR_DEP_RE.findall(t_norm)),
        ("epci", _EPCI_RE.findall(t_norm)),
        ("canton", _CANTON_RE.findall(t_norm)),
        ("commune", _COM_RE.findall(t_norm)),
        ("com_arr", _COM_ARR_RE.findall(t_norm)),
        ("iris", _IRIS_RE.findall(t_norm)),
    ]
    for level, codes in code_trials:
        for c in codes:
            c_norm = norm_code(c)
            if level in ref_sets and c_norm in ref_sets[level].get("codes", set()):
                hits.append({
                    "level": level, "granularity": level, "matched_by": "code",
                    "value": c_norm, "confidence": 0.97,
                    "evidence": f"Name fallback: text contains code {c_norm}."
                })

    # 2) Name (nom) matches using ref_sets names (normalized contains match with boundary proxy)
    tokens = _tokenise_for_names(text)
    hay = " " + " ".join(tokens) + " "
    for level, sets in ref_sets.items():
        for nm in sets.get("names", set()):
            # Avoid false positives for too-short names or numeric-only tokens
            if len(nm) < 4 or nm.isdigit():
                continue
            if f" {nm} " in hay or f"_{nm}_" in hay:
                hits.append({
                    "level": level, "granularity": level, "matched_by": "nom",
                    "value": nm, "confidence": 0.95,
                    "evidence": f"Name fallback: text mentions '{nm}'."
                })

    # Deduplicate by (level, matched_by, value)
    uniq = {(h["level"], h["matched_by"], h["value"]): h for h in hits}
    return list(uniq.values())

# Generic “geo” gate (lets us try all levels with a higher bar)
_SPATIAL_GENERIC_RE = re.compile(
    r"\b(zone|territoire|secteur|perimetre|unite(?:_?geo)?|codgeo|code_?geo|geo|geog|geographique|spatial)\b",
    re.I
)

def spatial_gate_from_colname(name: str) -> dict:
    """
    Return {'has_gate': bool, 'levels': [canonical levels], 'generic': bool, 'source': 'insee|code|alias|generic'}.
    """
    n = normalise_colname(name)
    hits_short = []
    source = None

    for key, pat in _SPATIAL_COLNAME_HINTS:
        loose_pat = re.compile(pat.pattern.replace(r"\b", ""), pat.flags)
        if loose_pat.search(n):
            hits_short.append(key)

            # source: based on the *column name* (robust to regex rewrites)
            if "insee" in n:
                source = source or "insee"
            elif "code" in n:
                source = source or "code"
            else:
                source = source or "alias"

    generic = bool(_SPATIAL_GENERIC_RE.search(n))

    seen, out = set(), []
    for lv in hits_short:
        if lv not in seen:
            out.append(lv); seen.add(lv)

    return {"has_gate": bool(out) or generic, "levels": out, "generic": generic, "source": source or ("generic" if generic else None)}

# --- Column-name hints for geo formats (geometry / WKT-GeoJSON / geohash) ---
_GEOFORMAT_HINTS = {
    # Typical names seen across GIS exports
    "geometry": re.compile(r"\b(the_)?geom(?:etry)?\b|\bshape\b|\bgeom_wkt\b|\bgeomjson\b", re.I),
    "wkt":      re.compile(r"\bwkt\b|well[_\- ]?known[_\- ]?text", re.I),
    "geojson":  re.compile(r"\bgeojson\b|\bgeom?_?json\b|\bjson_?geom\b", re.I),
    "geohash":  re.compile(r"\bgeo_?hash\b|\bgeohash\b|\bgh\b", re.I),
}

def geoformat_hints_from_colname(name: str) -> dict:
    """Return flags telling whether the column name suggests a geo format."""
    n = normalise_colname(name)
    return {k: bool(p.search(n)) for k, p in _GEOFORMAT_HINTS.items()}