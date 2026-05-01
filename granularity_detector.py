from typing import List, Set, Union, Optional
from general_function import get_most_specific_in_path, level_name


def get_granularity(
    granularities: List[str],
    hierarchies: List[List[Union[str, tuple]]]
) -> Optional[str]:
    """
    Determine the final dataset granularity from detected granularities
    and a list of hierarchy paths.

    Logic
    -----
    1. For each hierarchy path, pick the most specific matched level.
    2. If no path matches, return None.
    3. If all picked levels belong to a single hierarchy path, collapse them
       to the most specific level within that path.
    4. Otherwise, return the first picked level (deduplicated, order-preserving).

    Parameters
    ----------
    granularities : List[str]
        Detected granularities from attributes.

    hierarchies : List[List[Union[str, tuple]]]
        A list of hierarchy paths. Each path is ordered from more general
        to more specific.

    Returns
    -------
    Optional[str]
        Final granularity level, or None if no valid match is found.
    """

    # Step 1: pick the most specific match for each hierarchy path
    picks: List[str] = []
    for path in hierarchies:
        top = get_most_specific_in_path(granularities, path)
        if top:
            picks.append(top)

    if not picks:
        return None

    # Step 2: check whether all picked levels can be explained by one single path
    pickset: Set[str] = set(picks)
    for path in hierarchies:
        path_names = {level_name(level) for level in path}
        if pickset.issubset(path_names):
            collapsed = get_most_specific_in_path(granularities, path)
            return collapsed if collapsed else None

    # Step 3: fallback = return the first unique pick (order-preserving)
    seen: Set[str] = set()
    out: List[str] = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return out[0] if out else None