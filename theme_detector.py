from reference import *
from uml_class import Theme
from typing import List, Dict, Any, Optional, Set

def find_longest_common_prefix(paths):
    """
    Find the longest common prefix among a list of paths.

    Parameters:
        paths (list): List of theme paths as strings.

    Returns:
        list: Longest common prefix as a list of levels.
    """
    if not paths:
        return []

    # Split each path into a list of levels
    split_paths = [path.split(">") for path in paths]

    # Find the shortest path for boundary comparison
    min_length = min(len(path) for path in split_paths)

    common_prefix = []
    for i in range(min_length):
        # Check if all paths share the same level at index i
        level_set = {path[i] for path in split_paths}
        if len(level_set) == 1:
            common_prefix.append(level_set.pop())
        else:
            break

    return common_prefix


def validate_common_prefix(theme_structure, common_prefix):
    """
    Validate the common prefix against the theme folder structure.

    Parameters:
        theme_structure (dict): Nested dictionary representing the theme hierarchy.
        common_prefix (list): List of levels representing the common prefix.

    Returns:
        bool: True if the prefix is valid within the theme structure, False otherwise.
    """
    current_structure = theme_structure
    for level in common_prefix:
        if level in current_structure:
            current_structure = current_structure[level]
        else:
            return False  # The prefix doesn't exist in the theme structure
    return True


def _split_to_levels(path: str) -> List[str]:
    """Split a hierarchical path into levels and trim whitespace.
    Supports separators: '>', '/', '\\', '→', '»'.
    """
    parts = re.split(r'>|/|\\|→|»', str(path))
    return [p.strip() for p in parts if p and p.strip()]

def find_min_common_theme(atts_theme: List[Dict[str, Any]]) -> Optional[str]:
    """
    Find the minimum common theme across attributes (indicators + others).

    Input:
      - atts_theme: list of dicts, each containing "theme" which can be:
            - Theme instance (expects .themeName)
            - dict with keys like themeName/name/title
            - plain string path
      - THEME_FOLDER_STRUCTURE: nested dict representing the theme hierarchy tree

    Output:
      - A normalised theme path joined by ' > ' if a valid common prefix exists
      - None if no themes or no valid common prefix
    """
    # 1) Collect raw theme names
    raw_themes: List[str] = []
    for att in atts_theme:
        th = att.get("theme")
        if not th:
            continue
        if isinstance(th, Theme):
            raw_themes.append(th.themeName)
        elif isinstance(th, dict):
            name = th.get("themeName") or th.get("name") or th.get("title") or th.get("theme_name")
            if name:
                raw_themes.append(str(name))
        elif isinstance(th, str):
            raw_themes.append(th)

    if not raw_themes:
        print("No themes found in atts_theme.")
        return None

    # # 2) Normalise to lists of levels (so prefix search is separator-agnostic)
    # normalised_paths = ["/".join(_split_to_levels(t)) for t in raw_themes]

    # 3) Longest common prefix on normalised string paths
    common_prefix_levels = find_longest_common_prefix(raw_themes)  # returns a list of levels

    # 4) Validate against the folder structure (expects list of levels)
    if validate_common_prefix(THEME_FOLDER_STRUCTURE, common_prefix_levels):
        # Join with the required separator for final output
        return " > ".join(common_prefix_levels)
    else:
        print("Common prefix does not exist in theme folder structure.")
        return None


def collect_all_themes_set(atts_theme: List[Dict[str, Any]], separator: str = " > ") -> Set[str]:
    """
    收集 atts_theme 中所有主题，返回去重后的集合（已归一化）。
    归一化规则：把任意分隔符统一拆分后，用 separator 重新连接。
    """
    themes: Set[str] = set()

    for att in atts_theme:
        th = att.get("theme")
        if not th:
            continue

        # 兼容三种输入：Theme 实例、dict、纯字符串
        if isinstance(th, Theme):
            raw = th.themeName
        elif isinstance(th, dict):
            raw = th.get("themeName") or th.get("name") or th.get("title") or th.get("theme_name")
        elif isinstance(th, str):
            raw = th
        else:
            raw = None

        if not raw:
            continue

        # 归一化到统一层级路径
        levels = _split_to_levels(raw)
        if levels:
            themes.add(separator.join(levels))

    return themes

