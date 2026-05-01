from __future__ import annotations

import json
import inspect
from typing import Any, Dict, List, Optional, Set, Union
from collections.abc import Generator, Iterable


class DSSpatialScope:
    def __init__(self, spatialScopeLevel, spatialScope):
        self.spatialScopeLevel = spatialScopeLevel
        self.spatialScope = spatialScope

    def to_dict(self):
        return {
            "spatialScopeLevel": self.spatialScopeLevel,
            "spatialScope": self.spatialScope
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["spatialScopeLevel"], data["spatialScope"])


class TimePeriod:
    def __init__(self, startTime: Any, endTime: Any):
        # Initialise the time period with a start and end time
        self.startTime = startTime
        self.endTime = endTime

    def to_dict(self) -> Dict[str, Any]:
        # Convert the object into a dictionary for serialisation
        return {
            "startTime": self.startTime,
            "endTime": self.endTime
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimePeriod":
        # Create a TimePeriod instance from a dictionary
        return cls(
            startTime=data["startTime"],
            endTime=data["endTime"]
        )


class DSTemporalScope:
    def __init__(self, temporalScopeLevel: str, timePeriods: List[TimePeriod] = None):
        # Initialise with scope level and a list of TimePeriod objects
        self.temporalScopeLevel = temporalScopeLevel
        self.timePeriods = timePeriods if timePeriods is not None else []

    def to_dict(self) -> Dict[str, Any]:
        def _period_to_dict(p):
            # Case 1: p is a TimePeriod object → call its own to_dict()
            if hasattr(p, "to_dict"):
                return p.to_dict()
            # Case 2: p is a tuple or list of length 2 → treat as (start, end)
            if isinstance(p, (tuple, list)) and len(p) == 2:
                return {"startTime": p[0], "endTime": p[1]}
            # Case 3: fallback → return as-is (not recommended, but avoids crash)
            return p

        return {
            "temporalScopeLevel": self.temporalScopeLevel,
            "timePeriods": [_period_to_dict(p) for p in self.timePeriods],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSTemporalScope":
        # Create a DSTemporalScope instance from a dictionary,
        # converting each time period into a TimePeriod object
        time_periods = [TimePeriod.from_dict(p) for p in data.get("timePeriods", [])]
        return cls(
            temporalScopeLevel=data["temporalScopeLevel"],
            timePeriods=time_periods
        )

    def add_period(self, start_time: Any, end_time: Any) -> TimePeriod:
        # Add a new time period to the scope
        period = TimePeriod(start_time, end_time)
        self.timePeriods.append(period)
        return period


class Theme:
    def __init__(self, themeName, themeDescription):
        self.themeName = themeName
        self.themeDescription = themeDescription

    def to_dict(self):
        return {
            "themeName": self.themeName,
            "themeDescription": self.themeDescription
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["themeName"], data["themeDescription"])


# Attribute
class Attribute:
    def __init__(self, dataName, dataDescription, dataType):
        self.dataName = dataName
        self.dataDescription = dataDescription
        self.dataType = dataType

    def to_dict(self):
        return {
            "dataName": self.dataName,
            "dataDescription": self.dataDescription,
            "dataType": self.dataType
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["dataName"], data["dataDescription"], data["dataType"])


# subclass：Parameter
class Parameter(Attribute):
    def __init__(self, dataName, dataDescription, dataType):
        super().__init__(dataName, dataDescription, dataType)

    def to_dict(self):
        data = super().to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"]
        )


# subclass：SpatialParameter
class SpatialParameter(Parameter):
    def __init__(self, dataName, dataDescription, dataType, spatialLevel):
        super().__init__(dataName, dataDescription, dataType)
        self.spatialLevel = spatialLevel

    def to_dict(self):
        data = super().to_dict()
        data["spatialLevel"] = self.spatialLevel
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["spatialLevel"]
        )


# subclass：TemporalParameter
class TemporalParameter(Parameter):
    def __init__(self, dataName, dataDescription, dataType, temporalLevel):
        super().__init__(dataName, dataDescription, dataType)
        self.temporalLevel = temporalLevel

    def to_dict(self):
        data = super().to_dict()
        data["temporalLevel"] = self.temporalLevel
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data["temporalLevel"]
        )


# subclass：ComplementaryInformation
class ComplementaryInformation(Attribute):
    def __init__(self, dataName, dataDescription, dataType, granularity, theme):
        super().__init__(dataName, dataDescription, dataType)
        self.granularity = granularity   # string, e.g. "NUTS3", "day", "year"
        self.theme = theme               # Theme object or None

    def to_dict(self):
        data = super().to_dict()
        data["granularity"] = self.granularity
        if isinstance(self.theme, Theme):
            data["theme"] = self.theme.to_dict()
        else:
            data["theme"] = None
        return data

    @classmethod
    def from_dict(cls, data):
        theme = Theme.from_dict(data["theme"]) if isinstance(data.get("theme"), dict) else data.get("theme")
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data.get("granularity"),   # now just a string instead of Parameter subclass
            theme
        )

# subclass：ExistingIndicator
class ExistingIndicator(Attribute):
    def __init__(self, dataName, dataDescription, dataType, indicatorType, theme):
        super().__init__(dataName, dataDescription, dataType)
        self.indicatorType = indicatorType
        self.theme = theme

    def to_dict(self):
        data = super().to_dict()
        data["indicatorType"] = self.indicatorType
        # Write theme under the "theme" key (not "related_parameter")
        if isinstance(self.theme, Theme):
            data["theme"] = self.theme.to_dict()
        else:
            data["theme"] = None
        return data


    @classmethod
    def from_dict(cls, data):
        # If theme is a dict, hydrate Theme; otherwise pass through/None
        theme = Theme.from_dict(data["theme"]) if isinstance(data.get("theme"), dict) else data.get("theme")
        return cls(
            data["dataName"],
            data["dataDescription"],
            data["dataType"],
            data.get("indicatorType"),
            theme
        )



class Dataset:
    def __init__(
        self,
        # --- Identification ---
        title: str,
        description: str,
        dataFormat: str,   # e.g. "structured" / "semi-structured"
        fileType: str,     # extension, e.g. ".csv"
        rawzonePath: str,

        # --- Update & source (required) ---
        updateFrequency: str,
        sourceName: str,
        sourceType: str,
        sourceAddress: str,

        # --- Five core elements (required) ---
        spatialGranularity: str,
        spatialScope: Optional[List["DSSpatialScope"]],
        temporalGranularity: str,
        temporalScope: Optional[List["DSTemporalScope"]],
        themes: Optional[Set["Theme"]],  # Theme must be hashable if stored in a set

        # --- Attributes (optional list) ---
        attributes: Optional[List["Attribute"]] = None,

        # --- Generic size metrics (always meaningful) ---
        fileSizeBytes: Optional[int] = None,
        fileSizeHuman: Optional[str] = None,

        # --- Tabular metrics (only for structured files) ---
        nRows: Optional[int] = None,
        nCols: Optional[int] = None,

        # --- Semi-structured optional metrics ---
        nRecords: Optional[int] = None,
        nFeatures: Optional[int] = None,
        uncompressedSizeBytes: Optional[int] = None
    ):
        # Identification
        self.title = title
        self.description = description
        self.dataFormat = dataFormat
        self.fileType = fileType
        self.rawzonePath = rawzonePath

        # Update & source
        self.updateFrequency = updateFrequency
        self.sourceName = sourceName
        self.sourceType = sourceType
        self.sourceAddress = sourceAddress

        # Five core elements
        self.spatialGranularity = spatialGranularity
        self.spatialScope = spatialScope
        self.temporalGranularity = temporalGranularity
        self.temporalScope = temporalScope

        # Normalize theme into a Set[Theme]
        self.themes = self._coerce_theme_set(themes)

        # Attributes
        self.attributes: List["Attribute"] = attributes or []

        # Generic size
        self.fileSizeBytes = fileSizeBytes
        self.fileSizeHuman = fileSizeHuman

        # Tabular
        self.nRows = nRows
        self.nCols = nCols

        # Semi-structured
        self.nRecords = nRecords
        self.nFeatures = nFeatures
        self.uncompressedSizeBytes = uncompressedSizeBytes

        # Internal counters
        self._spatial_counter = 1
        self._temporal_counter = 1
        self._complementary_counter = 1
        self._indicator_counter = 1
        self._other_counter = 1

    # ---------------------------
    # Helpers for Theme handling
    # ---------------------------
    @staticmethod
    def _iterify(value):
        """
        Turn input into a one-shot iterable.
        Accepts: None, generator, iterable, Theme, dict, str.
        """
        if value is None:
            return []
        if inspect.isgenerator(value) or isinstance(value, Generator):
            return list(value)
        if isinstance(value, (Theme, dict, str)):
            return [value]
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
            return list(value)
        return [value]

    @staticmethod
    def _coerce_theme_item(item) -> "Theme":
        """
        Convert a single item into a Theme instance.
        Accepts Theme / dict / str.
        """
        if isinstance(item, Theme):
            return item
        if isinstance(item, dict):
            return Theme.from_dict(item)
        if isinstance(item, str):
            return Theme(item, item)
        raise TypeError(f"Unsupported theme item type: {type(item)}")

    @classmethod
    def _coerce_theme_set(cls, value) -> Optional[Set["Theme"]]:
        """
        Normalize various inputs into Set[Theme] or None.
        """
        items = cls._iterify(value)
        if not items:
            return None
        themes: Set["Theme"] = set()
        for it in items:
            themes.add(cls._coerce_theme_item(it))
        return themes if themes else None

    # ---------------------------
    # Serialization
    # ---------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Dataset to a plain dict.
        """
        return {
            # identification
            "title": self.title,
            "description": self.description,
            "dataFormat": self.dataFormat,
            "fileType": self.fileType,
            "rawzonePath": self.rawzonePath,

            # size metrics
            "fileSizeBytes": self.fileSizeBytes,
            "fileSizeHuman": self.fileSizeHuman,

            # tabular
            "nRows": self.nRows,
            "nCols": self.nCols,

            # semi-structured
            "nRecords": self.nRecords,
            "nFeatures": self.nFeatures,
            "uncompressedSizeBytes": self.uncompressedSizeBytes,

            # update & source
            "updateFrequency": self.updateFrequency,
            "sourceName": self.sourceName,
            "sourceType": self.sourceType,
            "sourceAddress": self.sourceAddress,

            # five elements
            "spatialGranularity": self.spatialGranularity,
            "spatialScope": [s.to_dict() for s in self.spatialScope] if self.spatialScope else None,
            "temporalGranularity": self.temporalGranularity,
            "temporalScope": [t.to_dict() for t in self.temporalScope] if self.temporalScope else None,

            # themes: Set[Theme] -> List[dict]
            "themes": (
                [th.to_dict() for th in sorted(self.themes, key=lambda x: getattr(x, "themeName", ""))]
                if self.themes else None
            ),

            # attributes
            "attributes": [content.to_dict() for content in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """
        Reconstruct Dataset from a plain dict.
        """
        spatial_scope_raw = data.get("spatialScope")
        if isinstance(spatial_scope_raw, list):
            spatial_scope = [DSSpatialScope.from_dict(x) for x in spatial_scope_raw if isinstance(x, dict)]
        elif isinstance(spatial_scope_raw, dict):
            spatial_scope = [DSSpatialScope.from_dict(spatial_scope_raw)]
        else:
            spatial_scope = None

        temporal_scope_raw = data.get("temporalScope")
        if isinstance(temporal_scope_raw, list):
            temporal_scope = [DSTemporalScope.from_dict(x) for x in temporal_scope_raw if isinstance(x, dict)]
        elif isinstance(temporal_scope_raw, dict):
            temporal_scope = [DSTemporalScope.from_dict(temporal_scope_raw)]
        else:
            temporal_scope = None

        theme_set = cls._coerce_theme_set(data.get("themes"))

        attrs_raw = data.get("attributes") or data.get("Attribute") or []
        attributes: List["Attribute"] = [cls._map_to_subclass(content) for content in attrs_raw]

        return cls(
            # identification
            title=data["title"],
            description=data["description"],
            dataFormat=data["dataFormat"],
            fileType=data["fileType"],
            rawzonePath=data.get("rawzonePath", ""),

            # update & source
            updateFrequency=data["updateFrequency"],
            sourceName=data["sourceName"],
            sourceType=data["sourceType"],
            sourceAddress=data["sourceAddress"],

            # five elements
            spatialGranularity=data["spatialGranularity"],
            spatialScope=spatial_scope,
            temporalGranularity=data["temporalGranularity"],
            temporalScope=temporal_scope,
            themes=theme_set,

            # attributes
            attributes=attributes,

            # size metrics
            fileSizeBytes=data.get("fileSizeBytes"),
            fileSizeHuman=data.get("fileSizeHuman"),

            # tabular
            nRows=data.get("nRows"),
            nCols=data.get("nCols"),

            # semi-structured
            nRecords=data.get("nRecords"),
            nFeatures=data.get("nFeatures"),
            uncompressedSizeBytes=data.get("uncompressedSizeBytes"),
        )

    # ---------------------------
    # Add helpers
    # ---------------------------
    def add_spatial_parameter(
        self,
        col: Union[str, List[str]],
        description: str,
        dtype: Union[str, List[str]],
        level: str
    ) -> "SpatialParameter":
        param = SpatialParameter(col, description, dtype, level)
        self.attributes.append(param)
        self._spatial_counter += 1
        return param

    def add_temporal_parameter(
        self,
        col: Union[str, List[str]],
        description: str,
        dtype: Union[str, List[str]],
        level: str
    ) -> "TemporalParameter":
        param = TemporalParameter(col, description, dtype, level)
        self.attributes.append(param)
        self._temporal_counter += 1
        return param

    def add_existing_indicator(
        self,
        name: Union[str, List[str]],
        description: str,
        dtype: Union[str, List[str]],
        indicator_type: Optional[str],
        theme: Optional["Theme"]
    ) -> "ExistingIndicator":
        ind = ExistingIndicator(name, description, dtype, indicator_type, theme)
        self.attributes.append(ind)
        self._indicator_counter += 1
        return ind

    def add_complementary_information(
        self,
        name: Union[str, List[str]],
        description: str,
        dtype: Union[str, List[str]],
        granularity: Optional[str] = None,
        theme: Optional["Theme"] = None
    ) -> "ComplementaryInformation":
        ci = ComplementaryInformation(name, description, dtype, granularity, theme)
        self.attributes.append(ci)
        self._complementary_counter += 1
        return ci

    def add_other_content(
        self,
        name: Union[str, List[str]],
        description: str,
        dtype: Union[str, List[str]]
    ) -> "Attribute":
        content = Attribute(name, description, dtype)
        self.attributes.append(content)
        self._other_counter += 1
        return content

    @staticmethod
    def _map_to_subclass(data: Dict[str, Any]) -> "Attribute":
        """
        Factory: map a plain dict to the most appropriate subclass by inspecting keys.
        """
        if not isinstance(data, dict):
            raise TypeError("Expected a dict for _map_to_subclass")

        if "spatialLevel" in data:
            return SpatialParameter.from_dict(data)
        if "temporalLevel" in data:
            return TemporalParameter.from_dict(data)
        if "granularity" in data or "related_parameter" in data:
            return ComplementaryInformation.from_dict(data)
        if "theme" in data:
            return ExistingIndicator.from_dict(data)

        core = {"dataName", "dataDescription", "dataType"}
        if core.issubset(data.keys()):
            return Parameter.from_dict(data)

        return Attribute.from_dict(data)

    # ---------------------------
    # Persistence helpers
    # ---------------------------
    def save_to_json(self, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_json(cls, file_path: str) -> "Dataset":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
