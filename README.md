# Metadata Capture

Metadata Capture is a Python-based pipeline for extracting semantic metadata from heterogeneous spatio-temporal datasets stored in a MinIO/S3 data lake.

It reads raw datasets, detects spatial and temporal parameters, identifies existing indicators and complementary information, derives dataset-level descriptors and writes structured metadata into the governance zone of the data lake.

This prototype is developed to support metadata-driven management of multi-thematic spatio-temporal datasets, with a particular focus on well-being and age-friendliness analysis.

## Features

- Read raw datasets directly from MinIO/S3.
- Support common data formats such as CSV, TSV, Excel, JSON, GeoJSON, Parquet, ZIP and shapefile-related inputs.
- Detect spatial parameters, including administrative codes, names, addresses, coordinates and geometry columns.
- Detect temporal parameters, including year, quarter, month, week and date.
- Classify dataset attributes into spatial parameters, temporal parameters, existing indicators and complementary information.
- Assign indicators and complementary information to a predefined thematic hierarchy.
- Derive dataset-level descriptors, including thematic coverage, spatial scope, temporal scope, spatial granularity and temporal granularity.
- Generate one metadata JSON file per dataset.
- Generate one performance log JSON file per dataset.
- Maintain a global metadata catalog for processed datasets.
- Skip already processed datasets unless reprocessing is explicitly requested.

## Repository Structure

```text
metadata-capture/
│
├── metadata_output.py          # Batch processing and catalog generation
├── metadata_selector.py        # Single-dataset metadata construction
├── uml_class.py                # Metadata object model
├── semantic_helper.py          # Semantic column classification
├── attribute_classifier.py     # Attribute classification
├── spatial_detector.py         # Spatial attribute detection
├── temporal_detector.py        # Temporal attribute detection
├── granularity_detector.py     # Granularity derivation
├── scope_detector.py           # Scope derivation
├── indicator_detector.py       # Indicator detection
├── theme_detector.py           # Theme aggregation
├── general_function.py         # Utility functions and file readers
├── perform_monitor.py          # Performance log aggregation
└── reference.py                # Configuration, paths, hierarchies and references
