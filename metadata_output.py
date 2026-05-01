import os
import json
import inspect
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Generator, Iterable
from typing import Any, Dict, List, Optional

from uml_class import Dataset
from metadata_selector import construct_dataset
from reference import (
    DATA_DIR,
    METADATA_DIR,
    CATALOG_PATH,
    PERF_DIR,
    DEFAULT_EXTS,
    LAKE_BUCKET,
    s3,
    minio_key,
    object_exists,
)
from general_function import (
    copy_input_object_to_rawzone,
    uncompressed_zip_size,
)


# ----------------------------- Small utilities -----------------------------

def make_json_safe(obj):
    """
    Recursively convert sets and other non-serialisable types to JSON-safe ones.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, set):
        return sorted(make_json_safe(v) for v in obj)
    else:
        return obj


def read_json_object(path_obj) -> Any:
    """
    Read one JSON object from MinIO and return the parsed Python object.
    """
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_json_object(path_obj, payload: Any, indent: int = 2) -> None:
    """
    Write one JSON object to MinIO.
    """
    body = json.dumps(
        make_json_safe(payload),
        ensure_ascii=False,
        indent=indent
    ).encode("utf-8")

    s3.put_object(
        Bucket=LAKE_BUCKET,
        Key=minio_key(path_obj),
        Body=body,
        ContentType="application/json"
    )


def get_object_head(path_obj) -> Optional[dict]:
    """
    Return MinIO object metadata if the object exists, else None.
    """
    try:
        return s3.head_object(Bucket=LAKE_BUCKET, Key=minio_key(path_obj))
    except Exception:
        return None


def file_checksum(path: str, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """
    Compute a reproducible checksum for a MinIO object.

    The input `path` is a MinIO object key such as:
    'input_data/fr-esr-parcoursup_2021.csv'
    """
    h = hashlib.new(algo)
    obj = s3.get_object(Bucket=LAKE_BUCKET, Key=minio_key(path))

    stream = obj["Body"]
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        h.update(chunk)

    return h.hexdigest()


def dataset_id_from_path(path: str) -> str:
    """
    Derive a stable dataset ID from the MinIO object key.

    Since input files are stored in MinIO, we should not use os.path.abspath().
    """
    return hashlib.sha1(minio_key(path).encode("utf-8")).hexdigest()[:16]


def iter_data_files(root, allow_exts=None):
    """
    Yield MinIO object keys under the given prefix filtered by extensions.

    Example input root:
        DATA_DIR = Path("input_data")

    Example yielded key:
        input_data/fr-esr-parcoursup_2021.csv
    """
    allow = {e.lower() for e in (allow_exts or DEFAULT_EXTS)}
    prefix = minio_key(root).rstrip("/") + "/"

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=LAKE_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # skip directory placeholders
            if key.endswith("/"):
                continue

            filename = key.split("/")[-1]
            if filename.startswith(".") or filename.endswith(".ini"):
                continue

            ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
            if ext in allow:
                yield key


def _file_mtime_int(p: str) -> int:
    """
    Return object LastModified as integer epoch seconds.
    """
    head = get_object_head(p)
    if not head:
        return -1

    last_modified = head.get("LastModified")
    if last_modified is None:
        return -1

    try:
        return int(last_modified.timestamp())
    except Exception:
        return -1


def _outputs_exist_for(p: str) -> bool:
    """
    Fallback existence check:
    both metadata and perf outputs must exist in MinIO for a file
    to count as processed.
    """
    filename = p.split("/")[-1]
    meta_path = METADATA_DIR / f"{filename}.metadata.json"
    perf_path = PERF_DIR / f"{filename}.perf.json"
    return object_exists(meta_path) and object_exists(perf_path)


def _rawzone_object_exists(entry: Dict[str, Any]) -> bool:
    """
    Check whether the target raw-zone object exists according to the catalog entry.
    """
    rawzone_path = entry.get("rawzonePath")
    if not rawzone_path:
        return False

    try:
        return object_exists(rawzone_path)
    except Exception:
        return False


def was_processed_successfully(p: str, catalog: Dict[str, Any]) -> bool:
    """
    Decide whether to skip processing for file `p`.

    A file is considered successfully processed only if:
    - matching catalog entry exists
    - source mtime matches
    - status == "ok"
    - metadata object exists
    - perf object exists
    - raw-zone object exists
    """
    mtime = _file_mtime_int(p)
    pid = dataset_id_from_path(p)

    try:
        datasets = catalog.get("datasets", []) if isinstance(catalog, dict) else []
        for e in datasets:
            if e.get("id") != pid:
                continue

            if int(e.get("sourceMtime", -2)) != mtime:
                return False

            if str(e.get("status", "")).lower() != "ok":
                return False

            meta_path = e.get("metadataPath")
            perf_path = e.get("perfPath")

            if not meta_path or not object_exists(meta_path):
                return False

            if not perf_path or not object_exists(perf_path):
                return False

            if not e.get("rawzoneCopied", False):
                return False

            if not _rawzone_object_exists(e):
                return False

            return True
    except Exception:
        pass

    return _outputs_exist_for(p)


# ------------------------- Catalog / entry construction -------------------------

def _extract_theme_names(meta_theme) -> Optional[List[str]]:
    """
    Normalize various theme shapes into a list of theme names.
    """
    if meta_theme is None:
        return None

    if inspect.isgenerator(meta_theme) or isinstance(meta_theme, Generator):
        meta_theme = list(meta_theme)

    names: List[str] = []
    seen = set()

    def _add_name(val):
        if not val:
            return
        name = str(val).strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)

    def _extract_from_dict(d: Dict[str, Any]):
        return d.get("themeName") or d.get("name") or d.get("title") or d.get("theme_name")

    if isinstance(meta_theme, dict):
        _add_name(_extract_from_dict(meta_theme))
        return names or None

    if isinstance(meta_theme, str):
        _add_name(meta_theme)
        return names or None

    if isinstance(meta_theme, Iterable) and not isinstance(meta_theme, (bytes, bytearray, str)):
        for it in meta_theme:
            if isinstance(it, dict):
                _add_name(_extract_from_dict(it))
            elif isinstance(it, str):
                _add_name(it)
            else:
                _add_name(getattr(it, "themeName", None))
        return names or None

    _add_name(getattr(meta_theme, "themeName", None))
    return names or None


def build_metadata_entry(data_path: str, ds: Dataset) -> Dict[str, Any]:
    """
    Build one catalog entry from the Dataset object.

    Here `data_path` is the MinIO input object key.
    """
    meta_dict = ds.to_dict()

    filename = data_path.split("/")[-1]
    meta_filename = f"{filename}.metadata.json"
    perf_filename = f"{filename}.perf.json"

    meta_path = str(METADATA_DIR / meta_filename)
    perf_path = str(PERF_DIR / perf_filename)

    unzipped_size = ds.uncompressedSizeBytes
    if (unzipped_size is None) and data_path.lower().endswith(".zip"):
        unzipped_size = uncompressed_zip_size(data_path)

    theme_names = _extract_theme_names(meta_dict.get("themes"))

    entry = {
        "id": dataset_id_from_path(data_path),
        "title": ds.title,

        # provenance
        "sourcePath": ds.sourceAddress,
        "sourceName": ds.sourceName,
        "sourceType": ds.sourceType,

        # storage / outputs
        "inputObjectKey": data_path,
        "rawzonePath": ds.rawzonePath,
        "rawzoneCopied": False,
        "sourceMtime": _file_mtime_int(data_path),
        "metadataPath": meta_path,
        "perfPath": perf_path,
        "status": "ok",

        # descriptors
        "fileType": ds.fileType,
        "dataFormat": ds.dataFormat,
        "updateFrequency": ds.updateFrequency,
        "themes": theme_names,
        "spatialGranularity": ds.spatialGranularity,
        "temporalGranularity": ds.temporalGranularity,
        "spatialScope": meta_dict.get("spatialScope"),
        "temporalScope": meta_dict.get("temporalScope"),

        # file metrics
        "fileSizeBytes": ds.fileSizeBytes,
        "fileSizeHuman": ds.fileSizeHuman,
        "nRows": ds.nRows,
        "nCols": ds.nCols,
        "nRecords": ds.nRecords,
        "nFeatures": ds.nFeatures,
        "uncompressedSizeBytes": unzipped_size,

        # system info
        "checksum": file_checksum(data_path),
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": 1,
    }
    return entry


def build_perf_entry(data_path: str, timings: Dict[str, float], ds: Dataset) -> Dict[str, Any]:
    """
    Build one perf log entry.
    """
    return {
        "datasetId": dataset_id_from_path(data_path),
        "inputObjectKey": data_path,
        "rawzonePath": ds.rawzonePath,
        "fileSizeBytes": ds.fileSizeBytes,
        "timings": timings,
        "executedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }


def load_catalog() -> Dict[str, Any]:
    """
    Load existing catalog from MinIO or return an empty skeleton.
    """
    if object_exists(CATALOG_PATH):
        return read_json_object(CATALOG_PATH)
    return {"datasets": []}


def upsert_catalog_entry(catalog: Dict[str, Any], new_entry: Dict[str, Any]) -> None:
    """
    Insert or update a dataset entry in catalog, keyed by id.
    """
    datasets = catalog.setdefault("datasets", [])
    for i, e in enumerate(datasets):
        if e.get("id") == new_entry["id"]:
            datasets[i] = new_entry
            break
    else:
        datasets.append(new_entry)


def write_catalog(catalog: Dict[str, Any]) -> None:
    """
    Persist the current catalog state to MinIO.
    """
    catalog.setdefault("datasets", []).sort(
        key=lambda x: ((x.get("title") or "").lower(), x.get("id", ""))
    )
    write_json_object(CATALOG_PATH, catalog, indent=4)


# ------------------------------- Processing I/O -------------------------------

def process_file(path: str):
    ds, timings = construct_dataset(path, measure=True)

    input_relative_path = os.path.relpath(path, start=str(DATA_DIR)).replace("\\", "/")
    copied = copy_input_object_to_rawzone(input_relative_path, ds.rawzonePath)

    meta_dict = ds.to_dict()
    meta_entry = build_metadata_entry(path, ds)
    meta_entry["rawzoneCopied"] = copied or object_exists("openlake", ds.rawzonePath)

    perf_entry = None
    if not timings.get("timing_skipped"):
        perf_entry = build_perf_entry(path, timings, ds)

    return meta_dict, meta_entry, perf_entry

def save_outputs(meta_dict, entry, perf_entry):
    """
    Save metadata and perf outputs to MinIO.
    """
    meta_path = entry["metadataPath"]
    write_json_object(meta_path, meta_dict, indent=2)

    if perf_entry:
        perf_path = entry["perfPath"]
        write_json_object(perf_path, perf_entry, indent=2)


# ----------------------------- Parallel catalogue -----------------------------

def build_catalog_for_dir_parallel(
    data_dir=DATA_DIR,
    max_workers: int = 4,
    allow_exts=None,
    *,
    force: bool = False,
) -> None:
    """
    Parallel metadata + performance extraction for all input objects under `data_dir`.

    Here `data_dir` is a MinIO prefix such as:
        input_data
    """
    paths = list(iter_data_files(data_dir, allow_exts=allow_exts or DEFAULT_EXTS))
    if not paths:
        print(f"No files found under '{data_dir}'.")
        return

    catalog = load_catalog()

    if force:
        pending = paths
        skipped = 0
        print("[INFO] force=True -> will reprocess all files.")
    else:
        pending = [p for p in paths if not was_processed_successfully(p, catalog)]
        skipped = len(paths) - len(pending)

    if skipped:
        print(f"Skipped {skipped}/{len(paths)} file(s) already processed and up-to-date.")
    if not pending:
        print("All files are up-to-date. Nothing to do.")
        return

    print(f"Discovered {len(paths)} file(s); {len(pending)} to process; max_workers={max_workers}.")

    if max_workers <= 1:
        for p in pending:
            fn = p.split("/")[-1]
            try:
                meta_dict, meta_entry, perf_entry = process_file(p)
                save_outputs(meta_dict, meta_entry, perf_entry)
                upsert_catalog_entry(catalog, meta_entry)
                write_catalog(catalog)
                print(f"[OK] {fn}")
            except Exception as ex:
                print(f"[ERR] {fn}: {ex}")

        print(f"Catalog written to {CATALOG_PATH}")
        return

    futures: Dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p in pending:
            futures[ex.submit(process_file, p)] = p

        for fut in as_completed(futures):
            p = futures[fut]
            fn = p.split("/")[-1]
            try:
                meta_dict, meta_entry, perf_entry = fut.result()
                save_outputs(meta_dict, meta_entry, perf_entry)
                upsert_catalog_entry(catalog, meta_entry)
                write_catalog(catalog)
                print(f"[OK] {fn}")
            except Exception as ex:
                print(f"[ERR] {fn}: {ex}")

    print(f"Catalog written to {CATALOG_PATH}")


if __name__ == "__main__":
    build_catalog_for_dir_parallel(
        DATA_DIR,
        max_workers=8,
        allow_exts=DEFAULT_EXTS,
        force=False
    )
