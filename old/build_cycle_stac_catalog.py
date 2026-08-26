#!/usr/bin/env python3
"""Build a cycle-based STAC catalog from an existing SWOT region folder.

This script is metadata-only. It does NOT reprocess NetCDF files into COGs.
It scans the Bay of Bengal folder structure you already have and writes a
clean STAC hierarchy:

Input folder shape expected:

  <region-root>/
    raw_nc/
      cycle_001/
        SWOT_L3_LR_SSH_Expert_001_217_20230728T224620_20230728T233747_v3.0.nc
      cycle_002/
        ...

    processed/
      cycle_001/
        cogs/
          swot_cycle_001_pass_217_ssha_unfiltered_cog.tif
        nadir_geojson/
          swot_cycle_001_pass_217_nadir.geojson
      cycle_002/
        ...

Output STAC folder shape:

  <output-root>/
    catalog.json
    build_report.json
    collections/
      bay_of_bengal_swot_cycle_001/
        collection.json
        items.geojson
        items/
          bay_of_bengal_swot_cycle_001_pass_217.json
      bay_of_bengal_swot_cycle_002/
        ...

STAC meaning for this project:

  Catalog    = Bay of Bengal SWOT catalog
  Collection = one SWOT cycle, such as cycle 001
  Item       = one SWOT pass inside that cycle, such as pass 217
  Assets     = files attached to that pass, such as COG, NetCDF, nadir GeoJSON

Each pass item gets:

  properties:
    cycle
    pass
    direction                 odd pass = ascending, even pass = descending
    start_datetime / end_datetime
    selected source_* fields  copied from the raw NetCDF global attributes
    model_runs: []            placeholder for future Johnny/Jacob model outputs

  assets:
    asset   COG used by TiTiler when the URL uses ?assets=asset
    ssha    same COG, but with a more readable asset name
    netcdf  original SWOT NetCDF, used later for analysis / Johnny model input
    nadir   nadir GeoJSON, kept for pass-track display

If metadata cannot be read from a NetCDF, the STAC item stays clean. The error
is written to build_report.json instead of being stored inside the item.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio


# -----------------------------------------------------------------------------
# Filename patterns
# -----------------------------------------------------------------------------
# We need to match both processed output filenames and original NetCDF filenames.
# The whole script depends on matching files by the pair:
#
#   (cycle, pass)
#
# Example:
#   COG:    swot_cycle_001_pass_217_ssha_unfiltered_cog.tif
#   NetCDF: SWOT_L3_LR_SSH_Expert_001_217_20230728T224620_20230728T233747_v3.0.nc
#   Nadir:  swot_cycle_001_pass_217_nadir.geojson
#
# All three become key = ("001", "217").
CYCLE_PASS_PATTERNS = [
    # Matches: swot_cycle_001_pass_217_ssha_unfiltered_cog.tif
    re.compile(
        r"cycle[_-](?P<cycle>\d{3}).*?pass[_-](?P<pass>\d{3})",
        re.IGNORECASE,
    ),

    # Matches: SWOT_L3_LR_SSH_Expert_001_217_20230728T224620_20230728T233747_v3.0.nc
    re.compile(r"_(?P<cycle>\d{3})_(?P<pass>\d{3})_"),
]

# The original SWOT filenames contain start and end timestamps.
# We parse these as a fallback if the NetCDF global attributes are unavailable.
SWOT_TIME_PATTERN = re.compile(
    r"_(?P<cycle>\d{3})_(?P<pass>\d{3})_"
    r"(?P<start>\d{8}T\d{6})_(?P<end>\d{8}T\d{6})_"
)


# -----------------------------------------------------------------------------
# Selected NetCDF global attributes to copy into STAC properties
# -----------------------------------------------------------------------------
# Keep this intentionally small. The raw NetCDF contains many global attributes,
# but copying all of them makes each STAC item difficult to read.
#
# Left side  = NetCDF global attribute name.
# Right side = STAC property name.
#
# The original NetCDF is still linked as assets.netcdf, so the full metadata is
# still available later if someone opens that file directly.
NETCDF_ATTRIBUTE_MAP = {
    "title": "source_title",
    "processing_level": "source_processing_level",
    "product_version": "source_product_version",
    "data_used": "source_data_used",
}


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--region-root",
        type=Path,
        required=True,
        help=(
            "Region folder containing raw_nc/ and processed/. "
            "Example: ~/Documents/MOSAICS-2026/SWOT-Data/Bay of Bengal"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Where to write the STAC catalog. "
            "Default: <region-root>/stac"
        ),
    )

    parser.add_argument(
        "--catalog-id",
        default="bay_of_bengal_swot",
        help=(
            "Top-level catalog id and prefix for cycle collection ids. "
            "Default: bay_of_bengal_swot"
        ),
    )

    parser.add_argument(
        "--catalog-title",
        default="Bay of Bengal SWOT",
        help="Human-readable catalog title. Default: Bay of Bengal SWOT.",
    )

    parser.add_argument(
        "--region-name",
        default="Bay of Bengal",
        help="Human-readable region name. Default: Bay of Bengal.",
    )

    parser.add_argument(
        "--variable",
        default="ssha_unfiltered",
        help="Variable represented by the COG. Default: ssha_unfiltered.",
    )

    parser.add_argument(
        "--processed-asset-root",
        type=Path,
        default=None,
        help=(
            "Optional alternate root for processed assets in STAC hrefs. "
            "Use this if TiTiler reads from a no-space symlink folder. "
            "Paths under <region-root>/processed are rewritten under this root. "
            "The original files are not moved. Only the href text changes."
        ),
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional maximum number of COG items to catalog, useful for testing.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root first if it already exists.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Small helper functions
# -----------------------------------------------------------------------------
def parse_cycle_pass(path: Path) -> tuple[str, str] | None:
    """Extract the three-digit cycle and pass from a filename.

    Returns:
        (cycle, pass) if the filename matches one of the known patterns.
        None if the filename does not look like one of our SWOT files.
    """

    name = path.name

    for pattern in CYCLE_PASS_PATTERNS:
        match = pattern.search(name)
        if match:
            return match.group("cycle"), match.group("pass")

    return None


def normalize_datetime(value: str | None) -> str | None:
    """Normalize SWOT timestamps to STAC-friendly UTC strings.

    Supports:
      - filename style: 20230728T224620
      - ISO style:      2023-07-28T22:46:20Z

    Returns:
      - ISO UTC string ending in Z, or None if the value is missing/unreadable.
    """

    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    # Already ISO-like. Keep it, but force a Z suffix when needed.
    if "-" in value[:10]:
        if value.endswith("Z"):
            return value

        if value.endswith("+00:00"):
            return value.replace("+00:00", "Z")

        return f"{value}Z"

    # Filename form: YYYYMMDDTHHMMSS
    try:
        dt = datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        return None

    dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def parse_times_from_name(path: Path) -> tuple[str | None, str | None]:
    """Extract start/end timestamps from a standard SWOT NetCDF filename."""

    match = SWOT_TIME_PATTERN.search(path.name)
    if not match:
        return None, None

    return (
        normalize_datetime(match.group("start")),
        normalize_datetime(match.group("end")),
    )


def direction_from_pass(pass_number: str) -> str:
    """Return ascending/descending from SWOT pass number.

    Rule from your project:
      - odd pass/path number  -> ascending
      - even pass/path number -> descending
    """

    try:
        value = int(pass_number)
    except ValueError:
        return "unknown"

    return "ascending" if value % 2 == 1 else "descending"


def json_safe(value: Any) -> Any:
    """Convert HDF5/NumPy metadata values into JSON-safe Python values.

    NetCDF global attributes can come back as bytes, NumPy scalars, or arrays.
    json.dumps() cannot reliably serialize those directly, so this function
    converts them to plain strings, numbers, or lists.
    """

    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return json_safe(value.item())

        return [json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    # String is safest for metadata because many NetCDF attributes are text.
    return str(value)


def read_netcdf_source_metadata(
    netcdf_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read selected source/provenance attributes from a raw NetCDF.

    We do NOT put read/error status into the STAC item.

    Returns:
      source_metadata:
        selected source_* fields to merge into item.properties.

      warning:
        None when everything is fine.
        A dictionary when the file could not be read. The warning goes into
        build_report.json, not into the STAC item.
    """

    if netcdf_path is None:
        return {}, None

    metadata: dict[str, Any] = {}

    try:
        with h5py.File(netcdf_path, "r") as dataset:
            for nc_key, stac_key in NETCDF_ATTRIBUTE_MAP.items():
                if nc_key not in dataset.attrs:
                    continue

                metadata[stac_key] = json_safe(dataset.attrs[nc_key])

    except Exception as error:
        return {}, {
            "type": "netcdf_metadata_read_failed",
            "netcdf": str(netcdf_path),
            "error": str(error),
        }

    return metadata, None


def file_href(path: Path) -> str:
    """Create a local file:// href for STAC assets.

    For now we keep spaces unescaped because this matches your current local
    MMGIS/TiTiler setup style.
    """

    return f"file://{path.expanduser().resolve().as_posix()}"


def maybe_rewrite_processed_path(
    path: Path,
    processed_root: Path,
    processed_asset_root: Path | None,
) -> Path:
    """Optionally rewrite processed asset hrefs under another root.

    Example use case:
      Your real files are under:
        ~/Documents/MOSAICS-2026/SWOT-Data/Bay of Bengal/processed/...

      But TiTiler reads from a symlink without spaces:
        ~/Documents/MOSAICS-2026/SWOT-Data/Bay_of_Bengal/processed/...

    This function changes only the path written into STAC. It does not move or
    copy data.
    """

    path = path.expanduser().resolve()

    if processed_asset_root is None:
        return path

    processed_root = processed_root.expanduser().resolve()
    processed_asset_root = processed_asset_root.expanduser().resolve()

    try:
        relative = path.relative_to(processed_root)
    except ValueError:
        # If the file is not under processed_root, leave it unchanged.
        return path

    return processed_asset_root / relative


def polygon_from_bounds(bounds: list[float]) -> dict[str, Any]:
    """Create a simple rectangular GeoJSON polygon from raster bounds.

    This is the STAC footprint for the COG. It is only a rectangular bounding
    footprint, not the exact curved SWOT swath shape.
    """

    west, south, east, north = bounds

    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def index_files_by_cycle_pass(files: list[Path]) -> dict[tuple[str, str], Path]:
    """Build a lookup table from (cycle, pass) to file path."""

    indexed: dict[tuple[str, str], Path] = {}

    for path in files:
        parsed = parse_cycle_pass(path)
        if parsed is None:
            continue

        # Keep the first match so the result is deterministic.
        indexed.setdefault(parsed, path.expanduser().resolve())

    return indexed


def collection_extent(items: list[dict[str, Any]]) -> tuple[list[float], list[str | None]]:
    """Compute the spatial and temporal extent for a cycle collection."""

    west_values = []
    south_values = []
    east_values = []
    north_values = []
    times = []

    for item in items:
        bbox = item["bbox"]
        west_values.append(bbox[0])
        south_values.append(bbox[1])
        east_values.append(bbox[2])
        north_values.append(bbox[3])

        start_time = item["properties"].get("start_datetime")
        end_time = item["properties"].get("end_datetime")

        if start_time:
            times.append(start_time)

        if end_time:
            times.append(end_time)

    bbox = [
        min(west_values),
        min(south_values),
        max(east_values),
        max(north_values),
    ]

    if times:
        interval: list[str | None] = [min(times), max(times)]
    else:
        interval = [None, None]

    return bbox, interval


# -----------------------------------------------------------------------------
# STAC item creation
# -----------------------------------------------------------------------------
def make_item(
    cog_path: Path,
    raw_nc_path: Path | None,
    nadir_path: Path | None,
    processed_root: Path,
    processed_asset_root: Path | None,
    catalog_id: str,
    region_name: str,
    variable: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Create one STAC Item for one COG/pass.

    One COG corresponds to one SWOT pass item. We attach the matching raw NetCDF
    and nadir GeoJSON when they exist.
    """

    warnings: list[dict[str, Any]] = []

    parsed = parse_cycle_pass(cog_path)
    if parsed is None:
        warnings.append(
            {
                "type": "unrecognized_cog_filename",
                "cog": str(cog_path),
            }
        )
        return None, warnings

    cycle, pass_number = parsed
    collection_id = f"{catalog_id}_cycle_{cycle}"
    item_id = f"{collection_id}_pass_{pass_number}"

    # The COG bounds become the item's bbox and rectangular geometry.
    # STAC needs these so the item can be spatially searched/displayed.
    try:
        with rasterio.open(cog_path) as dataset:
            bounds = dataset.bounds
            bbox = [
                float(bounds.left),
                float(bounds.bottom),
                float(bounds.right),
                float(bounds.top),
            ]
    except Exception as error:
        warnings.append(
            {
                "type": "cog_bounds_read_failed",
                "cycle": cycle,
                "pass": pass_number,
                "cog": str(cog_path),
                "error": str(error),
            }
        )
        return None, warnings

    geometry = polygon_from_bounds(bbox)

    # Start/end times come from the filename first.
    # If NetCDF global attributes include time coverage, we prefer those.
    start_time = None
    end_time = None
    source_file_name = None
    source_metadata: dict[str, Any] = {}

    if raw_nc_path is not None:
        source_file_name = raw_nc_path.name
        start_time, end_time = parse_times_from_name(raw_nc_path)

        source_metadata, metadata_warning = read_netcdf_source_metadata(raw_nc_path)
        if metadata_warning is not None:
            metadata_warning.update(
                {
                    "cycle": cycle,
                    "pass": pass_number,
                }
            )
            warnings.append(metadata_warning)

        # Prefer NetCDF global attributes if they exist.
        start_time = normalize_datetime(
            source_metadata.get("source_time_coverage_start")
        ) or start_time

        end_time = normalize_datetime(
            source_metadata.get("source_time_coverage_end")
        ) or end_time

    datetime_value = start_time or end_time

    # Rewrite processed asset paths only if requested by --processed-asset-root.
    # This is useful when TiTiler needs a no-space symlink path.
    rewritten_cog_path = maybe_rewrite_processed_path(
        cog_path,
        processed_root=processed_root,
        processed_asset_root=processed_asset_root,
    )

    assets: dict[str, Any] = {}

    cog_asset = {
        "href": file_href(rewritten_cog_path),
        "type": "image/tiff; application=geotiff; profile=cloud-optimized",
        "roles": ["data"],
        "title": "SWOT SSHA unfiltered COG",
    }

    # Keep both names:
    #   asset -> your current TiTiler URLs use ?assets=asset
    #   ssha  -> more readable name for humans
    assets["asset"] = cog_asset
    assets["ssha"] = cog_asset

    if raw_nc_path is not None:
        assets["netcdf"] = {
            "href": file_href(raw_nc_path),
            "type": "application/netcdf",
            "roles": ["source", "data"],
            "title": "Original SWOT L3 NetCDF",
        }

    if nadir_path is not None:
        rewritten_nadir_path = maybe_rewrite_processed_path(
            nadir_path,
            processed_root=processed_root,
            processed_asset_root=processed_asset_root,
        )

        assets["nadir"] = {
            "href": file_href(rewritten_nadir_path),
            "type": "application/geo+json",
            "roles": ["metadata", "overview"],
            "title": "SWOT nadir track GeoJSON",
        }

    properties: dict[str, Any] = {
        "datetime": datetime_value,
        "start_datetime": start_time,
        "end_datetime": end_time,
        "region": region_name,
        "cycle": cycle,
        "pass": pass_number,
        "direction": direction_from_pass(pass_number),
        "variable": variable,
        "source_file": source_file_name,

        # Future detector outputs can be added here, or stored in a separate
        # model-run sidecar catalog. For now this stays empty.
        "model_runs": [],
    }

    # Add only the selected clean NetCDF source fields.
    # No read/error debug fields are added to the STAC item.
    properties.update(source_metadata)

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "collection": collection_id,
        "bbox": bbox,
        "geometry": geometry,
        "properties": properties,
        "links": [
            {
                "rel": "collection",
                "href": "../collection.json",
                "type": "application/json",
            },
            {
                "rel": "parent",
                "href": "../collection.json",
                "type": "application/json",
            },
            {
                "rel": "root",
                "href": "../../catalog.json",
                "type": "application/json",
            },
            {
                "rel": "self",
                "href": f"items/{item_id}.json",
                "type": "application/json",
            },
        ],
        "assets": assets,
    }

    return item, warnings


# -----------------------------------------------------------------------------
# File writing helpers
# -----------------------------------------------------------------------------
def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary as pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def safe_delete_output_root(
    output_root: Path,
    region_root: Path,
    processed_root: Path,
    raw_nc_root: Path,
) -> None:
    """Delete output_root safely for --overwrite.

    This prevents accidents like deleting the whole region folder or raw data.
    """

    output_root = output_root.expanduser().resolve()

    unsafe_paths = {
        Path("/").resolve(),
        region_root.expanduser().resolve(),
        processed_root.expanduser().resolve(),
        raw_nc_root.expanduser().resolve(),
    }

    if output_root in unsafe_paths:
        raise ValueError(f"Refusing to delete unsafe output root: {output_root}")

    shutil.rmtree(output_root)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    region_root = args.region_root.expanduser().resolve()
    raw_nc_root = region_root / "raw_nc"
    processed_root = region_root / "processed"

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else region_root / "stac"
    )

    # Validate the input folder structure before doing any work.
    if not region_root.exists():
        raise FileNotFoundError(f"Region root does not exist: {region_root}")

    if not raw_nc_root.exists():
        raise FileNotFoundError(f"raw_nc folder does not exist: {raw_nc_root}")

    if not processed_root.exists():
        raise FileNotFoundError(f"processed folder does not exist: {processed_root}")

    # For development, it is useful to rebuild the STAC output from scratch.
    if output_root.exists() and args.overwrite:
        safe_delete_output_root(
            output_root=output_root,
            region_root=region_root,
            processed_root=processed_root,
            raw_nc_root=raw_nc_root,
        )

    # Source of truth for STAC items:
    # every processed COG becomes one STAC item.
    cog_files = sorted(processed_root.glob("cycle_*/cogs/*.tif")) + sorted(
        processed_root.glob("cycle_*/cogs/*.tiff")
    )

    if args.max_items is not None:
        cog_files = cog_files[: args.max_items]

    # The raw NetCDF and nadir files are matched to the COGs by cycle/pass.
    raw_nc_files = sorted(raw_nc_root.rglob("*.nc"))
    nadir_files = sorted(processed_root.glob("cycle_*/nadir_geojson/*.geojson"))

    raw_nc_by_key = index_files_by_cycle_pass(raw_nc_files)
    nadir_by_key = index_files_by_cycle_pass(nadir_files)

    print("Building cycle-based SWOT STAC catalog")
    print(f"Region root:       {region_root}")
    print(f"Processed root:    {processed_root}")
    print(f"Raw NetCDF files:  {len(raw_nc_files)}")
    print(f"COG files:         {len(cog_files)}")
    print(f"Nadir files:       {len(nadir_files)}")
    print(f"Output root:       {output_root}")
    print()

    items_by_cycle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    warnings: list[dict[str, Any]] = []

    missing_netcdf_count = 0
    missing_nadir_count = 0
    skipped_cog_count = 0

    # Build every STAC item.
    for cog_path in cog_files:
        parsed = parse_cycle_pass(cog_path)
        if parsed is None:
            skipped_cog_count += 1
            warnings.append(
                {
                    "type": "unrecognized_cog_filename",
                    "cog": str(cog_path),
                }
            )
            continue

        cycle, pass_number = parsed
        key = (cycle, pass_number)

        raw_nc_path = raw_nc_by_key.get(key)
        nadir_path = nadir_by_key.get(key)

        if raw_nc_path is None:
            missing_netcdf_count += 1
            warnings.append(
                {
                    "type": "missing_netcdf_asset",
                    "cycle": cycle,
                    "pass": pass_number,
                    "cog": str(cog_path),
                }
            )

        if nadir_path is None:
            missing_nadir_count += 1
            warnings.append(
                {
                    "type": "missing_nadir_asset",
                    "cycle": cycle,
                    "pass": pass_number,
                    "cog": str(cog_path),
                }
            )

        item, item_warnings = make_item(
            cog_path=cog_path,
            raw_nc_path=raw_nc_path,
            nadir_path=nadir_path,
            processed_root=processed_root,
            processed_asset_root=args.processed_asset_root,
            catalog_id=args.catalog_id,
            region_name=args.region_name,
            variable=args.variable,
        )

        warnings.extend(item_warnings)

        if item is None:
            skipped_cog_count += 1
            continue

        items_by_cycle[cycle].append(item)

    if not items_by_cycle:
        raise RuntimeError("No STAC items were created. Check COG filenames and folder paths.")

    catalog_links = [
        {
            "rel": "self",
            "href": "catalog.json",
            "type": "application/json",
        }
    ]

    total_items = 0

    # Write one STAC Collection per cycle.
    for cycle in sorted(items_by_cycle):
        items = sorted(items_by_cycle[cycle], key=lambda item: item["id"])
        total_items += len(items)

        collection_id = f"{args.catalog_id}_cycle_{cycle}"
        collection_dir = output_root / "collections" / collection_id
        items_dir = collection_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)

        collection_bbox, temporal_interval = collection_extent(items)

        collection = {
            "type": "Collection",
            "stac_version": "1.0.0",
            "id": collection_id,
            "title": f"{args.catalog_title} Cycle {cycle}",
            "description": f"SWOT L3 SSHA COGs for {args.region_name}, cycle {cycle}.",
            "license": "proprietary",
            "extent": {
                "spatial": {"bbox": [collection_bbox]},
                "temporal": {"interval": [temporal_interval]},
            },
            "summaries": {
                "region": [args.region_name],
                "cycle": [cycle],
                "variable": [args.variable],
                "direction": ["ascending", "descending"],
            },
            "providers": [
                {
                    "name": "AVISO/DUACS",
                    "roles": ["producer", "licensor"],
                    "url": "https://aviso.altimetry.fr",
                },
                {
                    "name": "NASA/JPL and CNES",
                    "roles": ["producer"],
                    "url": "https://swot.jpl.nasa.gov/",
                },
            ],
            "links": [
                {
                    "rel": "root",
                    "href": "../../catalog.json",
                    "type": "application/json",
                },
                {
                    "rel": "self",
                    "href": "collection.json",
                    "type": "application/json",
                },
            ],
        }

        # Write each item as its own JSON file and link it from collection.json.
        for item in items:
            item_filename = f"{item['id']}.json"
            item_path = items_dir / item_filename
            write_json(item_path, item)

            collection["links"].append(
                {
                    "rel": "item",
                    "href": f"items/{item_filename}",
                    "type": "application/json",
                }
            )

        # Also write items.geojson because MMGIS/STAC browsers can display a
        # FeatureCollection of all pass items in the cycle.
        write_json(collection_dir / "collection.json", collection)
        write_json(
            collection_dir / "items.geojson",
            {
                "type": "FeatureCollection",
                "features": items,
            },
        )

        catalog_links.append(
            {
                "rel": "child",
                "href": f"collections/{collection_id}/collection.json",
                "type": "application/json",
                "title": f"{args.catalog_title} Cycle {cycle}",
            }
        )

        print(f"Cycle {cycle}: {len(items)} items")

    # Write the top-level catalog linking to all cycle collections.
    catalog = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": args.catalog_id,
        "title": args.catalog_title,
        "description": f"Cycle-based STAC catalog for {args.catalog_title}.",
        "links": catalog_links,
    }

    write_json(output_root / "catalog.json", catalog)

    # Write a separate report for debugging, instead of putting debug fields in
    # each STAC item.
    build_report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "catalog_id": args.catalog_id,
        "catalog_title": args.catalog_title,
        "region_name": args.region_name,
        "region_root": str(region_root),
        "processed_root": str(processed_root),
        "raw_nc_root": str(raw_nc_root),
        "output_root": str(output_root),
        "input_counts": {
            "raw_netcdf_files": len(raw_nc_files),
            "cog_files_used": len(cog_files),
            "nadir_files": len(nadir_files),
        },
        "output_counts": {
            "collections_created": len(items_by_cycle),
            "items_created": total_items,
            "skipped_cogs": skipped_cog_count,
            "missing_netcdf_assets": missing_netcdf_count,
            "missing_nadir_assets": missing_nadir_count,
            "warnings": len(warnings),
        },
        "warnings": warnings,
    }

    write_json(output_root / "build_report.json", build_report)

    print()
    print("Done")
    print(f"Collections:          {len(items_by_cycle)}")
    print(f"Items:                {total_items}")
    print(f"Missing NetCDF asset: {missing_netcdf_count}")
    print(f"Missing nadir asset:  {missing_nadir_count}")
    print(f"Warnings:             {len(warnings)}")
    print(f"Catalog:              {output_root / 'catalog.json'}")
    print(f"Build report:         {output_root / 'build_report.json'}")


if __name__ == "__main__":
    main()
