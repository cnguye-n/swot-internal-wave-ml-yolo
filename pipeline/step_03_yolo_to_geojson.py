#!/usr/bin/env python3
"""
step_03_yolo_to_geojson.py

Purpose:
    Convert internal-wave YOLO per-box mask NetCDF outputs into GeoJSON polygons.

Preferred behavior:
    Use internal_wave_box_masks:
        one mask per YOLO box

    This creates:
        one GeoJSON feature per YOLO box

Fallback behavior:
    If the NetCDF was made by the old step 02 and does not have
    internal_wave_box_masks, use the old combined internal_wave_bbox_mask
    and connected components.

Why:
    The old combined mask can merge YOLO boxes if they touch/overlap.
    The new per-box mask preserves:
        - one feature per YOLO box
        - real confidence value per YOLO box
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


PLUGIN_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_MANIFEST = (
    PLUGIN_ROOT / "pipeline" / "model_outputs" / "iw_yolo_mask_outputs.json"
)

DEFAULT_OUTPUT_GEOJSON = (
    PLUGIN_ROOT / "pipeline" / "model_outputs" / "iw_yolo_detections.geojson"
)

MASK_VARIABLE_NAME = "internal_wave_bbox_mask"
YOLO_BOX_MASK_VARIABLE_NAME = "internal_wave_box_masks"


def load_manifest(path: Path) -> list[dict]:
    """
    Read the step 02 manifest.

    This file points to every mask.nc produced by the internal-wave YOLO model.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError("Manifest must contain a list.")

    return records


def get_2d_array(ds: xr.Dataset, variable_name: str) -> np.ndarray:
    """
    Read a variable from xarray and make sure it is 2D.
    """
    if variable_name not in ds:
        raise KeyError(f"Variable not found in NetCDF: {variable_name}")

    arr = np.squeeze(ds[variable_name].values)

    if arr.ndim != 2:
        raise ValueError(
            f"{variable_name} should be 2D after squeeze, got shape {arr.shape}"
        )

    return arr


def read_per_box_masks(ds: xr.Dataset) -> np.ndarray | None:
    """
    Read internal_wave_box_masks if it exists.

    Expected shape:
        detection x num_lines x num_pixels
    """
    if YOLO_BOX_MASK_VARIABLE_NAME not in ds:
        return None

    arr = ds[YOLO_BOX_MASK_VARIABLE_NAME].values

    if arr.ndim != 3:
        raise ValueError(
            f"{YOLO_BOX_MASK_VARIABLE_NAME} should be 3D "
            f"(detection, rows, cols), got shape {arr.shape}"
        )

    return np.nan_to_num(arr, nan=0).astype(np.uint8)


def nearest_valid_point(
    rows: np.ndarray,
    cols: np.ndarray,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    target_row: int,
    target_col: int,
) -> list[float]:
    """
    Find the valid mask pixel closest to a desired row/column corner.

    This is used as a fallback if ConvexHull fails.
    """
    distances = (rows - target_row) ** 2 + (cols - target_col) ** 2
    index = int(np.argmin(distances))

    return [float(lon_values[index]), float(lat_values[index])]


def fallback_quadrilateral_from_valid_mask(
    valid: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> list[list[float]]:
    """
    Build a slanted quadrilateral from the valid mask's row/column corners.
    """
    rows, cols = np.where(valid)

    lat_values = latitude[valid]
    lon_values = longitude[valid]

    min_row = int(np.min(rows))
    max_row = int(np.max(rows))
    min_col = int(np.min(cols))
    max_col = int(np.max(cols))

    top_left = nearest_valid_point(
        rows, cols, lon_values, lat_values, min_row, min_col
    )
    top_right = nearest_valid_point(
        rows, cols, lon_values, lat_values, min_row, max_col
    )
    bottom_right = nearest_valid_point(
        rows, cols, lon_values, lat_values, max_row, max_col
    )
    bottom_left = nearest_valid_point(
        rows, cols, lon_values, lat_values, max_row, min_col
    )

    coords = [top_left, top_right, bottom_right, bottom_left, top_left]

    return coords


def footprint_polygon_from_valid_mask(
    valid_mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[list[list[float]], dict] | None:
    """
    Convert one binary mask into a GeoJSON polygon.

    This is used for:
        - one YOLO box mask, preferred
        - one connected component, fallback
    """
    valid = (
        valid_mask.astype(bool)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
    )

    if not np.any(valid):
        return None

    lats = latitude[valid].astype(float)
    lons = longitude[valid].astype(float)

    if len(lats) < 3:
        return None

    west = float(np.nanmin(lons))
    south = float(np.nanmin(lats))
    east = float(np.nanmax(lons))
    north = float(np.nanmax(lats))

    properties = {
        "bbox_west": west,
        "bbox_south": south,
        "bbox_east": east,
        "bbox_north": north,
        "pixel_count": int(np.count_nonzero(valid)),
    }

    points = np.column_stack([lons, lats])

    # Remove duplicate lon/lat points so the hull is faster and cleaner.
    points = np.unique(np.round(points, 7), axis=0)

    if len(points) < 3:
        return None

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        coords = points[hull.vertices].tolist()

        # Close GeoJSON polygon ring.
        coords.append(coords[0])

        return coords, properties

    except Exception:
        coords = fallback_quadrilateral_from_valid_mask(
            valid=valid,
            latitude=latitude,
            longitude=longitude,
        )

        return coords, properties


def find_connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Fallback only.

    Find connected regions where combined mask == 1.
    """
    binary_mask = mask.astype(bool)

    if not np.any(binary_mask):
        return np.zeros_like(mask, dtype=np.int32), 0

    from scipy import ndimage

    structure = np.ones((3, 3), dtype=np.int8)
    labels, number_of_labels = ndimage.label(binary_mask, structure=structure)

    return labels.astype(np.int32), int(number_of_labels)


def get_detection_threshold(record: dict):
    """
    Prefer the new name, but support the old name.
    """
    return record.get("detection_threshold", record.get("confidence"))


def get_box_record(record: dict, zero_based_index: int) -> dict:
    """
    Get the matching YOLO box metadata from the step 02 manifest.
    """
    yolo_boxes = record.get("yolo_boxes", [])

    if zero_based_index < len(yolo_boxes):
        return yolo_boxes[zero_based_index]

    return {}


def features_from_per_box_masks(
    record: dict,
    ds: xr.Dataset,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> list[dict] | None:
    """
    Preferred path:
        one GeoJSON feature per YOLO box.
    """
    box_masks = read_per_box_masks(ds)

    if box_masks is None:
        return None

    features = []

    for zero_based_index in range(box_masks.shape[0]):
        box_index = zero_based_index + 1
        single_box_mask = box_masks[zero_based_index]

        result = footprint_polygon_from_valid_mask(
            valid_mask=single_box_mask,
            latitude=latitude,
            longitude=longitude,
        )

        if result is None:
            continue

        coords, footprint_properties = result
        box_record = get_box_record(record, zero_based_index)

        feature_id = (
            f"iw_yolo_cycle_{record.get('cycle')}"
            f"_pass_{record.get('pass')}"
            f"_box_{box_index:03d}"
        )

        feature = {
            "type": "Feature",
            "properties": {
                "id": feature_id,

                # Friendly name for frontend popup.
                "model": "Bay of Bengal Internal Wave Yolo Detector",

                # Stable id for code/filtering.
                "model_id": "iw_yolo",

                "cycle": record.get("cycle"),
                "pass": record.get("pass"),
                "source_item_id": record.get("item_id"),
                "mask_netcdf": str(Path(record["mask_netcdf"]).expanduser().resolve()),

                # Threshold used to keep/reject YOLO predictions.
                "detection_threshold": get_detection_threshold(record),

                # Actual model confidence for this YOLO box.
                "confidence_value": box_record.get("confidence_value"),

                "box_index": box_record.get("box_index", box_index),
                "class_id": box_record.get("class_id"),
                "class_name": box_record.get("class_name"),
                "image_xyxy": box_record.get("image_xyxy"),
                "original_grid_bbox": box_record.get("original_grid_bbox"),
                "filter_type": record.get("filter_type"),
                "geometry_method": "per_yolo_box_mask_footprint_convex_hull",
                **footprint_properties,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        }

        features.append(feature)

    return features


def features_from_combined_mask_fallback(
    record: dict,
    ds: xr.Dataset,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> list[dict]:
    """
    Fallback for old mask files made before we saved per-box masks.

    This creates one feature per connected component, not necessarily one
    feature per original YOLO box.
    """
    mask_variable = record.get("mask_variable", MASK_VARIABLE_NAME)
    mask = get_2d_array(ds, mask_variable)

    mask = np.nan_to_num(mask, nan=0).astype(np.uint8)

    labels, number_of_components = find_connected_components(mask)

    if number_of_components == 0:
        print(
            f"  No detections for cycle {record.get('cycle')} "
            f"pass {record.get('pass')}"
        )
        return []

    features = []

    for component_id in range(1, number_of_components + 1):
        component_mask = labels == component_id

        result = footprint_polygon_from_valid_mask(
            valid_mask=component_mask,
            latitude=latitude,
            longitude=longitude,
        )

        if result is None:
            continue

        coords, footprint_properties = result

        # Best-effort match for old files only.
        box_record = get_box_record(record, component_id - 1)

        feature_id = (
            f"iw_yolo_cycle_{record.get('cycle')}"
            f"_pass_{record.get('pass')}"
            f"_component_{component_id:03d}"
        )

        feature = {
            "type": "Feature",
            "properties": {
                "id": feature_id,
                "model": "Bay of Bengal Internal Wave YOLO Detector",
                "model_id": "iw_yolo",
                "cycle": record.get("cycle"),
                "pass": record.get("pass"),
                "source_item_id": record.get("item_id"),
                "mask_netcdf": str(Path(record["mask_netcdf"]).expanduser().resolve()),
                "detection_threshold": get_detection_threshold(record),
                "confidence_value": box_record.get("confidence_value"),
                "component_id": component_id,
                "box_index": box_record.get("box_index"),
                "class_id": box_record.get("class_id"),
                "class_name": box_record.get("class_name"),
                "filter_type": record.get("filter_type"),
                "geometry_method": "combined_mask_connected_component_fallback",
                **footprint_properties,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        }

        features.append(feature)

    return features


def mask_netcdf_to_features(record: dict) -> list[dict]:
    """
    Convert one mask NetCDF into GeoJSON detection polygons.
    """
    mask_path = Path(record["mask_netcdf"]).expanduser().resolve()

    if not mask_path.exists():
        print(f"Warning: mask file not found, skipping: {mask_path}")
        return []

    print(f"Reading mask: {mask_path}")

    ds = xr.open_dataset(mask_path)

    try:
        latitude = get_2d_array(ds, "latitude")
        longitude = get_2d_array(ds, "longitude")

        # Preferred new behavior: one polygon per YOLO box.
        per_box_features = features_from_per_box_masks(
            record=record,
            ds=ds,
            latitude=latitude,
            longitude=longitude,
        )

        if per_box_features is not None:
            print(
                f"  Created {len(per_box_features)} per-YOLO-box "
                f"GeoJSON feature(s) for cycle {record.get('cycle')} "
                f"pass {record.get('pass')}"
            )
            return per_box_features

        # Fallback old behavior.
        fallback_features = features_from_combined_mask_fallback(
            record=record,
            ds=ds,
            latitude=latitude,
            longitude=longitude,
        )

        print(
            f"  Created {len(fallback_features)} fallback component "
            f"GeoJSON feature(s) for cycle {record.get('cycle')} "
            f"pass {record.get('pass')}"
        )

        return fallback_features

    finally:
        ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-manifest",
        default=str(DEFAULT_INPUT_MANIFEST),
        help="Manifest from the internal-wave YOLO step 02 pipeline.",
    )

    parser.add_argument(
        "--output-geojson",
        default=str(DEFAULT_OUTPUT_GEOJSON),
        help="Combined detection GeoJSON output path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_manifest = Path(args.input_manifest).expanduser().resolve()
    output_geojson = Path(args.output_geojson).expanduser().resolve()

    records = load_manifest(input_manifest)

    all_features = []

    for record in records:
        features = mask_netcdf_to_features(record)
        all_features.extend(features)

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_geojson.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("GeoJSON conversion complete.")
    print(f"Input records: {len(records)}")
    print(f"Total detection polygons: {len(all_features)}")
    print(f"Wrote: {output_geojson}")
    print("=" * 80)


if __name__ == "__main__":
    main()