#!/usr/bin/env python3
"""
step_04_write_outputs.py

Purpose:
    Prepare Johnny detection outputs for display.

This step:
    - Reads the GeoJSON from step 03
    - Writes a stable latest GeoJSON file:
        pipeline/model_outputs/latest_detections.geojson
    - Writes a small latest manifest:
        pipeline/model_outputs/latest_detections_manifest.json

Why this updated version exists:
    Step 03 now saves:
        - model
        - cycle
        - pass
        - detection_threshold
        - confidence_value

    Step 04 does not change those GeoJSON properties.
    It just copies the GeoJSON and writes a useful manifest summary so we can
    quickly verify whether confidence values were saved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_GEOJSON = (
    PLUGIN_ROOT / "pipeline" / "model_outputs" / "johnny_iw_detections.geojson"
)

DEFAULT_OUTPUT_DIR = PLUGIN_ROOT / "pipeline" / "model_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-geojson",
        default=str(DEFAULT_INPUT_GEOJSON),
        help="GeoJSON from step_03_yolo_to_geojson.py.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for stable latest output files.",
    )

    return parser.parse_args()


def get_feature_properties(feature: dict) -> dict:
    """
    Safely get GeoJSON feature properties.
    """
    properties = feature.get("properties", {})

    if not isinstance(properties, dict):
        return {}

    return properties


def get_numeric_property(properties: dict, key: str) -> float | None:
    """
    Read a numeric property from GeoJSON properties.

    Returns None if the value is missing or cannot be converted to a float.
    """
    value = properties.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_manifest(
    input_geojson: Path,
    latest_geojson: Path,
    geojson: dict,
) -> dict:
    """
    Build a small manifest that summarizes the latest detection GeoJSON.

    This helps the frontend/backend know:
        - where the stable latest GeoJSON is
        - how many detection polygons exist
        - whether real confidence values were saved
        - what detection threshold was used
    """
    features = geojson.get("features", [])

    if not isinstance(features, list):
        raise ValueError("GeoJSON 'features' must be a list.")

    feature_count = len(features)

    confidence_values = []
    detection_thresholds = set()
    cycles = set()
    passes = set()
    models = set()
    geometry_methods = set()

    for feature in features:
        properties = get_feature_properties(feature)

        confidence_value = get_numeric_property(properties, "confidence_value")
        detection_threshold = get_numeric_property(
            properties,
            "detection_threshold",
        )

        if confidence_value is not None:
            confidence_values.append(confidence_value)

        if detection_threshold is not None:
            detection_thresholds.add(detection_threshold)

        if properties.get("cycle") is not None:
            cycles.add(str(properties.get("cycle")))

        if properties.get("pass") is not None:
            passes.add(str(properties.get("pass")))

        if properties.get("model") is not None:
            models.add(str(properties.get("model")))

        if properties.get("geometry_method") is not None:
            geometry_methods.add(str(properties.get("geometry_method")))

    manifest = {
        "detections_geojson": str(latest_geojson),
        "feature_count": feature_count,
        "source_geojson": str(input_geojson),

        # New confidence summary.
        "has_confidence_values": len(confidence_values) > 0,
        "confidence_value_count": len(confidence_values),
        "min_confidence_value": (
            min(confidence_values) if confidence_values else None
        ),
        "max_confidence_value": (
            max(confidence_values) if confidence_values else None
        ),
        "mean_confidence_value": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),

        # Usually this should be [0.4].
        "detection_thresholds": sorted(detection_thresholds),

        # Helpful display/debug metadata.
        "cycles": sorted(cycles),
        "passes": sorted(passes),
        "models": sorted(models),
        "geometry_methods": sorted(geometry_methods),

        "note": (
            "This is the latest internal-wave detection GeoJSON. "
            "When generated with the updated step 02 and step 03 pipeline, "
            "each feature should represent one YOLO box and include both "
            "detection_threshold and confidence_value."
        ),
    }

    return manifest


def main() -> None:
    args = parse_args()

    input_geojson = Path(args.input_geojson).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_geojson.exists():
        raise FileNotFoundError(f"Input GeoJSON not found: {input_geojson}")

    output_dir.mkdir(parents=True, exist_ok=True)

    latest_geojson = output_dir / "latest_detections.geojson"
    latest_manifest = output_dir / "latest_detections_manifest.json"

    geojson = json.loads(input_geojson.read_text(encoding="utf-8"))

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Input GeoJSON must be a FeatureCollection.")

    # Copy the full GeoJSON unchanged.
    # This preserves model, cycle, pass, detection_threshold, confidence_value, etc.
    shutil.copyfile(input_geojson, latest_geojson)

    manifest = build_manifest(
        input_geojson=input_geojson,
        latest_geojson=latest_geojson,
        geojson=geojson,
    )

    latest_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("Display outputs prepared.")
    print(f"Detection boxes:         {manifest['feature_count']}")
    print(f"Has confidence values:   {manifest['has_confidence_values']}")
    print(f"Confidence value count:  {manifest['confidence_value_count']}")
    print(f"Detection thresholds:    {manifest['detection_thresholds']}")
    print(f"Latest GeoJSON:          {latest_geojson}")
    print(f"Latest manifest:         {latest_manifest}")
    print("=" * 80)


if __name__ == "__main__":
    main()