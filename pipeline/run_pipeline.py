#!/usr/bin/env python3
"""
Run one or more source STAC Items through Johnny's local pipeline.

The backend normally calls this file once per source Item so the UI can display
one pass at a time and save one deterministic STAC result Item per pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PLUGIN_ROOT / "pipeline"

#default location for generated model outputs when the user didn't configure a custom output directory yet
DEFAULT_OUTPUT_DIR = (
    PIPELINE_DIR / "model_outputs"
)

#allow each MMGIS installation to choose where model outputs are stored --> avoids hardcoding machine-specific path in plugin
#Example: SWOT_DETECTOR_OUTPUT_DIR=/data/swot/model_outputs
#if environment variable is not set, fall back to plugin's local pipeline/model_outpputs directory
OUTPUT_DIR = Path(
    os.environ.get(
        "SWOT_DETECTOR_OUTPUT_DIR",
        DEFAULT_OUTPUT_DIR,
    )
).expanduser().resolve()

def run_command(command: list[str]) -> None:
    print()
    print("=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=PLUGIN_ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default="iw_yolo",
        help="Registered model ID.",
    )

    parser.add_argument(
        "--item-ids",
        nargs="+",
        required=True,
        help="Selected source STAC Item IDs.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.4,
        help="YOLO detection threshold.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Run-specific output folder.",
    )

    parser.add_argument(
        "--result-json",
        default=None,
        help="Path for the final machine-readable pipeline result JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_id != "iw_yolo":
        raise ValueError(
            f"Only iw_yolo is wired right now. Got: {args.model_id}"
        )

    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_items_json = output_dir / "selected_items.json"
    mask_manifest_json = output_dir / "iw_yolo_mask_outputs.json"
    detections_geojson = output_dir / "detections.geojson"
    latest_manifest_json = output_dir / "latest_detections_manifest.json"

    result_json = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json
        else output_dir / "pipeline_result.json"
    )

    run_command(
        [
            sys.executable,
            str(PIPELINE_DIR / "step_01_read_stac.py"),
            "--item-ids",
            *args.item_ids,
            "--output-json",
            str(selected_items_json),
        ]
    )

    run_command(
        [
            sys.executable,
            str(PIPELINE_DIR / "step_02_run_iw.py"),
            "--input-json",
            str(selected_items_json),
            "--output-dir",
            str(output_dir),
            "--confidence-threshold",
            str(args.confidence_threshold),
        ]
    )

    run_command(
        [
            sys.executable,
            str(PIPELINE_DIR / "step_03_yolo_to_geojson.py"),
            "--input-manifest",
            str(mask_manifest_json),
            "--output-geojson",
            str(detections_geojson),
        ]
    )

    run_command(
        [
            sys.executable,
            str(PIPELINE_DIR / "step_04_write_outputs.py"),
            "--input-geojson",
            str(detections_geojson),
            "--output-dir",
            str(output_dir),
        ]
    )

    output_records = json.loads(
        mask_manifest_json.read_text(encoding="utf-8")
    )

    if not output_records:
        raise ValueError("Johnny pipeline did not produce an output record.")

    if len(output_records) != 1:
        raise ValueError(
            "The persistent model endpoint expects one source Item per pipeline run."
        )

    output_record = output_records[0]

    result = {
        "status": "completed",
        "model_id": args.model_id,
        "source_item_id": output_record["item_id"],
        "cycle": output_record.get("cycle"),
        "pass": output_record.get("pass"),
        "detection_threshold": args.confidence_threshold,
        "detection_count": output_record.get("detection_count", 0),
        "analysis_netcdf": output_record.get("mask_netcdf"),
        "detections_geojson": str(detections_geojson),
        "boxed_preview": output_record.get("boxed_preview"),
        "summary_text": output_record.get("summary_text"),
        "manifest_json": str(latest_manifest_json),
        "mask_manifest_json": str(mask_manifest_json),
        "filtered_variable": output_record.get("filtered_variable"),
        "mask_variable": output_record.get("mask_variable"),
        "box_mask_variable": output_record.get("box_mask_variable"),
    }

    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("Pipeline complete.")
    print(json.dumps(result, indent=2))
    print(f"Result JSON: {result_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
