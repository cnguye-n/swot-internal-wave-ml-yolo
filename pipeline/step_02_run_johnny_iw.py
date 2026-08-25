#!/usr/bin/env python3
"""
step_02_run_johnny_iw.py

Purpose:
    Run Johnny's internal-wave detector on selected SWOT NetCDF files.

This version saves:
    1. A combined mask:
        internal_wave_bbox_mask
        1 = inside any YOLO box

    2. A per-YOLO-box mask stack:
        internal_wave_box_masks
        shape = detection x num_lines x num_pixels

    3. Real YOLO confidence values:
        yolo_box_confidence

Why:
    The combined mask is useful for quick display, but it loses the identity
    of each individual YOLO box. The per-box mask stack lets step 03 create
    one GeoJSON polygon per YOLO box and attach the correct confidence value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# ------------------------------------------------------------
# File locations
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))

from swot_internal_wave_detector import SWOTInternalWaveDetector

DEFAULT_INPUT_JSON = ( REPO_ROOT / "pipeline" / "model_outputs" / "selected_items_test.json")
DEFAULT_OUTPUT_DIR = ( REPO_ROOT / "pipeline" / "model_outputs")
DEFAULT_MODEL_PATH = REPO_ROOT / "last.pt"

# ------------------------------------------------------------
# Johnny's model settings
# ------------------------------------------------------------

JOHNNY_LAT_MIN = 3.0
JOHNNY_LAT_MAX = 15.5
JOHNNY_FILTER_TYPE = "rolling"
JOHNNY_CMAP = "magma"

# This is the YOLO detection threshold, not the per-box confidence.
DEFAULT_JOHNNY_CONFIDENCE_THRESHOLD = 0.4

# Combined binary mask.
MASK_VARIABLE_NAME = "internal_wave_bbox_mask"

# Per-box binary mask stack.
YOLO_BOX_MASK_VARIABLE_NAME = "internal_wave_box_masks"


def load_selected_items(input_json: Path) -> list[dict]:
    """
    Read selected NetCDF records from step 01.
    """
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    records = json.loads(input_json.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list.")

    return records


def make_output_prefix(record: dict, confidence_threshold: float) -> str:
    """
    Create a clean output filename prefix.

    Example:
        johnny_iw_cycle_050_pass_008
    """
    cycle = record.get("cycle", "unknown")
    pass_number = record.get("pass", "unknown")

    threshold_token = f"{round(float(confidence_threshold) * 100):03d}"

    return (
        f"johnny_iw_cycle_{cycle}_pass_{pass_number}_thr_{threshold_token}"
    )


def get_detection_count(result) -> int:
    """
    Count YOLO detections safely.
    """
    if result is None:
        return 0

    if result.boxes is None:
        return 0

    return len(result.boxes)


def to_python_list(value) -> list:
    """
    Convert Torch/Numpy/list values into a normal Python list.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu()

    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


def yolo_box_to_original_mask(
    box_xyxy: list[float],
    transform_info: dict,
) -> np.ndarray:
    """
    Convert one YOLO box from transformed image coordinates back to the
    original clipped SWOT pixel grid.

    This mirrors Johnny's yolo_boxes_to_original_mask(), but it does it
    for one box at a time so we can keep box identity and confidence.
    """
    transformed_shape = transform_info["transformed_shape"]
    original_shape = transform_info["original_shape"]
    was_transposed = transform_info["was_transposed"]

    mask_transformed = np.zeros(transformed_shape, dtype=np.uint8)

    height, width = transformed_shape

    x1, y1, x2, y2 = box_xyxy

    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 >= x1 and y2 >= y1:
        mask_transformed[y1 : y2 + 1, x1 : x2 + 1] = 1

    # Undo the left-right flip used to create the YOLO image.
    mask_unflipped = np.fliplr(mask_transformed)

    # Undo transpose if the SWOT swath was transposed for YOLO.
    if was_transposed:
        mask_original = mask_unflipped.T
    else:
        mask_original = mask_unflipped

    if mask_original.shape != original_shape:
        raise ValueError(
            f"Single-box mask shape mismatch. Expected {original_shape}, "
            f"got {mask_original.shape}"
        )

    return mask_original.astype(np.uint8)


def get_base_data_array(ds: xr.Dataset, base_var: str) -> xr.DataArray:
    """
    Get the 2D base SWOT DataArray used for dimensions/coordinates.
    """
    ds_plot = ds.isel(file=0) if "file" in ds.dims else ds

    if base_var not in ds_plot:
        raise KeyError(f"Missing base variable: {base_var}")

    base_da = ds_plot[base_var].squeeze()

    if base_da.ndim != 2:
        raise ValueError(
            f"Expected 2D base variable after squeeze, got dims {base_da.dims}"
        )

    return base_da


def get_yolo_box_records_and_masks(
    result,
    transform_info: dict,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Extract real YOLO confidence values and create one original-grid mask
    for each YOLO box.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return [], []

    names = getattr(result, "names", {}) or {}

    box_records = []
    box_masks = []

    for box_index, box in enumerate(result.boxes, start=1):
        class_id = int(to_python_list(box.cls)[0])
        confidence_value = float(to_python_list(box.conf)[0])
        box_xyxy = [float(value) for value in to_python_list(box.xyxy[0])]

        box_mask = yolo_box_to_original_mask(
            box_xyxy=box_xyxy,
            transform_info=transform_info,
        )

        valid = box_mask.astype(bool)

        if np.any(valid):
            rows, cols = np.where(valid)

            original_grid_bbox = {
                "min_row": int(np.min(rows)),
                "max_row": int(np.max(rows)),
                "min_col": int(np.min(cols)),
                "max_col": int(np.max(cols)),
            }
        else:
            original_grid_bbox = {
                "min_row": None,
                "max_row": None,
                "min_col": None,
                "max_col": None,
            }

        box_record = {
            "box_index": box_index,
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "confidence_value": confidence_value,
            "image_xyxy": box_xyxy,
            "original_grid_bbox": original_grid_bbox,
            "mask_pixel_count": int(np.count_nonzero(valid)),
        }

        box_records.append(box_record)
        box_masks.append(box_mask)

    return box_records, box_masks


def add_yolo_box_masks_to_dataset(
    ds_with_outputs: xr.Dataset,
    base_da: xr.DataArray,
    box_masks: list[np.ndarray],
    box_records: list[dict],
) -> xr.Dataset:
    """
    Save one mask per YOLO box into the NetCDF.

    Output variable:
        internal_wave_box_masks[detection, num_lines, num_pixels]
    """
    if len(box_masks) == 0:
        ds_with_outputs.attrs["yolo_box_mask_variable"] = YOLO_BOX_MASK_VARIABLE_NAME
        ds_with_outputs.attrs["yolo_box_mask_count"] = 0
        return ds_with_outputs

    mask_stack = np.stack(box_masks, axis=0).astype(np.uint8)

    expected_shape = (len(box_masks),) + tuple(base_da.shape)

    if mask_stack.shape != expected_shape:
        raise ValueError(
            f"Per-box mask stack shape mismatch. Expected {expected_shape}, "
            f"got {mask_stack.shape}"
        )

    detection_numbers = np.arange(1, len(box_masks) + 1, dtype=np.int32)

    coords = {
        "detection": detection_numbers,
    }

    for dim in base_da.dims:
        if dim in base_da.coords:
            coords[dim] = base_da.coords[dim]

    ds_with_outputs[YOLO_BOX_MASK_VARIABLE_NAME] = xr.DataArray(
        mask_stack,
        dims=("detection",) + tuple(base_da.dims),
        coords=coords,
        attrs={
            "description": "One binary mask per YOLO detection box.",
            "mask_values": "1 = inside this YOLO box, 0 = outside this YOLO box",
            "note": (
                "This preserves individual YOLO box identity. "
                "The combined mask internal_wave_bbox_mask loses box identity."
            ),
        },
    )

    ds_with_outputs["yolo_box_confidence"] = xr.DataArray(
        np.array(
            [box["confidence_value"] for box in box_records],
            dtype=np.float32,
        ),
        dims=("detection",),
        coords={"detection": detection_numbers},
        attrs={
            "description": "Actual YOLO confidence value for each detection box.",
            "note": "This is not the threshold. The threshold is saved separately.",
        },
    )

    ds_with_outputs["yolo_box_class_id"] = xr.DataArray(
        np.array(
            [box["class_id"] for box in box_records],
            dtype=np.int32,
        ),
        dims=("detection",),
        coords={"detection": detection_numbers},
        attrs={
            "description": "YOLO class id for each detection box.",
        },
    )

    ds_with_outputs.attrs["yolo_box_mask_variable"] = YOLO_BOX_MASK_VARIABLE_NAME
    ds_with_outputs.attrs["yolo_box_mask_count"] = len(box_masks)

    return ds_with_outputs


def run_johnny_on_one_record(
    detector: SWOTInternalWaveDetector,
    record: dict,
    output_dir: Path,
    confidence_threshold: float,
) -> dict:
    """
    Run Johnny's detector on one NetCDF file and save:
        - combined mask
        - per-box masks
        - real YOLO confidence values
    """
    item_id = record["item_id"]
    cycle = record.get("cycle", "unknown")
    pass_number = record.get("pass", "unknown")
    netcdf_path = Path(record["netcdf_path"]).expanduser().resolve()

    if not netcdf_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {netcdf_path}")

    output_prefix = make_output_prefix(record, confidence_threshold)

    mask_output_path = output_dir / f"{output_prefix}_mask.nc"
    summary_output_path = output_dir / f"{output_prefix}_summary.txt"
    preview_output_path = output_dir / f"{output_prefix}_boxed.jpg"

    print()
    print("=" * 80)
    print("Running Johnny internal-wave detector")
    print(f"Item:      {item_id}")
    print(f"Cycle:     {cycle}")
    print(f"Pass:      {pass_number}")
    print(f"NetCDF:    {netcdf_path}")
    print(f"Threshold: {confidence_threshold}")
    print("=" * 80)

    clipped_ds = detector.clip_pass_by_lat_only_keep_width(netcdf_path)

    try:
        # We run the lower-level steps instead of detect_yolo_on_l3_file()
        # because we need transform_info to make one mask per YOLO box.
        transformed_arr, transform_info, filtered_arr = detector.transform_dataset_for_yolo(
            clipped_ds,
            filter_type=JOHNNY_FILTER_TYPE,
            return_filtered=True,
        )

        pil_image = detector.array_to_pil_image(
            transformed_arr,
            cmap=JOHNNY_CMAP,
        )

        result = detector.run_yolo_detection(
            pil_image,
            conf=confidence_threshold,
        )

        yolo_text = detector.yolo_results_to_text(result)

        boxed_image = detector.draw_yolo_bounding_boxes(
            pil_image,
            result,
        )

        # Combined mask: this is the old behavior.
        combined_mask = detector.yolo_boxes_to_original_mask(
            result,
            transform_info,
        )

        ds_with_outputs = detector.add_yolo_bbox_mask_to_dataset(
            clipped_ds=clipped_ds,
            mask=combined_mask,
            mask_var_name=MASK_VARIABLE_NAME,
        )

        # Save the grid-aligned high-pass field used to create the YOLO image.
        # This lets the persistent result NetCDF contain both the analysis
        # input and the model masks.
        filtered_variable = detector.make_filtered_var_name(
            JOHNNY_FILTER_TYPE
        )

        ds_with_outputs = detector.add_filtered_array_to_dataset(
            clipped_ds=ds_with_outputs,
            filtered_arr=filtered_arr,
            filter_type=JOHNNY_FILTER_TYPE,
            filtered_var_name=filtered_variable,
        )

        # New behavior: one mask per YOLO box.
        yolo_boxes, box_masks = get_yolo_box_records_and_masks(
            result=result,
            transform_info=transform_info,
        )

        base_da = get_base_data_array(
            ds_with_outputs,
            detector.base_var,
        )

        ds_with_outputs = add_yolo_box_masks_to_dataset(
            ds_with_outputs=ds_with_outputs,
            base_da=base_da,
            box_masks=box_masks,
            box_records=yolo_boxes,
        )

        ds_with_outputs.attrs["yolo_detection_threshold"] = float(
            confidence_threshold
        )
        ds_with_outputs.attrs["yolo_filter_type"] = str(JOHNNY_FILTER_TYPE)
        ds_with_outputs.attrs["yolo_cmap"] = str(JOHNNY_CMAP)

        detection_count = len(yolo_boxes)

        ds_with_outputs.to_netcdf(mask_output_path)

    finally:
        clipped_ds.close()

    summary_output_path.write_text(yolo_text, encoding="utf-8")
    boxed_image.save(preview_output_path)

    ds_with_outputs.close()

    confidence_values = [
        box["confidence_value"]
        for box in yolo_boxes
        if box.get("confidence_value") is not None
    ]

    print(yolo_text)
    print()
    print(f"Detection count: {detection_count}")
    print(f"Mask NetCDF:     {mask_output_path}")
    print(f"Summary text:    {summary_output_path}")
    print(f"Boxed preview:   {preview_output_path}")

    return {
        "item_id": item_id,
        "cycle": cycle,
        "pass": pass_number,
        "netcdf_path": str(netcdf_path),
        "mask_netcdf": str(mask_output_path),
        "summary_text": str(summary_output_path),
        "boxed_preview": str(preview_output_path),
        "mask_variable": MASK_VARIABLE_NAME,
        "box_mask_variable": YOLO_BOX_MASK_VARIABLE_NAME,
        "filtered_variable": filtered_variable,
        "detection_count": detection_count,

        # Keep this legacy field so old code does not break.
        # But this value is a threshold, not actual confidence.
        "confidence": confidence_threshold,

        # Use these new fields going forward.
        "detection_threshold": confidence_threshold,
        "confidence_values": confidence_values,
        "max_confidence_value": (
            max(confidence_values) if confidence_values else None
        ),
        "mean_confidence_value": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
        "yolo_boxes": yolo_boxes,
        "filter_type": JOHNNY_FILTER_TYPE,
        "cmap": JOHNNY_CMAP,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT_JSON),
        help="JSON from step_01_read_stac.py.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to Johnny's YOLO .pt model.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for mask outputs.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_JOHNNY_CONFIDENCE_THRESHOLD,
        help="YOLO detection threshold between 0 and 1.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")

    if not model_path.exists():
        raise FileNotFoundError(f"Johnny model not found: {model_path}")

    selected_records = load_selected_items(input_json)

    if len(selected_records) == 0:
        raise ValueError("No selected records found.")

    print(f"Input JSON: {input_json}")
    print(f"Model:      {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Records:    {len(selected_records)}")
    print(f"Threshold:  {args.confidence_threshold}")

    detector = SWOTInternalWaveDetector(
        lat_min=JOHNNY_LAT_MIN,
        lat_max=JOHNNY_LAT_MAX,
        model_path=model_path,
    )

    output_records = []

    for record in selected_records:
        output_record = run_johnny_on_one_record(
            detector=detector,
            record=record,
            output_dir=output_dir,
            confidence_threshold=args.confidence_threshold,
        )

        output_records.append(output_record)

    manifest_path = output_dir / "johnny_iw_mask_outputs.json"
    manifest_path.write_text(json.dumps(output_records, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("Done.")
    print(f"Wrote manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()