#!/usr/bin/env python3
"""
Plot native SWOT pixels, Johnny's mask, and Step 3 GeoJSON polygons together.

This checks whether:
1. internal_wave_bbox_mask is aligned with the SWOT swath
2. Step 3 GeoJSON polygons are aligned with the mask
3. MMGIS is possibly loading/displaying something different
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def get_2d_array(ds: xr.Dataset, variable_name: str) -> np.ndarray:
    if variable_name not in ds:
        raise KeyError(f"Missing variable in mask NetCDF: {variable_name}")

    arr = np.squeeze(ds[variable_name].values)

    if arr.ndim != 2:
        raise ValueError(
            f"{variable_name} should be 2D after squeeze, got {arr.shape}"
        )

    return arr


def choose_background_variable(ds: xr.Dataset) -> str | None:
    """
    Prefer the science variable if it exists.
    Otherwise the plot still works with only lat/lon + mask.
    """
    candidates = [
        "ssha_unfiltered",
        "ssha_filtered",
        "filtered_ssha",
        "internal_wave_bbox_mask",
    ]

    for name in candidates:
        if name in ds:
            return name

    return None


def iter_polygon_rings(geometry: dict):
    """
    Yield rings from Polygon or MultiPolygon GeoJSON geometry.
    """
    geometry_type = geometry.get("type")

    if geometry_type == "Polygon":
        for ring in geometry["coordinates"]:
            yield ring

    elif geometry_type == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            for ring in polygon:
                yield ring


def load_matching_features(
    geojson_path: Path,
    cycle: str,
    pass_number: str,
) -> list[dict]:
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))

    features = []

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})

        if str(props.get("cycle")) == cycle and str(props.get("pass")) == pass_number:
            features.append(feature)

    return features


def plot_alignment(
    mask_nc: Path,
    geojson_path: Path,
    output_png: Path,
    cycle: str,
    pass_number: str,
    mask_variable: str,
    display_min: float,
    display_max: float,
) -> None:
    print(f"Reading mask NetCDF: {mask_nc}")
    print(f"Reading GeoJSON:     {geojson_path}")

    ds = xr.open_dataset(mask_nc)

    try:
        longitude = get_2d_array(ds, "longitude")
        latitude = get_2d_array(ds, "latitude")
        mask = get_2d_array(ds, mask_variable)

        mask = np.nan_to_num(mask, nan=0).astype(bool)

        background_name = choose_background_variable(ds)

        if background_name is not None:
            background = get_2d_array(ds, background_name).astype(float)
        else:
            background = np.full_like(longitude, np.nan, dtype=float)

    finally:
        ds.close()

    features = load_matching_features(
        geojson_path=geojson_path,
        cycle=cycle,
        pass_number=pass_number,
    )

    print(f"Matching GeoJSON features: {len(features)}")
    print(f"Mask pixels: {int(np.count_nonzero(mask)):,}")

    finite_background = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(background)
    )

    finite_mask = (
        mask
        & np.isfinite(longitude)
        & np.isfinite(latitude)
    )

    figure, axis = plt.subplots(figsize=(9, 11), dpi=180)

    # Native SWOT pixels as background.
    if np.any(finite_background):
        image = axis.scatter(
            longitude[finite_background],
            latitude[finite_background],
            c=background[finite_background],
            s=3,
            cmap="RdBu_r",
            vmin=display_min,
            vmax=display_max,
            linewidths=0,
            rasterized=True,
            label=f"Native {background_name}",
        )

        figure.colorbar(
            image,
            ax=axis,
            label=background_name,
            shrink=0.8,
        )
    else:
        # Fallback: show all finite swath pixels in light gray.
        finite_swath = np.isfinite(longitude) & np.isfinite(latitude)

        axis.scatter(
            longitude[finite_swath],
            latitude[finite_swath],
            s=3,
            c="lightgray",
            linewidths=0,
            rasterized=True,
            label="Native swath pixels",
        )

    # Johnny mask pixels.
    if np.any(finite_mask):
        axis.scatter(
            longitude[finite_mask],
            latitude[finite_mask],
            s=7,
            c="black",
            linewidths=0,
            alpha=0.55,
            label="Johnny mask pixels",
        )

    # Step 3 GeoJSON polygon outlines.
    for feature_index, feature in enumerate(features, start=1):
        props = feature.get("properties", {})
        component_id = props.get("component_id", feature_index)

        for ring in iter_polygon_rings(feature["geometry"]):
            ring_array = np.asarray(ring, dtype=float)

            axis.plot(
                ring_array[:, 0],
                ring_array[:, 1],
                linewidth=2.0,
                label=f"GeoJSON component {component_id}",
            )

            # Mark polygon vertices so we can see whether Step 3 used real mask pixels.
            axis.scatter(
                ring_array[:, 0],
                ring_array[:, 1],
                s=20,
                marker="x",
            )

    axis.set_title(
        f"Step 3 Alignment Diagnostic\n"
        f"Cycle {cycle}, Pass {pass_number}"
    )

    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)

    # Zoom to the finite swath/mask area.
    finite_any = np.isfinite(longitude) & np.isfinite(latitude)

    if np.any(finite_any):
        lon_min = float(np.nanmin(longitude[finite_any]))
        lon_max = float(np.nanmax(longitude[finite_any]))
        lat_min = float(np.nanmin(latitude[finite_any]))
        lat_max = float(np.nanmax(latitude[finite_any]))

        lon_pad = max(0.05, (lon_max - lon_min) * 0.05)
        lat_pad = max(0.05, (lat_max - lat_min) * 0.05)

        axis.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        axis.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_png,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print(f"Wrote diagnostic plot: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask-nc",
        default="pipeline/model_outputs/johnny_iw_cycle_003_pass_439_mask.nc",
        help="Johnny mask NetCDF from Step 2.",
    )

    parser.add_argument(
        "--geojson",
        default="pipeline/model_outputs/latest_detections.geojson",
        help="GeoJSON from Step 3 or Step 4.",
    )

    parser.add_argument(
        "--output",
        default="pipeline/model_outputs/diagnostics/pass_439_step_03_alignment.png",
        help="Output PNG path.",
    )

    parser.add_argument("--cycle", default="003")
    parser.add_argument("--pass-number", default="439")
    parser.add_argument("--mask-variable", default="internal_wave_bbox_mask")
    parser.add_argument("--display-min", type=float, default=-0.2)
    parser.add_argument("--display-max", type=float, default=0.2)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_alignment(
        mask_nc=Path(args.mask_nc).expanduser().resolve(),
        geojson_path=Path(args.geojson).expanduser().resolve(),
        output_png=Path(args.output).expanduser().resolve(),
        cycle=args.cycle,
        pass_number=args.pass_number,
        mask_variable=args.mask_variable,
        display_min=args.display_min,
        display_max=args.display_max,
    )


if __name__ == "__main__":
    main()
    