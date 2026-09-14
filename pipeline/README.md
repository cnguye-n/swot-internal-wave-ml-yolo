# Internal Wave Detection Pipeline

This document explains how the internal-wave YOLO model (developed by Johnathan Aguilar)
actually processes a SWOT granule, from raw NetCDF to GeoJSON detections. It covers both:

- the detection algorithm itself (`swot_internal_wave_detector.py`, at the repo root)
- the 4-step CLI pipeline in this folder that drives it (`run_pipeline.py`)

The same algorithm is also invoked directly by the FastAPI service (`api.py`) without
going through the CLI steps — see [How the API service uses this pipeline](#how-the-api-service-uses-this-pipeline).

## The detection algorithm (`SWOTInternalWaveDetector`)

The detector class lives at the repo root in `swot_internal_wave_detector.py` and is
reused by every entry point (CLI pipeline and API). Given one SWOT L3 NetCDF file, it
runs the following stages:

1. **Clip by latitude** (`clip_pass_by_lat_only_keep_width`)
   Opens the NetCDF with `xarray` and keeps only the along-track lines whose latitude
   falls inside `[lat_min, lat_max]`. The default bounds used everywhere in this repo
   are `lat_min=3.0`, `lat_max=15.5`, i.e. the **Bay of Bengal**. The full cross-track
   swath width is preserved — only latitude is used to crop.

2. **Extract and quality-mask SSHA** (`prepare_base_array`)
   Reads `ssha_unfiltered` (the sea surface height anomaly variable) and
   `quality_flag`. Any pixel whose quality flag is not one of `0, 5, 30, 50` is set to
   `NaN`, and any SSHA value greater than 10 is treated as bad data and dropped.

3. **High-pass filter** (`apply_selected_filter`)
   Internal waves show up as short-wavelength ripples on top of a smooth background
   sea-surface signal, so the detector removes the smooth background before running
   YOLO. Four filter modes are implemented:
   - `normal` — no filtering, quality-masked SSHA as-is
   - `rolling` — original minus a NaN-aware rolling/uniform mean (**this is the mode
     used by the pipeline and the API**, `IW_YOLO_FILTER_TYPE = "rolling"`)
   - `gaussian` — original minus a NaN-aware Gaussian low-pass
   - `stepped` — original minus a stepped radial convolution (inner/middle/outer
     rings with different weights), tuned in km via `pixel_size_km`

4. **Orient for YOLO** (`orient_array_for_yolo`)
   YOLO expects a roughly upright, consistently-oriented image. The filtered array is
   transposed so the long dimension is vertical, then flipped left-right. This
   transform is recorded (`transform_info`) so bounding boxes can be mapped back to
   the original grid later.

5. **Render to an image** (`array_to_pil_image`)
   The filtered/oriented array is normalized using the 2.5th/97.5th percentile as
   `vmin`/`vmax` and colored with the `magma` matplotlib colormap. `NaN` pixels are
   rendered white. This RGB image is what actually gets fed into YOLO.

6. **Run YOLO inference** (`run_yolo_detection`)
   Loads the Ultralytics YOLO model from `last.pt` and runs `model.predict()` on the
   rendered image with a confidence threshold (`conf`). Each detection has a class,
   a confidence score, and an axis-aligned bounding box in image coordinates.

7. **Map boxes back to the original grid** (`yolo_boxes_to_original_mask`,
   `yolo_box_to_original_mask` in `step_02_run_iw.py`)
   Each YOLO box is un-flipped and un-transposed back into the original clipped
   `(num_lines, num_pixels)` grid, producing one binary mask per detection box
   (`internal_wave_box_masks`) plus a combined mask (`internal_wave_bbox_mask`).
   Per-box confidence, class id, and class name are kept alongside the masks
   (`yolo_box_confidence`, `yolo_box_class_id`).

8. **Persist to NetCDF**
   The clipped dataset is written back out with the filtered array, the combined
   mask, and the per-box mask stack added as new variables — so the output NetCDF
   contains both the original SWOT data and everything the model produced.

## The 4-step CLI pipeline

`run_pipeline.py` orchestrates four scripts as separate subprocesses. This is the
"standalone" way to run the model outside of the API (e.g. for batch runs or the older
MMGIS plugin backend integration):

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `step_01_read_stac.py` | Selected STAC item ids (e.g. `bay_of_bengal_swot_cycle_050_pass_008`) | `selected_items.json` — resolved NetCDF paths, fetched from the MMGIS plugin's STAC API (`MMGIS_API_BASE`, default `http://localhost:8888/api/swotDetectorApi`) |
| 2 | `step_02_run_iw.py` | `selected_items.json` | One mask NetCDF, one text summary, and one boxed-preview JPEG per record, plus a combined `iw_yolo_mask_outputs.json` manifest. Runs the full detector algorithm above. |
| 3 | `step_03_yolo_to_geojson.py` | `iw_yolo_mask_outputs.json` | `detections.geojson` — one Polygon feature per YOLO box (see [GeoJSON conversion](#geojson-conversion)) |
| 4 | `step_04_write_outputs.py` | `detections.geojson` | Stable "latest" copies (`latest_detections.geojson`, `latest_detections_manifest.json`) for the standalone/legacy MMGIS plugin backend to read. **Not used by the Docker API service.** |

Run it directly with:

```bash
python pipeline/run_pipeline.py \
  --item-ids bay_of_bengal_swot_cycle_050_pass_008 \
  --confidence-threshold 0.4 \
  --output-dir pipeline/model_outputs
```

Useful environment variables:
- `MMGIS_API_BASE` — base URL of the MMGIS plugin backend's STAC lookup API (step 1 only)
- `SWOT_DETECTOR_OUTPUT_DIR` — where `run_pipeline.py` writes outputs by default

### GeoJSON conversion

`step_03_yolo_to_geojson.py` prefers the per-box mask stack (`internal_wave_box_masks`)
so each YOLO detection becomes its own GeoJSON `Polygon`: it takes the lat/lon of every
valid pixel in that box's mask and wraps them in a convex hull (falling back to a
slanted quadrilateral from the mask's corner pixels if the hull fails). Each feature
carries the properties described in [`../model_output_schema.json`](../model_output_schema.json)
(`confidence_value`, `class_name`, `bbox_*`, `pixel_count`, etc.).

If a mask NetCDF was produced by an older run that only has the combined mask, it falls
back to connected-component labeling on `internal_wave_bbox_mask` instead — this loses
per-box identity if two boxes touch or overlap, which is why the per-box mask stack is
the preferred path.

## How the API service uses this pipeline

`api.py` does **not** shell out to the 4 CLI steps. Instead, on `/predict` it:
1. Resolves the requested granule directly from the mounted data volume (see the
   [main README](../README.md#docker-image-model-service) for the volume-mount contract).
2. Calls `run_iw_yolo_on_one_record()` from `step_02_run_iw.py` directly, in-process,
   writing its NetCDF/summary/preview outputs to a temporary directory that is deleted
   after the request completes.
3. Calls `mask_netcdf_to_features()` from `step_03_yolo_to_geojson.py` directly to get
   GeoJSON features.
4. Strips internal fields (like the temp `mask_netcdf` path) and returns a single JSON
   response containing the GeoJSON `FeatureCollection` plus request metadata.

Step 1 (STAC lookup) and step 4 (stable "latest" files) are skipped entirely — the API
request already carries the resolved `source_file`, and there's no need for on-disk
"latest" copies for a stateless HTTP service.

## Files not part of the live service

`old/build_cycle_stac_catalog.py` is a legacy, metadata-only script for building a
cycle-based STAC catalog from an existing folder of raw/processed SWOT files. It predates
the Docker/FastAPI service and is not imported or run by anything in `pipeline/` or `api.py`.
