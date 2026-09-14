# swot-internal-wave-ml-yolo

YOLO-based internal wave detection model for SWOT satellite data, developed by
**Johnathan Aguilar**, packaged as a Docker/FastAPI model service for
[MMGIS](https://github.com/NASA-AMMOS/MMGIS) integration.

The model detects internal waves in [SWOT](https://swot.jpl.nasa.gov/) (Surface Water
and Ocean Topography) L3 sea surface height data over the **Bay of Bengal** using a
YOLO object-detection model trained on filtered SSHA imagery.

| | |
|---|---|
| **Region** | Bay of Bengal (latitude 3.0°–15.5°) |
| **Input** | SWOT L3 SSH NetCDF granule (`.nc`) |
| **Output** | GeoJSON `FeatureCollection` of detected internal waves, one polygon per detection, each with a confidence score and supporting metadata |
| **Model** | YOLO (Ultralytics), weights in [`last.pt`](last.pt) |

For an in-depth explanation of the detection algorithm and the standalone CLI pipeline,
see [`pipeline/README.md`](pipeline/README.md).

## Repository layout

```
swot_internal_wave_detector.py   # Core detection algorithm (filtering + YOLO)
api.py                           # FastAPI service used by the Docker image
model_output_schema.json         # Contract describing every GeoJSON property the model returns
model_ui_for_mmgis.json          # Display/formatting hints for the MMGIS plugin UI
last.pt                          # Trained YOLO model weights
Dockerfile                       # Builds the model service image
environment.yml                  # Conda/micromamba environment definition
pipeline/                        # Standalone 4-step CLI pipeline + in-depth docs
tests/                           # pytest suite (unit + integration)
test-data/                       # Sample SWOT granules used by the integration test
old/                             # Legacy, unused STAC-catalog-building script
```

## Output: what the model returns

Every detection is returned as a GeoJSON `Polygon` feature. The full list of properties
is defined in [`model_output_schema.json`](model_output_schema.json) — this is the
contract MMGIS reads to know what fields exist and how to label/type them:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique id for the detection, e.g. `iw_yolo_cycle_050_pass_230_box_001` |
| `confidence_value` | number | YOLO's confidence score for this detection (0–1) |
| `class_id` / `class_name` | integer / string | YOLO class of the detection |
| `box_index` | integer | Index of this detection within the source pass |
| `pixel_count` | integer | Number of valid pixels inside the detection's mask |
| `bbox_west` / `bbox_south` / `bbox_east` / `bbox_north` | number (degrees) | Lat/lon bounding box of the detection |
| `detection_threshold` | number | Confidence threshold used to keep/reject YOLO predictions for this run |
| `filter_type` | string | High-pass filter applied before detection (`rolling` by default) |

`model_output_schema.json` is served by the API at `GET /output-schema` so MMGIS can
introspect the contract at runtime instead of hardcoding it.

## `model_ui_for_mmgis.json`: display customization

This file is **not** part of the data contract — it only tells the
[swot-ml-detector-plugin](https://github.com/cnguye-n/swot-ml-detector-plugin) UI how to
*present* fields from the schema above: which fields to show in the run summary, which
to show per-feature, how many decimals to round to, and what units to display. For
example, it tells the plugin to show `confidence_value` rounded to 3 decimals in both
the feature popup and the feature table, and to show `detection_count` /
`processing_seconds` in the run summary. It's served by the API at `GET /metadata`.

If you want to change how results are displayed in MMGIS (labels, rounding, which
fields show up where) without changing what the model actually computes, edit this file.

## MMGIS integration

[MMGIS](https://github.com/NASA-AMMOS/MMGIS) is NASA-AMMOS's multimission web GIS
platform. This repo does not embed any MMGIS or plugin code — it is a **standalone
model service** that the
[swot-ml-detector-plugin](https://github.com/cnguye-n/swot-ml-detector-plugin) MMGIS
plugin calls over HTTP. The plugin is responsible for:
- letting a user pick a SWOT granule/pass in MMGIS
- calling this service's `/predict` endpoint with the granule reference
- rendering the returned GeoJSON on the map, styled using `model_ui_for_mmgis.json`

This service and the plugin communicate purely over the HTTP API described below —
they can be deployed, versioned, and scaled independently.

## API contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service/model liveness check |
| `GET` | `/metadata` | Returns `model_ui_for_mmgis.json` (display config) |
| `GET` | `/output-schema` | Returns `model_output_schema.json` (data contract) |
| `POST` | `/predict` | Runs detection on one granule and returns GeoJSON |

`POST /predict` request body:

```json
{
  "input": {
    "type": "stac_item",
    "collection": "bay_of_bengal_swot_ssha",
    "item_id": "bay_of_bengal_swot_cycle_050_pass_230",
    "source_file": "SWOT_L3_LR_SSH_Expert_050_230_....nc"
  },
  "parameters": {
    "confidenceThreshold": 0.4
  },
  "source": {
    "collection": "bay_of_bengal_swot_ssha",
    "cycle": "050",
    "pass": "230",
    "direction": "descending",
    "datetime": "2024-01-01T00:00:00Z",
    "variable": "ssha_unfiltered"
  }
}
```

The service resolves `source_file` against the mounted data volume (see below), runs
the detector, and returns `status`, `detection_count`, `processing_seconds`, and a
`geojson` `FeatureCollection` shaped by `model_output_schema.json`.

## Docker image (model service)

The Dockerfile builds a self-contained image: micromamba environment, model weights,
API code, and pipeline modules — everything needed to serve `/predict`.

### Build

```bash
docker build -t swot-internal-wave-yolo .
```

### Run

The service expects **SWOT NetCDF granules** to be available on disk, mounted read-only
into the container at `/data`. Granules can either sit directly under `/data`, or be
organized by cycle under `/data/cycle_<3-digit-cycle>/` (checked first):

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/your/swot/netcdf/files:/data:ro \
  swot-internal-wave-yolo
```

For example, if a request's `source.cycle` is `"050"` and `input.source_file` is
`SWOT_L3_LR_SSH_Expert_050_230_....nc`, the service looks for the file in this order:

1. `/data/cycle_050/SWOT_L3_LR_SSH_Expert_050_230_....nc`
2. `/data/SWOT_L3_LR_SSH_Expert_050_230_....nc`

Override the mount point with the `SWOT_DATA_ROOT` environment variable if you need the
container to look somewhere other than `/data`:

```bash
docker run --rm -p 8000:8000 \
  -e SWOT_DATA_ROOT=/mnt/swot \
  -v /path/to/your/swot/netcdf/files:/mnt/swot:ro \
  swot-internal-wave-yolo
```

Once running, verify it's healthy and try a prediction:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Local development (without Docker)

1. Create the conda/micromamba environment:

   ```bash
   micromamba create -f environment.yml
   micromamba activate swot-internal-wave
   ```

   (Or `conda env create -f environment.yml` if you don't have micromamba.)

2. Run the API locally:

   ```bash
   SWOT_DATA_ROOT=./test-data uvicorn api:app --reload --port 8000
   ```

3. Run the tests:

   ```bash
   pytest
   ```

   `tests/test_model.py` and `tests/test_api.py` are unit tests (the latter mocks the
   detector). `tests/test_integration.py` runs the real model end-to-end against the
   sample granules in [`test-data/`](test-data/) — no mocking, so it exercises the full
   filter → YOLO → GeoJSON path described in [`pipeline/README.md`](pipeline/README.md).

4. To run the standalone CLI pipeline instead of the API, see
   [`pipeline/README.md`](pipeline/README.md).
