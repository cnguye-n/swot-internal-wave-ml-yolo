#!/usr/bin/env python3
"""
step_01_read_stac.py


Purpose:
    Given selected STAC item ids, get the NetCDF paths.

    - MMGIS backend already talks to PgSTAC.
    - We call the existing plugin API:
        http://localhost:8888/api/swotDetectorApi/items?cycle=050

Input:
    item ids,ex:
        bay_of_bengal_swot_cycle_050_pass_008

Output:
    JSON list with:
        item_id
        cycle
        pass
        netcdf_path

Next step:
    step_02 opens each netcdf_path with xarray and runs Johnny's model.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import urllib.request
from pathlib import Path

#tell Node backend tell python the corect API URL
MMGIS_API_BASE = os.environ.get(
    "MMGIS_API_BASE",
    "http://localhost:8888/api/swotDetectorApi",
)



def strip_file_uri(path_or_uri: str) -> str:
    """
    Convert:
        file:///home/ctnguyen/file.nc

    Into:
        /home/ctnguyen/file.nc
    """
    if path_or_uri.startswith("file://"):
        return path_or_uri.replace("file://", "", 1)

    return path_or_uri


def get_cycle_from_item_id(item_id: str) -> str:
    """
    Extract cycle number from item id.

    Example:
        bay_of_bengal_swot_cycle_050_pass_008
    returns:
        050
    """
    match = re.search(r"cycle_(\d{3})", item_id)

    if not match:
        raise ValueError(f"Could not find cycle in item id: {item_id}")

    return match.group(1)


def fetch_json(url: str) -> dict:
    """
    Read JSON from a URL using only Python standard library.
    No requests package needed.
    """
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def get_items_for_cycle(cycle: str) -> list[dict]:
    """
    Ask the existing MMGIS plugin backend for all STAC items in one cycle.

    This backend already knows how to query PgSTAC.
    """
    url = f"{MMGIS_API_BASE}/items?cycle={cycle}"
    data = fetch_json(url)
    return data.get("items", [])


def find_selected_netcdf_paths(item_ids: list[str]) -> list[dict]:
    """
    Main function.

    Given selected item ids:
        1. Group them by cycle.
        2. Ask MMGIS backend for items in that cycle.
        3. Match the selected ids.
        4. Return NetCDF paths only.
    """
    selected_ids = set(item_ids)

    cycles = sorted({get_cycle_from_item_id(item_id) for item_id in item_ids})

    selected_records = []

    for cycle in cycles:
        items = get_items_for_cycle(cycle)

        for item in items:
            if item.get("id") not in selected_ids:
                continue

            netcdf_href = item.get("assets", {}).get("netcdf")

            if not netcdf_href:
                print(f"Warning: no NetCDF asset for {item.get('id')}")
                continue

            netcdf_path = strip_file_uri(netcdf_href)

            selected_records.append(
                {
                    "item_id": item.get("id"),
                    "collection": item.get("collection"),
                    "cycle": item.get("cycle"),
                    "pass": item.get("pass"),
                    "datetime": item.get("datetime"),
                    "netcdf_href": netcdf_href,
                    "netcdf_path": str(Path(netcdf_path).expanduser().resolve()),
                }
            )

    found_ids = {record["item_id"] for record in selected_records}
    missing_ids = selected_ids - found_ids

    for missing_id in sorted(missing_ids):
        print(f"Warning: selected item id was not found: {missing_id}")

    return selected_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--item-ids",
        nargs="+",
        required=True,
        help="Selected STAC item ids from the MMGIS plugin.",
    )

    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save selected NetCDF records.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = find_selected_netcdf_paths(args.item_ids)

    print(json.dumps(records, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()