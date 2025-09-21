from __future__ import annotations

from typing import Dict

import pandas as pd

try:
    from .config import STATION_METADATA, STATION_MAPPING  # type: ignore[attr-defined]
except ImportError:
    from config import STATION_METADATA, STATION_MAPPING  # type: ignore

REQUIRED_COLUMNS = [
    "station_id",
    "short_name",
    "organization",
    "address",
    "coordinates_original",
    "latitude",
    "longitude",
]


def load_station_metadata() -> pd.DataFrame:
    """Return a dataframe with station metadata aligned to the configured mapping."""
    records = [record.__dict__ for record in STATION_METADATA]
    df = pd.DataFrame(records, columns=REQUIRED_COLUMNS)

    mapping: Dict[str, str] = STATION_MAPPING
    existing_ids = set(df["station_id"].astype(str))

    for station_id, short_name in mapping.items():
        if station_id not in existing_ids:
            df.loc[len(df)] = {
                "station_id": station_id,
                "short_name": short_name,
                "organization": None,
                "address": None,
                "coordinates_original": None,
                "latitude": None,
                "longitude": None,
            }
    df["station_id"] = df["station_id"].astype(str)
    df["short_name"] = df["short_name"].astype(str)
    df = df.sort_values("short_name").reset_index(drop=True)
    return df
