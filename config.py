from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

API_DATA_BASE_URL = "http://47.56.224.23:8005"
USER_TIMEZONE = "Asia/Qyzylorda"

STATION_MAPPING: Dict[str, str] = {
    "16098828": "app_center",
    "16101230": "app_2pavlodar",
    "16101231": "app_pspu",
    "16101232": "app_metallurg",
    "16101233": "app_zaton",
}


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    short_name: str
    organization: Optional[str]
    address: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    coordinates_original: Optional[str]


STATION_METADATA: List[StationRecord] = [
    StationRecord(
        station_id="16098828",
        short_name="app_center",
        organization="Secondary School No. 7",
        address="13 Victory Square, Pavlodar",
        latitude=52.28444444444444,
        longitude=76.94666666666667,
        coordinates_original=None,
    ),
    StationRecord(
        station_id="16101230",
        short_name="app_2pavlodar",
        organization="Secondary School No. 12 named after K.N. Bekhozhin",
        address="60 Saltykov-Shchedrin Street, Pavlodar",
        latitude=52.291666666666664,
        longitude=77.00305555555556,
        coordinates_original=None,
    ),
    StationRecord(
        station_id="16101231",
        short_name="app_pspu",
        organization="Alkey Margulan University",
        address="60 Olzhabay Batyr Street, Pavlodar",
        latitude=52.29888888888889,
        longitude=76.955,
        coordinates_original=None,
    ),
    StationRecord(
        station_id="16101232",
        short_name="app_metallurg",
        organization="Secondary School No. 37 named after Zh. Tashenov",
        address="6/2 Vorushin Street, Pavlodar",
        latitude=52.257222222222225,
        longitude=76.99194444444444,
        coordinates_original=None,
    ),
    StationRecord(
        station_id="16101233",
        short_name="app_zaton",
        organization="Secondary School named after M. Alimbayev",
        address="17/2 Pavel Vasiliev Street, Pavlodar",
        latitude=52.26027777777778,
        longitude=76.95277777777778,
        coordinates_original=None,
    ),
]

POLLUTANTS: Dict[str, str] = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "o3": "O3",
    "co": "CO",
}

POLLUTANTS_HTML: Dict[str, str] = {
    "pm2_5": "PM<sub>2.5</sub>",
    "pm10": "PM<sub>10</sub>",
    "no2": "NO<sub>2</sub>",
    "so2": "SO<sub>2</sub>",
    "o3": "O<sub>3</sub>",
    "co": "CO",
}
