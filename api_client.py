from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover - fallback for Windows/Py<3.9
    ZoneInfo = None  # type: ignore

try:
    from .aqi import compute_dominant_aqi_simple  # type: ignore[attr-defined]
except ImportError:
    try:
        from aqi import compute_dominant_aqi_simple  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive guard for optional dependency
        compute_dominant_aqi_simple = None  # type: ignore
except Exception:  # pragma: no cover - defensive guard for optional dependency
    compute_dominant_aqi_simple = None  # type: ignore

try:
    from .config import API_DATA_BASE_URL, USER_TIMEZONE  # type: ignore[attr-defined]
except ImportError:
    from config import API_DATA_BASE_URL, USER_TIMEZONE  # type: ignore

TIMESTAMP_DISPLAY_COLUMN = "Timestamp (-3h)"
REQUEST_TIMEOUT_SECONDS = 10.0
MEASUREMENT_OFFSET = timedelta(hours=3)
AQI_CATEGORY_BANDS: tuple[tuple[int, Optional[int], str], ...] = (
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, None, "Hazardous"),
)


@dataclass(slots=True)
class StationSnapshot:
    """Normalized representation of the API response for a single station."""

    device_id: str
    short_name: str
    status_code: Optional[str]
    api_message: Optional[str]
    api_device_id: Optional[str]
    measurement_time: Optional[str]
    measurement_time_raw: Optional[str]
    entity_count: int
    dataframe: pd.DataFrame
    pollutant_concentrations: Dict[str, float]
    display_concentrations: Dict[str, float]
    aqi_value: Optional[int]
    aqi_pollutant: Optional[str]
    aqi_category: Optional[str]
    error: Optional[str]
    request_url: str

    @property
    def ok(self) -> bool:
        return not self.error


class AirQualityAPIClient:
    """Thin wrapper around the telemetry API with timeout and error handling."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url or API_DATA_BASE_URL or "").rstrip("/")
        self.timeout = timeout
        self._external_session = session
        self._session = session or requests.Session()

    def fetch_station_snapshot(self, device_id: str, short_name: str) -> StationSnapshot:
        url = f"{self.base_url}/intfa/queryData/{device_id}" if self.base_url else f"/intfa/queryData/{device_id}"
        http_error: Optional[str] = None
        status_code: Optional[str] = None
        api_message: Optional[str] = None
        api_device_id: Optional[str] = None
        payload: Dict[str, Any] = {}

        try:
            response = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:  # pragma: no cover - network-specific guard
            http_error = f"HTTP error: {exc}"
        else:
            try:
                payload = response.json() if response.content else {}
            except ValueError as exc:
                http_error = f"JSON decode error: {exc}"
            else:
                if response.status_code != 200:
                    http_error = f"HTTP {response.status_code}: {response.reason}"
                elif isinstance(payload, dict):
                    status_code = _safe_str(payload.get("statusCode"))
                    api_message = _safe_str(payload.get("message")) or None
                    api_device_id = _safe_str(payload.get("deviceId")) or None
                    if status_code and status_code != "200":
                        http_error = f"API status {status_code}: {api_message or 'Unknown error'}"
                else:
                    http_error = "Unexpected payload structure: expected dict"

        if http_error:
            entities: list[dict[str, Any]] = []
        else:
            entities = _filter_entities(payload.get("entity")) if isinstance(payload, dict) else []

        measurement_time_raw = _extract_measurement_time(entities)
        measurement_time_display = _shift_measurement_time(measurement_time_raw) or measurement_time_raw
        dataframe = _build_entities_frame(entities)
        pollutant_concentrations, display_concentrations = _collect_pollutant_concentrations(entities)

        aqi_value: Optional[int] = None
        aqi_pollutant: Optional[str] = None
        aqi_category: Optional[str] = None
        if compute_dominant_aqi_simple and pollutant_concentrations and not http_error:
            try:
                result = compute_dominant_aqi_simple(pollutant_concentrations)
            except Exception:
                result = None
            if result:
                aqi_value, aqi_pollutant = result
                aqi_category = _aqi_category(aqi_value)

        return StationSnapshot(
            device_id=device_id,
            short_name=short_name,
            status_code=status_code,
            api_message=api_message,
            api_device_id=api_device_id,
            measurement_time=measurement_time_display,
            measurement_time_raw=measurement_time_raw,
            entity_count=len(entities),
            dataframe=dataframe,
            pollutant_concentrations=pollutant_concentrations,
            display_concentrations=display_concentrations,
            aqi_value=aqi_value,
            aqi_pollutant=aqi_pollutant,
            aqi_category=aqi_category,
            error=http_error,
            request_url=url,
        )

    def fetch_all(self, stations: Dict[str, str]) -> Dict[str, StationSnapshot]:
        return {device_id: self.fetch_station_snapshot(device_id, short_name) for device_id, short_name in stations.items()}

    def close(self) -> None:
        if not self._external_session:
            self._session.close()


# --- Helper utilities -----------------------------------------------------

def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _filter_entities(entities: Any) -> list[dict[str, Any]]:
    return [item for item in entities or [] if isinstance(item, dict)]


def _extract_measurement_time(entities: Iterable[dict[str, Any]]) -> Optional[str]:
    for item in entities:
        dt_value = item.get("datetime")
        if isinstance(dt_value, str) and dt_value.strip():
            return dt_value.strip()
    return None


def _shift_measurement_time(raw_value: Any) -> Optional[str]:
    if not raw_value:
        return None
    try:
        ts = pd.to_datetime(raw_value)
    except Exception:
        return None
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0] if not ts.empty else None
    if ts is None or pd.isna(ts):
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):
        return None
    shifted = ts - MEASUREMENT_OFFSET
    return shifted.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return value


SUBSCRIPT_TRANSLATION = {
    ord('\u2080'): '0',
    ord('\u2081'): '1',
    ord('\u2082'): '2',
    ord('\u2083'): '3',
    ord('\u2084'): '4',
    ord('\u2085'): '5',
    ord('\u2086'): '6',
    ord('\u2087'): '7',
    ord('\u2088'): '8',
    ord('\u2089'): '9',
}


def _normalize_pollutant_key(name: Any) -> Optional[str]:
    if not name:
        return None
    text = str(name).lower()
    text = text.translate(SUBSCRIPT_TRANSLATION)
    normalized = text.replace(" ", "").replace("-", "").replace("_", "")
    normalized = normalized.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    normalized = ''.join(ch for ch in normalized if ch.isalnum())
    if 'pm25' in normalized or 'pm2_5' in normalized or ('pm2' in normalized and 'pm10' not in normalized):
        return 'pm2_5'
    if 'pm10' in normalized:
        return 'pm10'
    if 'no2' in normalized or 'nitrogendioxide' in normalized:
        return 'no2'
    if 'so2' in normalized or 'sulfurdioxide' in normalized:
        return 'so2'
    if 'o3' in normalized or 'ozone' in normalized:
        return 'o3'
    if normalized == 'co' or normalized.endswith('co') or 'carbonmonoxide' in normalized:
        return 'co'
    return None


def _normalize_unit(unit: Any) -> str:
    if not unit:
        return ""
    text = str(unit).strip().lower()
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    text = text.replace("\u00b3", "3")
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    cleaned = ascii_text.replace(" ", "")
    mapping = {
        "ug/m3": "ug/m3",
        "ugm3": "ug/m3",
        "ug/m": "ug/m3",
        "mg/m3": "mg/m3",
        "ppm": "ppm",
        "ppb": "ppb",
    }
    return mapping.get(cleaned, cleaned)


def _collect_pollutant_concentrations(entities: Iterable[dict[str, Any]]) -> tuple[Dict[str, float], Dict[str, float]]:
    concentrations: Dict[str, float] = {}
    display_values: Dict[str, float] = {}
    for entry in entities:
        key = _normalize_pollutant_key(entry.get("eName") or entry.get("eKey"))
        if not key:
            continue
        value = _coerce_value(entry.get("eValue"))
        try:
            numeric_value = float(value)  # type: ignore[arg-type]
        except Exception:
            continue
        unit = _normalize_unit(entry.get("eUnit"))
        display_values[key] = numeric_value
        if key == "co" and unit == "ppb":
            concentrations[key] = numeric_value / 1000.0
        else:
            concentrations[key] = numeric_value
    return concentrations, display_values


def _aqi_category(aqi_val: Optional[int]) -> Optional[str]:
    if aqi_val is None:
        return None
    value = int(aqi_val)
    for lower, upper, label in AQI_CATEGORY_BANDS:
        if upper is None:
            if value >= lower:
                return label
        elif lower <= value <= upper:
            return label
    return None


def _build_entities_frame(entities: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for entry in entities:
        rows.append(
            {
                "Metric": entry.get("eName") or entry.get("eKey"),
                "Value": _coerce_value(entry.get("eValue")),
                "Unit": entry.get("eUnit"),
                "Timestamp": entry.get("datetime"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "Timestamp" in df.columns:
        df[TIMESTAMP_DISPLAY_COLUMN] = df["Timestamp"].apply(lambda x: _shift_measurement_time(x) or x)
        df = df.drop(columns=["Timestamp"])
    column_order = ["Metric", "Value", "Unit", TIMESTAMP_DISPLAY_COLUMN]
    available_cols = [col for col in column_order if col in df.columns]
    return df[available_cols]


def snapshots_to_aqi_frame(snapshots: Iterable[StationSnapshot]) -> pd.DataFrame:
    records: list[Dict[str, Any]] = []
    for snapshot in snapshots:
        records.append(
            {
                "station_id": snapshot.device_id,
                "short_name": snapshot.short_name,
                "aqi": snapshot.aqi_value,
                "aqi_category": snapshot.aqi_category,
                "dominant_pollutant": snapshot.aqi_pollutant,
                "measurement_time": snapshot.measurement_time,
                "error": snapshot.error,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("short_name").reset_index(drop=True)
    return df


def format_local_timestamp(dt_utc: Optional[datetime]) -> str:
    if dt_utc is None:
        return "-"
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    if ZoneInfo and USER_TIMEZONE:
        try:
            local = dt_utc.astimezone(ZoneInfo(USER_TIMEZONE))
        except Exception:
            local = dt_utc.astimezone(timezone.utc)
    else:
        local = dt_utc.astimezone(timezone.utc)
    return local.strftime("%Y-%m-%d %H:%M:%S")

