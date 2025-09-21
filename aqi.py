from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Source table with AQI breakpoints
DEFAULT_AQI_TABLE = Path(__file__).resolve().parent / "data" / "aqi.csv"


@dataclass(frozen=True)
class Breakpoint:
    pollutant: str
    conc_low: float
    conc_high: float
    aqi_low: int
    aqi_high: int


def _parse_range(val: str) -> Optional[Tuple[float, float]]:
    val = (val or "").strip()
    if not val or val == "-":
        return None
    parts = val.split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected range format: {val}")
    a, b = parts[0].strip(), parts[1].strip()
    return float(a), float(b)


def _round_aqi(x: float) -> int:
    return int(math.floor(x + 0.5))


def _find_segment(breakpoints: List[Breakpoint], concentration: float) -> Optional[Breakpoint]:
    for seg in breakpoints:
        if seg.conc_low <= concentration <= seg.conc_high:
            return seg
    if breakpoints:
        max_seg = max(breakpoints, key=lambda s: s.conc_high)
        if concentration > max_seg.conc_high:
            return max_seg
        min_seg = min(breakpoints, key=lambda s: s.conc_low)
        if concentration < min_seg.conc_low:
            return min_seg
    return None


def _aqi_from_breakpoints(bps: List[Breakpoint], c: float) -> Optional[int]:
    seg = _find_segment(bps, c)
    if seg is None:
        return None
    cl, ch, il, ih = seg.conc_low, seg.conc_high, seg.aqi_low, seg.aqi_high
    aqi_val = ih if ch == cl else (ih - il) / (ch - cl) * (c - cl) + il
    return max(0, min(500, _round_aqi(aqi_val)))


def load_aqi_breakpoints(csv_path: Path | str = DEFAULT_AQI_TABLE) -> Dict[str, List[Breakpoint]]:
    """Load breakpoint segments from `aqi.csv` for simplified AQI calculation.

    Returns keys: 'o3_8h','o3_1h','pm2_5_24h','pm10_24h','co_8h','so2_1h','so2_24h','no2_1h'.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"AQI table not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Resolve real column names by substring match (case-insensitive)
        cols = {k: None for k in [
            "O3 (ppb) 8-hr",
            "O3 (ppb) 1-hr",
            "PM2.5",
            "PM10",
            "CO (ppm) 8-hr",
            "SO2 (ppb) 1-hr; 24-hr",
            "NO2 (ppb) 1-hr",
        ]}
        for name in reader.fieldnames or []:
            n = name.strip()
            for key in list(cols.keys()):
                if key.lower() in n.lower():
                    # Store the original header name (with spaces),
                    # because DictReader row keys use exact originals
                    cols[key] = name
        # Fallbacks for PM headers with encoding variants
        if cols["PM2.5"] is None:
            for name in reader.fieldnames or []:
                if "PM2.5 (".lower() in name.lower():
                    cols["PM2.5"] = name
                    break
        if cols["PM10"] is None:
            for name in reader.fieldnames or []:
                if "PM10 (".lower() in name.lower():
                    cols["PM10"] = name
                    break

        aqi_col = None
        for name in reader.fieldnames or []:
            if name.strip().lower() == "aqi":
                aqi_col = name
                break
        if not aqi_col:
            raise ValueError("CSV must contain 'AQI' column")

        rows = list(reader)

    bk: Dict[str, List[Breakpoint]] = {
        "o3_8h": [],
        "o3_1h": [],
        "pm2_5_24h": [],
        "pm10_24h": [],
        "co_8h": [],
        "so2_1h": [],
        "so2_24h": [],
        "no2_1h": [],
    }

    def push(pollutant_key: str, conc_range: Optional[Tuple[float, float]], aqi_range: Tuple[int, int]):
        if conc_range is None:
            return
        cl, ch = conc_range
        il, ih = aqi_range
        bk[pollutant_key].append(Breakpoint(pollutant=pollutant_key, conc_low=cl, conc_high=ch, aqi_low=il, aqi_high=ih))

    for r in rows:
        aqi_lo, aqi_hi = _parse_range(r[aqi_col])  # type: ignore
        assert aqi_lo is not None and aqi_hi is not None
        aqi_range = (int(aqi_lo), int(aqi_hi))

        if cols["PM2.5"]:
            push("pm2_5_24h", _parse_range(r[cols["PM2.5"]]), aqi_range)
        if cols["PM10"]:
            push("pm10_24h", _parse_range(r[cols["PM10"]]), aqi_range)
        if cols["O3 (ppb) 8-hr"]:
            push("o3_8h", _parse_range(r[cols["O3 (ppb) 8-hr"]]), aqi_range)
        if cols["O3 (ppb) 1-hr"]:
            push("o3_1h", _parse_range(r[cols["O3 (ppb) 1-hr"]]), aqi_range)
        if cols["CO (ppm) 8-hr"]:
            push("co_8h", _parse_range(r[cols["CO (ppm) 8-hr"]]), aqi_range)
        if cols["NO2 (ppb) 1-hr"]:
            push("no2_1h", _parse_range(r[cols["NO2 (ppb) 1-hr"]]), aqi_range)

        # SO2: split 1h for bands up to 200, 24h for bands from 201
        so2_col = cols["SO2 (ppb) 1-hr; 24-hr"]
        if so2_col:
            cr = _parse_range(r[so2_col])
            if cr is not None:
                if aqi_range[1] <= 200:
                    push("so2_1h", cr, aqi_range)
                elif aqi_range[0] >= 201:
                    push("so2_24h", cr, aqi_range)

    for k in bk:
        bk[k].sort(key=lambda b: (b.conc_low, b.conc_high))
    return bk


def compute_dominant_aqi_simple(
    pollutant_concentrations: Dict[str, float],
    csv_path: Path | str = DEFAULT_AQI_TABLE,
) -> Optional[Tuple[int, str]]:
    """Compute dominant AQI using fixed columns and simple O3 rule.

    Accepted keys: pm2_5|pm25|pm2.5, pm10, o3, so2, no2, co (case-insensitive).
    Returns (AQI_value_int, dominant_pollutant_name) or None.
    """
    if not pollutant_concentrations:
        return None

    bk = load_aqi_breakpoints(csv_path)
    so2_combined = sorted((bk.get("so2_1h", []) + bk.get("so2_24h", [])), key=lambda b: (b.conc_low, b.conc_high))

    def norm_key(k: str) -> str:
        s = k.strip().lower().replace(" ", "")
        if s in {"pm2.5", "pm25", "pm2_5"}:
            return "pm2_5"
        return s

    def human_name(nk: str) -> str:
        return {"pm2_5": "PM2.5", "pm10": "PM10", "o3": "O3", "so2": "SO2", "no2": "NO2", "co": "CO"}.get(nk, nk)

    results: List[Tuple[int, str]] = []
    for k_in, conc in pollutant_concentrations.items():
        try:
            c = float(conc)
        except Exception:
            continue
        key = norm_key(k_in)

        if key == "o3":
            sub = "o3_1h" if c > 200 else "o3_8h"
            aqi_val = _aqi_from_breakpoints(bk.get(sub, []), c)
            if aqi_val is not None:
                results.append((aqi_val, human_name("o3")))
        elif key == "pm2_5":
            aqi_val = _aqi_from_breakpoints(bk.get("pm2_5_24h", []), c)
            if aqi_val is not None:
                results.append((aqi_val, human_name("pm2_5")))
        elif key == "pm10":
            aqi_val = _aqi_from_breakpoints(bk.get("pm10_24h", []), c)
            if aqi_val is not None:
                results.append((aqi_val, human_name("pm10")))
        elif key == "co":
            aqi_val = _aqi_from_breakpoints(bk.get("co_8h", []), c)
            if aqi_val is not None:
                results.append((aqi_val, human_name("co")))
        elif key == "no2":
            aqi_val = _aqi_from_breakpoints(bk.get("no2_1h", []), c)
            if aqi_val is not None:
                results.append((aqi_val, human_name("no2")))
        elif key == "so2":
            aqi_val = _aqi_from_breakpoints(so2_combined, c)
            if aqi_val is not None:
                results.append((aqi_val, human_name("so2")))

    if not results:
        return None
    return max(results, key=lambda t: t[0])


__all__ = ["compute_dominant_aqi_simple"]

