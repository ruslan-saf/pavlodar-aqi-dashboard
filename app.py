from __future__ import annotations

import base64
import io
import json
import numbers
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium import plugins, raster_layers
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon
from streamlit_folium import st_folium

try:
    from pykrige.ok import OrdinaryKriging
    KRIGING_AVAILABLE = True
except ImportError:
    KRIGING_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional dependency
    st_autorefresh = None  # type: ignore

try:  # pragma: no cover - allow running via ``streamlit run app.py``
    from .config import POLLUTANTS, STATION_MAPPING
    from .api_client import (
        AirQualityAPIClient,
        StationSnapshot,
        format_local_timestamp,
        snapshots_to_aqi_frame,
    )
    from .metadata import load_station_metadata
except ImportError:  # pragma: no cover - fallback for local execution
    from config import POLLUTANTS, STATION_MAPPING  # type: ignore
    from api_client import (  # type: ignore
        AirQualityAPIClient,
        StationSnapshot,
        format_local_timestamp,
        snapshots_to_aqi_frame,
    )
    from metadata import load_station_metadata  # type: ignore

st.set_page_config(layout="wide", page_title="Pavlodar Air Quality Monitoring Project", page_icon="AQI")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }
    .hero {
        background-color: #eaf2f8;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 45px rgba(52, 152, 219, 0.12);
    }
    .hero p {
        font-size: 18px;
        color: #34495e;
        margin-bottom: 24px;
    }
    .cta-button {
        background-color: #3498db;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-size: 18px;
        display: inline-block;
    }
    .cta-button:hover {
        background-color: #2c81ba;
    }
    .section-wrapper {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .section-description {
        color: #5d6d7e;
        font-size: 16px;
        margin-bottom: 1.2rem;
    }
    .status-card-wrapper {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 18px 36px rgba(26, 47, 96, 0.12);
        padding: 12px 12px 4px 12px;
        margin-top: 1rem;
    }
    .kpi-section {
        margin-top: 1.5rem;
    }
    .map-description {
        color: #5d6d7e;
        margin-bottom: 1rem;
    }
    .table-description {
        color: #5d6d7e;
        margin-bottom: 1rem;
    }
    .stDataFrame table thead th {
        white-space: pre-line;
        line-height: 1.2;
    }
    footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 40px;
    }
    footer a {
        color: #3498db;
        text-decoration: none;
    }
    footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CACHE_TTL_SECONDS = 300  # 5 minutes
DEFAULT_REFRESH_MINUTES = 5
MAX_REFRESH_MINUTES = 30
REFRESH_COUNTER_KEY = "public_app_refresh_counter"
REFRESH_MINUTES_KEY = "public_app_refresh_minutes"

AQI_CATEGORY_COLORS: Dict[str, str] = {
    "Good": "#2ecc71",
    "Moderate": "#f1c40f",
    "Unhealthy for Sensitive": "#e67e22",
    "Unhealthy": "#e74c3c",
    "Very Unhealthy": "#8e44ad",
    "Hazardous": "#7f0000",
}

CATEGORY_BADGE_COLORS: Dict[str, str] = {
    "Good": "#ebf7ee",
    "Moderate": "#fcf7e3",
    "Unhealthy for Sensitive": "#fbe9dd",
    "Unhealthy": "#fbdfdf",
    "Very Unhealthy": "#f0e6fa",
    "Hazardous": "#f5d7d7",
}


CATEGORY_LABELS_RU: Dict[str, str] = {
    "Good": "Good",
    "Moderate": "Moderate",
    "Unhealthy for Sensitive": "Unhealthy for Sensitive",
    "Unhealthy": "Unhealthy",
    "Very Unhealthy": "Very Unhealthy",
    "Hazardous": "Hazardous",
}



CATEGORY_FACE: Dict[str, str] = {
    "Good": "&#128578;",
    "Moderate": "&#128528;",
    "Unhealthy for Sensitive": "&#128567;",
    "Unhealthy": "&#128561;",
    "Very Unhealthy": "&#129314;",
    "Hazardous": "&#9760;",
}

POLLUTANT_UNITS: Dict[str, str] = {
    "pm2_5": "µg/m³",
    "pm10": "µg/m³",
    "no2": "ppb",
    "so2": "ppb",
    "o3": "ppb",
    "co": "ppb",
}

POLLUTANTS_DISPLAY: Dict[str, str] = {
    "pm2_5": "PM₂.₅",
    "pm10": "PM₁₀",
    "no2": "NO₂",
    "so2": "SO₂",
    "o3": "O₃",
    "co": "CO",
}


TEMPERATURE_COLUMN_LABEL = "Temperature (°C)"
HUMIDITY_COLUMN_LABEL = "Humidity (%)"
TEMPERATURE_KEYWORDS = ("temperature", "температур", "temp", "温度")
HUMIDITY_KEYWORDS = ("humidity", "влажн", "relativehumidity", "rh", "湿度")






BOUNDARY_GEOJSON_PATH = Path(__file__).resolve().parent / 'data' / 'research_area_boundary.geojson'
AQI_BOUNDARIES = [0, 50, 100, 150, 200, 300, 500]
AQI_COLOR_SCALE = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8e44ad', '#7f0000']
GRID_RESOLUTION = 120


@lru_cache(maxsize=1)
def _load_boundary() -> tuple[dict, Polygon]:
    if not BOUNDARY_GEOJSON_PATH.exists():
        raise FileNotFoundError(f"Boundary file not found: {BOUNDARY_GEOJSON_PATH}")
    with open(BOUNDARY_GEOJSON_PATH, 'r', encoding='utf-8') as handle:
        boundary_json = json.load(handle)
    coords = boundary_json['features'][0]['geometry']['coordinates'][0]
    polygon = Polygon([(lon, lat) for lon, lat in coords])
    return boundary_json, polygon


def _interpolate_aqi_surface(points: np.ndarray, values: np.ndarray, boundary: Polygon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_lon, min_lat, max_lon, max_lat = boundary.bounds
    padding = 0.01
    grid_lon = np.linspace(min_lon - padding, max_lon + padding, GRID_RESOLUTION)
    grid_lat = np.linspace(min_lat - padding, max_lat + padding, GRID_RESOLUTION)
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)

    grid_values = None
    if KRIGING_AVAILABLE and len(points) >= 3:
        try:
            ok = OrdinaryKriging(
                points[:, 0],
                points[:, 1],
                values,
                variogram_model='gaussian',
                verbose=False,
                enable_plotting=False,
            )
            kriged, _ = ok.execute('grid', grid_lon, grid_lat)
            grid_values = np.array(kriged)
        except Exception:
            grid_values = None

    if grid_values is None:
        grid_values = griddata(points, values, (lon_mesh, lat_mesh), method='cubic')
        if np.isnan(grid_values).all():
            grid_values = griddata(points, values, (lon_mesh, lat_mesh), method='linear')
        if np.isnan(grid_values).all():
            grid_values = griddata(points, values, (lon_mesh, lat_mesh), method='nearest')

    mask = np.array([[boundary.contains(Point(lon, lat)) for lon in grid_lon] for lat in grid_lat])
    grid_values = np.where(mask, grid_values, np.nan)

    return lon_mesh, lat_mesh, grid_values



def _create_overlay_image(lon_mesh: np.ndarray, lat_mesh: np.ndarray, grid_values: np.ndarray) -> tuple[str, list[list[float]]] | None:
    masked = np.ma.masked_invalid(grid_values)
    if masked.mask.all():
        return None

    levels = AQI_BOUNDARIES
    colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023']

    fig, ax = plt.subplots(figsize=(7, 7), dpi=220)
    ax.set_axis_off()
    try:
        contour = ax.contourf(
            lon_mesh,
            lat_mesh,
            masked,
            levels=levels,
            colors=colors,
            extend='both',
        )
        contour_lines = ax.contour(
            lon_mesh,
            lat_mesh,
            masked,
            levels=levels,
            colors='#212121',
            linewidths=0.6,
            alpha=0.75,
        )
        ax.clabel(contour_lines, inline=True, fontsize=7, fmt=lambda v: f"{int(v)}")
    except Exception:
        cmap = mcolors.LinearSegmentedColormap.from_list('aqi_surface', colors, N=256)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        ax.imshow(
            masked,
            origin='lower',
            extent=[lon_mesh.min(), lon_mesh.max(), lat_mesh.min(), lat_mesh.max()],
            cmap=cmap,
            norm=norm,
            alpha=0.55,
        )

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    bounds = [[float(lat_mesh.min()), float(lon_mesh.min())], [float(lat_mesh.max()), float(lon_mesh.max())]]
    return f"data:image/png;base64,{img_base64}", bounds




def _add_kriging_overlay(map_obj: folium.Map, metadata: pd.DataFrame, snapshot_by_id: Dict[str, StationSnapshot]) -> None:
    boundary_json, boundary_polygon = _load_boundary()
    point_rows: list[tuple[float, float, float]] = []
    for _, row in metadata.iterrows():
        lat = row.get('latitude')
        lon = row.get('longitude')
        station_id = str(row.get('station_id'))
        snap = snapshot_by_id.get(station_id)
        if snap and snap.aqi_value is not None and not pd.isna(lat) and not pd.isna(lon):
            point_rows.append((float(lon), float(lat), float(snap.aqi_value)))

    if len(point_rows) < 3:
        folium.GeoJson(
            boundary_json,
            name='Boundary',
            style_function=lambda _: {'color': 'red', 'weight': 2, 'fillOpacity': 0, 'fill': False},
        ).add_to(map_obj)
        return

    lon_mesh, lat_mesh, grid_values = _interpolate_aqi_surface(
        np.array(point_rows)[:, :2],
        np.array(point_rows)[:, 2],
        boundary_polygon,
    )
    overlay = _create_overlay_image(lon_mesh, lat_mesh, grid_values)
    if overlay is not None:
        image_str, bounds = overlay
        raster_layers.ImageOverlay(
            image=image_str,
            bounds=bounds,
            opacity=0.45,
            interactive=True,
            cross_origin=False,
            name='AQI surface',
            zindex=1,
        ).add_to(map_obj)

    folium.GeoJson(
        boundary_json,
        name='Boundary outline',
        style_function=lambda _: {'color': 'red', 'weight': 2, 'fillOpacity': 0, 'fill': False},
    ).add_to(map_obj)

    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; width: 300px; background-color: white; border:2px solid grey; z-index:9999; font-size:15px; padding: 16px; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);">
        <p style="font-size:18px; margin-bottom:12px;"><b>AQI Levels</b></p>
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0; white-space:nowrap;"><i class="fa fa-square" style="color:#00E400; font-size:16px;"></i><span>0-50 Good</span></div>
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0; white-space:nowrap;"><i class="fa fa-square" style="color:#FFFF00; font-size:16px;"></i><span>51-100 Moderate</span></div>
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0; white-space:nowrap;"><i class="fa fa-square" style="color:#FF7E00; font-size:16px;"></i><span>101-150 Unhealthy for Sensitive</span></div>
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0; white-space:nowrap;"><i class="fa fa-square" style="color:#FF0000; font-size:16px;"></i><span>151-200 Unhealthy</span></div>
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0; white-space:nowrap;"><i class="fa fa-square" style="color:#8F3F97; font-size:16px;"></i><span>201-300 Very Unhealthy</span></div>
        <div style="display:flex; align-items:center; gap:10px; margin:6px 0; white-space:nowrap;"><i class="fa fa-square" style="color:#7E0023; font-size:16px;"></i><span>300+ Hazardous</span></div>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))




def _build_map(metadata: pd.DataFrame, snapshots_list: Iterable[StationSnapshot]) -> folium.Map:
    merged = metadata.copy()
    snapshot_by_id = {snap.device_id: snap for snap in snapshots_list}

    merged["aqi"] = merged["station_id"].map(lambda sid: snapshot_by_id.get(sid).aqi_value if sid in snapshot_by_id else None)
    merged["aqi_category"] = merged["station_id"].map(lambda sid: snapshot_by_id.get(sid).aqi_category if sid in snapshot_by_id else None)
    merged["dominant_pollutant"] = merged["station_id"].map(lambda sid: snapshot_by_id.get(sid).aqi_pollutant if sid in snapshot_by_id else None)

    if merged["latitude"].dropna().empty or merged["longitude"].dropna().empty:
        center = [52.2873, 76.9674]
    else:
        center = [merged["latitude"].dropna().mean() - 0.0005, merged["longitude"].dropna().mean() + 0.017]

    fmap = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
        scrollWheelZoom=False,
    )
    plugins.Fullscreen().add_to(fmap)

    _add_kriging_overlay(fmap, merged, snapshot_by_id)

    folium.LayerControl().add_to(fmap)

    tooltip_template = "{station}: AQI {aqi}, {pollutant} {value}"
    for _, row in merged.iterrows():
        snap = snapshot_by_id.get(row.get("station_id"))
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue

        category = (snap.aqi_category if snap else None) or row.get("aqi_category") or "No data"
        color = AQI_CATEGORY_COLORS.get(category, "#7f8c8d")
        badge_color = CATEGORY_BADGE_COLORS.get(category, "#ecf0f1")
        aqi_value = (snap.aqi_value if snap else None) or row.get("aqi") or "--"
        dominant = (snap.aqi_pollutant if snap else None) or row.get("dominant_pollutant") or "--"
        organization = row.get("organization") or "--"
        address = row.get("address") or "--"

        pollutant_value = None
        pollutant_unit = ""
        if snap and snap.display_concentrations:
            mapped_key = next((key for key, label in POLLUTANTS.items() if label == dominant), None)
            if mapped_key:
                pollutant_value = snap.display_concentrations.get(mapped_key)
                pollutant_unit = POLLUTANT_UNITS.get(mapped_key, "")
        if isinstance(pollutant_value, numbers.Number):
            value_display = f"{pollutant_value:.1f} {pollutant_unit}".strip()
        else:
            value_display = "--"
        tooltip_text = tooltip_template.format(
            station=row.get("short_name"),
            aqi=aqi_value,
            pollutant=dominant,
            value=value_display,
        )

        popup_html = f"""
        <div style='font-size:14px;'>
            <strong>{row.get('short_name')}</strong><br/>
            <span style='background:{badge_color};padding:6px 10px;border-radius:12px;'>AQI: {aqi_value} ({category})</span><br/>
            Dominant pollutant: {dominant}<br/>
            Organization: {organization}<br/>
            Address: {address}
        </div>
        """

        folium.CircleMarker(
            location=[row.get("latitude"), row.get("longitude")],
            radius=10,
            color='#000000',
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip_text,
            z_index_offset=500,
        ).add_to(fmap)


    return fmap


def _format_pollutant_label(pollutant_key: str) -> str:
    base_label = POLLUTANTS_DISPLAY.get(pollutant_key, POLLUTANTS.get(pollutant_key, pollutant_key.upper()))
    unit = POLLUTANT_UNITS.get(pollutant_key)
    if unit:
        return f"{base_label} ({unit})"
    return base_label


def _extract_environmental_metric(snapshot: StationSnapshot, keywords: tuple[str, ...]) -> tuple[float | None, str]:
    frame = getattr(snapshot, "dataframe", None)
    if frame is None or frame.empty:
        return None, ""
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    try:
        rows_iter = frame.itertuples(index=False)
    except Exception:
        rows_iter = []
    for row in rows_iter:
        metric_name = str(getattr(row, "Metric", "") or "").lower()
        if not metric_name:
            continue
        if not any(keyword in metric_name for keyword in lowered_keywords):
            continue
        raw_value = getattr(row, "Value", None)
        unit = str(getattr(row, "Unit", "") or "")
        if isinstance(raw_value, numbers.Number):
            return float(raw_value), unit
        try:
            numeric = float(raw_value)  # type: ignore[arg-type]
        except Exception:
            return None, unit
        return numeric, unit
    return None, ""


def _build_measurement_table(snapshots_list: Iterable[StationSnapshot]) -> pd.DataFrame:
    pollutant_display = {key: _format_pollutant_label(key) for key in POLLUTANTS.keys()}
    rows = []
    for snapshot in snapshots_list:
        row: Dict[str, object] = {
            "Station": snapshot.short_name,
            "Timestamp": snapshot.measurement_time or "--",
            "AQI": snapshot.aqi_value,
            "AQI Category": snapshot.aqi_category or "--",
            "Dominant\npollutant": snapshot.aqi_pollutant or "--",
        }
        for pollutant_key, display_name in pollutant_display.items():
            raw_value = snapshot.display_concentrations.get(pollutant_key)
            if isinstance(raw_value, numbers.Number):
                row[display_name] = round(float(raw_value), 2)
            else:
                row[display_name] = raw_value
        temp_value, _ = _extract_environmental_metric(snapshot, TEMPERATURE_KEYWORDS)
        humidity_value, _ = _extract_environmental_metric(snapshot, HUMIDITY_KEYWORDS)
        if temp_value is not None:
            row[TEMPERATURE_COLUMN_LABEL] = round(float(temp_value), 1)
        else:
            row[TEMPERATURE_COLUMN_LABEL] = "--"
        if humidity_value is not None:
            row[HUMIDITY_COLUMN_LABEL] = round(float(humidity_value), 1)
        else:
            row[HUMIDITY_COLUMN_LABEL] = "--"
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("Station").reset_index(drop=True)


def _render_status_card(
    container: st.delta_generator.DeltaGenerator,
    snapshot: StationSnapshot | None,
    last_update: str,
    total_stations: int,
    avg_temperature: float | None = None,
    avg_humidity: float | None = None,
) -> None:
    if snapshot is None:
        container.info("No AQI value available yet.")
        return

    category = snapshot.aqi_category or "No data"
    bg = AQI_CATEGORY_COLORS.get(category, "#7f8c8d")
    category_ru = CATEGORY_LABELS_RU.get(category, category)
    face_icon = CATEGORY_FACE.get(category, "&#128578;")
    aqi_value = snapshot.aqi_value
    aqi_display = f"{aqi_value}" if aqi_value is not None else "&mdash;"

    human_pollutant = snapshot.aqi_pollutant or "&mdash;"
    normalized_pollutant = next((key for key, label in POLLUTANTS.items() if label == human_pollutant), None)
    pollutant_value = snapshot.display_concentrations.get(normalized_pollutant) if normalized_pollutant else None
    if isinstance(pollutant_value, numbers.Number):
        pollutant_str = f"{human_pollutant} | {pollutant_value:.1f} {POLLUTANT_UNITS.get(normalized_pollutant, '')}".strip()
    else:
        pollutant_str = human_pollutant

    measurement_label = snapshot.measurement_time or "&mdash;"
    station_label = snapshot.short_name or "&mdash;"

    last_update_label = "Last update"
    station_caption = "Station"
    network_caption = "Stations in network"
    dominant_caption = "Dominant pollutant"
    aqi_label = "AQI USA"
    avg_temperature_label = "Avg temperature"
    avg_humidity_label = "Avg humidity"
    avg_temperature_display = f"{avg_temperature:.1f} °C" if avg_temperature is not None else "—"
    avg_humidity_display = f"{avg_humidity:.1f} %" if avg_humidity is not None else "—"

    card_html = f"""
    <div class=\"status-card-wrapper\">
        <div style=\"border-radius:18px; padding:18px 24px; background:{bg}; color:#fff; box-shadow:0 18px 36px rgba(26, 47, 96, 0.18);\">
            <div style=\"display:flex; justify-content:space-between; font-size:0.9rem; opacity:0.9;\">
                <span>{last_update_label} {last_update}</span>
                <span>{station_caption} {station_label}</span>
            </div>
            <div style=\"display:flex; align-items:center; justify-content:space-between; margin:18px 0; gap:18px; flex-wrap:wrap;\">
                <div style=\"display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:140px;\">
                    <span style=\"font-size:3.2rem;\">{face_icon}</span>
                    <span style=\"font-size:1rem; opacity:0.85;\">{category_ru}</span>
                </div>
                <div style=\"display:flex; flex-wrap:wrap; justify-content:center; gap:60px; flex:1;\">
                    <div style=\"text-align:center; min-width:140px;\">
                        <div style=\"font-size:3.4rem; font-weight:700; line-height:1;\">{aqi_display}</div>
                        <div style=\"font-size:0.95rem; letter-spacing:2px; text-transform:uppercase; opacity:0.85;\">{aqi_label}</div>
                    </div>
                    <div style=\"text-align:center; min-width:140px;\">
                        <div style=\"font-size:3.2rem; font-weight:700; line-height:1;\">{avg_temperature_display}</div>
                        <div style=\"font-size:0.9rem; letter-spacing:1px; text-transform:uppercase; opacity:0.85;\">{avg_temperature_label}</div>
                    </div>
                    <div style=\"text-align:center; min-width:140px;\">
                        <div style=\"font-size:3.2rem; font-weight:700; line-height:1;\">{avg_humidity_display}</div>
                        <div style=\"font-size:0.9rem; letter-spacing:1px; text-transform:uppercase; opacity:0.85;\">{avg_humidity_label}</div>
                    </div>
                </div>
                <div style=\"display:flex; flex-direction:column; align-items:flex-end; justify-content:center; min-width:160px; gap:8px;\">
                    <span style=\"font-size:1.05rem; font-weight:600; text-align:right;\">{measurement_label}</span>
                    <span style=\"font-size:0.9rem; background:rgba(255,255,255,0.22); padding:6px 12px; border-radius:12px;\">{pollutant_str}</span>
                </div>
            </div>
            <div style=\"display:flex; justify-content:space-between; font-size:0.95rem; opacity:0.95; gap:12px; flex-wrap:wrap;\">
                <span>&#128338; <strong>{measurement_label}</strong></span>
                <span>&#128205; <strong>{total_stations}</strong> {network_caption}</span>
                <span>&#127981; {dominant_caption}: <strong>{human_pollutant}</strong></span>
            </div>
        </div>
    </div>
    """
    container.markdown(card_html, unsafe_allow_html=True)


def _style_measurement_table(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    def highlight_row(row: pd.Series) -> list[str]:
        category = row.get("AQI Category")
        color = CATEGORY_BADGE_COLORS.get(category, "white")
        return [f"background-color: {color}"] * len(row)

    styler = table.style.apply(highlight_row, axis=1)
    styler.format(precision=2)
    styler = styler.set_table_styles(
        [
            {
                "selector": "th.col_heading",
                "props": [
                    ("white-space", "pre-line"),
                    ("line-height", "1.2"),
                    ("padding", "8px 10px"),
                ],
            },
        ]
    )
    return styler


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_live_snapshots(counter: int) -> tuple[Dict[str, StationSnapshot], str]:
    client = AirQualityAPIClient()
    try:
        snapshots = client.fetch_all(STATION_MAPPING)
    finally:
        client.close()
    fetched_at = datetime.now(timezone.utc).isoformat()
    return snapshots, fetched_at


def render_dashboard() -> None:
    st.markdown(
        """
        <div class=\"hero\">
            <h1>Pavlodar Air Quality Monitoring Project</h1>
            <p>Public real-time air quality monitoring for Pavlodar</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### About the project")
    st.markdown(
        """
Air quality monitoring is not just a technical task — it’s a vital measure directly tied to our health and overall quality of life. Polluted air can contain harmful substances like fine particulate matter (PM₂.₅), nitrogen dioxide, ozone, and other pollutants that silently penetrate our lungs and bloodstream. This can trigger or worsen asthma, bronchitis, cardiovascular conditions, and even cancer. Children, the elderly, and those with chronic illnesses are especially vulnerable.

By tracking air quality in real time, we gain the ability to make informed decisions — avoiding outdoor activities during peak pollution hours, reducing health risks, shaping urban policies, and even adjusting household behavior. Without monitoring, we simply can’t know how safe the air is — and without that knowledge, effective management becomes impossible.

Moreover, air pollution isn’t just a local issue — it’s a global one. It affects the climate, accelerates warming, and undermines environmental sustainability. That’s why monitoring isn’t just about collecting data; it’s a tool that helps us breathe freely, live longer, and build a safer future.

This project has been funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan within the framework of the grant AP19677560 “Monitoring and mapping of the ecological state of the Pavlodar air environment using machine learning methods”.
        """
    )

    st.markdown("#### Data refresh settings")
    controls_col, button_col = st.columns([3, 1])

    with controls_col:
        stored_minutes = st.session_state.get(REFRESH_MINUTES_KEY, DEFAULT_REFRESH_MINUTES)
        refresh_minutes = st.slider(
            "Auto-refresh interval (minutes)",
            min_value=1,
            max_value=MAX_REFRESH_MINUTES,
            value=int(stored_minutes),
            step=1,
            help="Control how often the page reruns. The API cache expires every five minutes.",
        )
        if refresh_minutes != stored_minutes:
            st.session_state[REFRESH_MINUTES_KEY] = refresh_minutes

    with button_col:
        manual_refresh = st.button("Refresh now", type="primary", use_container_width=True)

    if REFRESH_COUNTER_KEY not in st.session_state:
        st.session_state[REFRESH_COUNTER_KEY] = 0

    if manual_refresh:
        st.session_state[REFRESH_COUNTER_KEY] += 1

    refresh_counter = st.session_state[REFRESH_COUNTER_KEY]

    auto_refresh_counter = 0
    if st_autorefresh is not None:
        auto_refresh_counter = st_autorefresh(
            interval=int(refresh_minutes * 60_000),
            limit=None,
            key=f"public_app_autorefresh_{refresh_minutes}",
        )

    with st.spinner("Fetching station telemetry..."):
        counter = refresh_counter + int(auto_refresh_counter or 0)
        snapshots_dict, fetched_at_iso = load_live_snapshots(counter)

    snapshots = list(snapshots_dict.values())

    if not snapshots:
        st.error("API returned no station data. Please try again later.")
        st.stop()

    last_fetch_utc = datetime.fromisoformat(fetched_at_iso)
    last_fetch_display = format_local_timestamp(last_fetch_utc)
    st.markdown(
        f"**Last API call:** {last_fetch_display} · cache resets every {CACHE_TTL_SECONDS // 60} minutes."
    )

    best_snapshot = max(
        (s for s in snapshots if s.aqi_value is not None),
        key=lambda s: s.aqi_value,
        default=None,
    )

    measurement_table_full = _build_measurement_table(snapshots)

    avg_temperature_value: float | None = None
    avg_humidity_value: float | None = None
    if not measurement_table_full.empty:
        temperature_series = pd.to_numeric(
            measurement_table_full.get(TEMPERATURE_COLUMN_LABEL), errors='coerce'
        )
        humidity_series = pd.to_numeric(
            measurement_table_full.get(HUMIDITY_COLUMN_LABEL), errors='coerce'
        )
        if temperature_series is not None and not temperature_series.dropna().empty:
            avg_temperature_value = float(temperature_series.mean())
        if humidity_series is not None and not humidity_series.dropna().empty:
            avg_humidity_value = float(humidity_series.mean())

    st.markdown("### Current status")
    _render_status_card(
        st,
        best_snapshot,
        last_fetch_display,
        len(snapshots),
        avg_temperature=avg_temperature_value,
        avg_humidity=avg_humidity_value,
    )

    metadata_df = load_station_metadata()
    summary_df = snapshots_to_aqi_frame(snapshots)
    valid_summary = summary_df[summary_df["aqi"].notna()].copy()

    metrics_payload: Dict[str, float] | None = None
    if not valid_summary.empty:
        metrics_payload = {
            "max_aqi": float(valid_summary["aqi"].max()),
            "avg_aqi": float(valid_summary["aqi"].mean()),
            "stations": float(valid_summary.shape[0]),
        }

    st.markdown("### Key metrics")
    if metrics_payload is None:
        st.warning("Stations responded without valid AQI values. Waiting for the next update cycle.")
    else:
        max_aqi = int(metrics_payload["max_aqi"])
        avg_aqi = round(metrics_payload["avg_aqi"], 1)
        reporting_stations = int(metrics_payload["stations"])
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Max AQI", max_aqi)
        kpi_col2.metric("Avg AQI", avg_aqi)
        kpi_col3.metric("Stations reporting", reporting_stations)

    st.markdown("<div id='map'></div>", unsafe_allow_html=True)
    st.markdown("### Interactive pollution map")
    st.markdown(
        "The map shows monitoring stations, the interpolated AQI surface, and boundaries of the study area.",
        unsafe_allow_html=False,
    )
    map_obj = _build_map(metadata_df, snapshots)
    st_folium(map_obj, use_container_width=True, height=640)

    st.markdown("### Latest measurements")
    st.markdown(
        "Per-station details: AQI value, dominant pollutant, concentration series, temperature, humidity, and measurement timestamp.",
    )
    if measurement_table_full.empty:
        st.info("Telemetry payloads do not contain pollutant concentrations. Waiting for the next refresh cycle.")
    else:
        measurement_table_display = measurement_table_full.head(5)
        styled_table = _style_measurement_table(measurement_table_display)
        st.dataframe(styled_table, use_container_width=True, height=212)

    st.markdown(
        """
        <footer style="text-align:center; font-size:14px; color:#888; margin-top:40px;">
            © 2025 Pavlodar Air Monitoring Project · Project Lead: R.Z. Safarov · L.N. Gumilyov Eurasian National University · Grant ID: AP19677560
        </footer>
        """,
        unsafe_allow_html=True,
    )


render_dashboard()
