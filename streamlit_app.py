#!/usr/bin/env python
"""
Streamlit UI for FloodRisk demo API

Features:
- Interactive map: click to set latitude/longitude
- Inputs for rainfall (mm) and optional elevation (m)
- Submit button to call FastAPI endpoint `/api/v1/predict`
- Modern, responsive layout with basic theming

Run with:
  streamlit run streamlit_app.py

Requires:
  pip install streamlit streamlit-folium requests folium
"""

import json
from typing import Optional, Dict, Any

import requests
import streamlit as st
from streamlit_folium import st_folium
import folium

API_URL = "http://localhost:8000/api/v1/predict"


def _init_state() -> None:
    """Initialize Streamlit session state defaults."""
    st.session_state.setdefault("latitude", 37.7749)   # Default: San Francisco
    st.session_state.setdefault("longitude", -122.4194)
    st.session_state.setdefault("rainfall_mm", 50.0)
    st.session_state.setdefault("elevation_m", None)


def _inject_css() -> None:
    """Inject minimal modern CSS to style the app."""
    st.markdown(
        """
        <style>
            .main > div {
                padding-top: 1rem;
            }
            .card {
                background: #ffffff;
                border-radius: 14px;
                border: 1px solid rgba(0,0,0,0.06);
                box-shadow: 0 6px 20px rgba(0,0,0,0.06);
                padding: 1.1rem 1.2rem;
            }
            .subtitle {
                color: #6b7280;
                font-size: 0.95rem;
                margin-top: -0.6rem;
                margin-bottom: 0.8rem;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
            }
            .metric-card {
                background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
                border: 1px solid rgba(0,0,0,0.06);
                border-radius: 12px;
                padding: 12px 14px;
            }
            .metric-title { color: #6b7280; font-size: 0.85rem; }
            .metric-value { color: #111827; font-size: 1.25rem; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _map(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Render a Folium map and return the latest click event payload, if any."""
    m = folium.Map(location=[lat, lon], zoom_start=8, control_scale=True, tiles="CartoDB positron")
    # Add an initial marker at the current coordinates
    folium.Marker([lat, lon], tooltip="Selected location").add_to(m)

    # Enable click to add marker behavior via JS
    m.add_child(folium.LatLngPopup())

    # Render in Streamlit and capture events
    return st_folium(m, width="100%", height=420)


def _submit_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit the POST request to the API with basic error handling."""
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.RequestException as e:
        details = None
        try:
            details = resp.json() if 'resp' in locals() and resp.content else None
        except Exception:
            details = None
        return {"ok": False, "error": str(e), "details": details}


def main() -> None:
    st.set_page_config(page_title="FloodRisk Demo UI", page_icon="🌊", layout="wide")
    _init_state()
    _inject_css()

    st.title("FloodRisk – Demo Predictor")
    st.markdown("<div class='subtitle'>Click on the map to select a location, then enter rainfall and optional elevation to predict flood depth.</div>", unsafe_allow_html=True)

    # Layout: map on top, inputs below
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        map_events = _map(st.session_state.latitude, st.session_state.longitude)

        # Handle clicks to update coordinates
        if map_events and map_events.get("last_clicked"):
            lat = map_events["last_clicked"].get("lat")
            lon = map_events["last_clicked"].get("lng")
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                st.session_state.latitude = float(lat)
                st.session_state.longitude = float(lon)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.number_input(
                "Latitude",
                value=float(st.session_state.latitude),
                key="latitude",
                help="Latitude in decimal degrees (-90 to 90)",
                format="%.6f",
            )
        with c2:
            st.number_input(
                "Longitude",
                value=float(st.session_state.longitude),
                key="longitude",
                help="Longitude in decimal degrees (-180 to 180)",
                format="%.6f",
            )
        with c3:
            st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=float(st.session_state.rainfall_mm),
                step=1.0,
                key="rainfall_mm",
                help="Total rainfall in millimeters",
            )

        c4, c5 = st.columns([1, 2])
        with c4:
            elevation_value = st.number_input(
                "Elevation (m) – optional",
                value=float(st.session_state.elevation_m) if st.session_state.elevation_m is not None else 0.0,
                step=1.0,
                help="Ground elevation at the location. Leave at 0 or blank if unknown.",
            )
            # Treat 0.0 as None for optional behavior
            elevation_payload = None if elevation_value == 0.0 else float(elevation_value)

        st.divider()
        submit = st.button("Predict Flood Risk", type="primary")

        if submit:
            payload = {
                "latitude": float(st.session_state.latitude),
                "longitude": float(st.session_state.longitude),
                "rainfall_mm": float(st.session_state.rainfall_mm),
            }
            if elevation_payload is not None:
                payload["elevation_m"] = elevation_payload

            with st.spinner("Contacting API…"):
                result = _submit_request(payload)

            if not result.get("ok"):
                st.error("Request failed: " + result.get("error", "Unknown error"))
                if result.get("details"):
                    st.code(json.dumps(result["details"], indent=2), language="json")
            else:
                data = result["data"]
                # Display results in metric cards
                st.markdown("<div class='metrics'>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Flood Depth (m)</div>
                        <div class='metric-value'>{data.get('flood_depth_m', '—')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Risk Level</div>
                        <div class='metric-value'>{data.get('risk_level', '—')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-title'>Confidence</div>
                        <div class='metric-value'>{round(float(data.get('confidence', 0))*100, 1)}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.caption(data.get("message", ""))
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
