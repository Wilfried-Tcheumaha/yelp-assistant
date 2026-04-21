"""Map view for the suggested restaurants — numbered red pins on a map."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st


def _items_to_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, it in enumerate(items, start=1):
        lat = it.get("latitude")
        lon = it.get("longitude")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        rows.append({
            "idx": idx,
            "label": str(idx),
            "name": it.get("name") or "",
            "address": it.get("address") or "",
            "lat": float(lat),
            "lon": float(lon),
        })
    return pd.DataFrame(rows)


def render_restaurants_map(items: list[dict[str, Any]]) -> None:
    """Plot suggested restaurants as numbered red pins on a pydeck map."""
    df = _items_to_frame(items)
    if df.empty:
        st.info("No location data for these suggestions.")
        return

    view = pdk.ViewState(
        latitude=float(df["lat"].mean()),
        longitude=float(df["lon"].mean()),
        zoom=11 if len(df) > 1 else 13,
        pitch=0,
    )

    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color=[211, 35, 35, 230],  # Yelp-ish red
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
        get_radius=120,
        radius_min_pixels=14,
        radius_max_pixels=20,
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=df,
        get_position="[lon, lat]",
        get_text="label",
        get_color=[255, 255, 255, 255],
        get_size=14,
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
        font_weight=700,
        pickable=False,
    )

    deck = pdk.Deck(
        layers=[pin_layer, text_layer],
        initial_view_state=view,
        tooltip={"text": "{idx}. {name}\n{address}"},
        map_style=None,  # default light basemap, no Mapbox token needed
    )
    st.pydeck_chart(deck, use_container_width=True)
