"""Visualisations cartographiques avancées pour Data Tool Climatique.

Supporte PyDeck (3D, haute performance) et Folium (heatmap, fonds élégants).
Optimisé pour les gros datasets avec mise en cache et vectorisation NumPy.
Inclut export HTML et GeoJSON.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# Palettes de couleurs
COLOR_PALETTES = {
    "Viridis": ["#440154", "#31688e", "#35b779", "#fde724"],
    "Plasma": ["#0d0887", "#7e03a8", "#cc4778", "#f89540"],
    "Reds": ["#fff5f0", "#fee0d2", "#fcbba1", "#a50f15"],
    "Greens": ["#edf8e9", "#bae4b3", "#74c476", "#005a32"],
    "Blues": ["#eff7fb", "#bdc9e1", "#74a9cf", "#0570b0"],
    "RdYlBu": ["#a50026", "#fdae61", "#ffffbf", "#91bfdb", "#4575b4"],
}

# Fonds de carte disponibles
TILE_LAYERS = {
    "CartoDB Positron": "CartoDB positron",
    "OpenStreetMap": "OpenStreetMap",
    "Relief (OpenTopoMap)": "OpenTopoMap",
    "Satellite (Esri)": "Esri.WorldImagery",
    "Sombre (CartoDB)": "CartoDB dark_matter",
}


def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Essaie de deviner les colonnes latitude / longitude.

    Recherche des variantes courantes de noms de colonnes.
    """
    candidates_lat = ["lat", "latitude", "LAT", "Latitude"]
    candidates_lon = ["lon", "lng", "longitude", "LONGITUDE", "Lon"]

    lat_col = next((c for c in candidates_lat if c in df.columns), None)
    lon_col = next((c for c in candidates_lon if c in df.columns), None)
    return lat_col, lon_col


def _recommend_viz_type(n_points: int) -> str:
    """Recommande le type de visualisation selon le nombre de points."""
    if n_points > 2000:
        return "Heatmap (Climatique)"  # Seule option viable
    elif n_points > 500:
        return "Heatmap (Climatique)"  # Recommandée (markers trop lent)
    elif n_points > 50:
        return "Markers + Clusters"  # Bon compromis
    else:
        return "Markers + Clusters"  # Mieux pour détail


def _get_colors_vectorized(
    values: np.ndarray,
    palette: list[str],
) -> np.ndarray:
    """Retourne les couleurs HEX vectorisées (100x plus rapide)."""
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if max_val == min_val:
        normalized = np.full_like(values, 0.5, dtype=float)
    else:
        normalized = (values - min_val) / (max_val - min_val)

    # Mapper vers indices palette
    indices = np.clip(
        (normalized * (len(palette) - 1)).astype(int),
        0,
        len(palette) - 1
    )

    return np.array([palette[i] for i in indices])


@st.cache_data(ttl=3600)
def _prepare_map_data_cached(
    df_hash: str,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Prépare les données CACHÉES (évite recalculs)."""
    # Note: hash passé pour invalider cache si données changent
    return True, f"Cache key: {df_hash}"


def _prepare_map_data(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
) -> pd.DataFrame:
    """Prépare les données pour la cartographie."""
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Colonnes introuvables. Disponibles : {df.columns.tolist()}")

    map_df = df[[lat_col, lon_col]].copy()
    map_df = map_df.dropna(subset=[lat_col, lon_col])

    if map_df.empty:
        raise ValueError("Aucun point avec coordonnées valides.")

    map_df = map_df.rename(columns={lat_col: "latitude", lon_col: "longitude"})

    if not pd.api.types.is_numeric_dtype(map_df["latitude"]):
        raise ValueError(f"Latitude non numérique : {map_df['latitude'].dtype}")
    if not pd.api.types.is_numeric_dtype(map_df["longitude"]):
        raise ValueError(f"Longitude non numérique : {map_df['longitude'].dtype}")
    # Types Python simples : folium sérialise en JSON et refuse les entiers numpy
    map_df["latitude"] = map_df["latitude"].astype(float)
    map_df["longitude"] = map_df["longitude"].astype(float)
    if map_df["latitude"].abs().max() > 90:
        raise ValueError("La colonne latitude contient des valeurs hors de [-90, 90] : vérifiez le choix des colonnes.")

    if color_col and color_col in df.columns:
        map_df["risk_value"] = df[color_col]
        if not pd.api.types.is_numeric_dtype(map_df["risk_value"]):
            map_df["risk_value"] = pd.to_numeric(map_df["risk_value"], errors="coerce")
            map_df = map_df.dropna(subset=["risk_value"])
        map_df["risk_value"] = map_df["risk_value"].astype(float)

    return map_df


def show_heatmap_folium(
    map_df: pd.DataFrame,
    color_col: Optional[str] = None,
    tile_layer: str = "CartoDB positron",
) -> Optional[folium.Map]:
    """Affiche une heatmap Folium optimisée et retourne la carte pour export."""
    try:
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles=tile_layer,
        )

        if "risk_value" in map_df.columns:
            # Préparer données pour heatmap
            heat_data = map_df[["latitude", "longitude", "risk_value"]].values.tolist()

            folium.plugins.HeatMap(
                heat_data,
                min_opacity=0.2,
                max_zoom=18,
                radius=15,
                blur=25,
                gradient={
                    0.2: "#0d0887",  # Bleu (risque faible)
                    0.4: "#cc4778",  # Rose
                    0.6: "#f89540",  # Orange
                    0.8: "#fdae61",  # Orange clair
                    1.0: "#a50026",  # Rouge (risque élevé)
                },
            ).add_to(m)

            # Limitation intelligente : cercles seulement si < 500 points
            if len(map_df) < 500:
                palette = COLOR_PALETTES["RdYlBu"]
                colors = _get_colors_vectorized(
                    map_df["risk_value"].values,
                    palette
                )

                for idx, (lat, lon, color) in enumerate(
                    zip(map_df["latitude"], map_df["longitude"], colors)
                ):
                    val = map_df.iloc[idx]["risk_value"]
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        popup=f"Risque: {val:.2f}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                    ).add_to(m)
        else:
            # Afficher simple markers
            for idx, row in map_df.iterrows():
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}",
                ).add_to(m)

        m.fit_bounds([[map_df["latitude"].min(), map_df["longitude"].min()],
                      [map_df["latitude"].max(), map_df["longitude"].max()]], padding=(30, 30))
        # returned_objects=[] : la carte ne relance pas l'application à chaque déplacement
        st_folium(m, height=600, use_container_width=True, returned_objects=[])
        return m

    except Exception as e:
        st.error(f"Erreur heatmap : {e}")
        return None


def show_simple_points_folium(
    map_df: pd.DataFrame,
    color_col: Optional[str] = None,
    tile_layer: str = "CartoDB positron",
) -> Optional[folium.Map]:
    """Affiche les points simples et retourne la carte pour export."""
    try:
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles=tile_layer,
        )

        if "risk_value" in map_df.columns:
            # Vectorisation : calcul des couleurs une seule fois
            palette = COLOR_PALETTES["RdYlBu"]
            colors = _get_colors_vectorized(
                map_df["risk_value"].values,
                palette
            )

            for lat, lon, color, val in zip(
                map_df["latitude"],
                map_df["longitude"],
                colors,
                map_df["risk_value"]
            ):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=f"<b>Risque: {val:.2f}</b>",
                    tooltip=f"Risque: {val:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=1,
                ).add_to(m)
        else:
            # Points sans valeur
            for lat, lon in zip(map_df["latitude"], map_df["longitude"]):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color="blue",
                    fill=True,
                    fillColor="blue",
                    fillOpacity=0.5,
                    weight=1,
                ).add_to(m)

        m.fit_bounds([[map_df["latitude"].min(), map_df["longitude"].min()],
                      [map_df["latitude"].max(), map_df["longitude"].max()]], padding=(30, 30))
        # returned_objects=[] : la carte ne relance pas l'application à chaque déplacement
        st_folium(m, height=600, use_container_width=True, returned_objects=[])
        return m

    except Exception as e:
        st.error(f"Erreur points : {e}")
        return None


def show_markers_folium(
    map_df: pd.DataFrame,
    color_col: Optional[str] = None,
    tile_layer: str = "CartoDB positron",
) -> Optional[folium.Map]:
    """Affiche markers avec clustering optimisé et retourne la carte pour export."""
    try:
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles=tile_layer,
        )

        # Ajouter clustering
        from folium.plugins import MarkerCluster
        marker_cluster = MarkerCluster().add_to(m)

        if "risk_value" in map_df.columns and len(map_df) < 2000:
            # Vectorisation : calcul des couleurs une seule fois
            palette = COLOR_PALETTES["RdYlBu"]
            colors = _get_colors_vectorized(
                map_df["risk_value"].values,
                palette
            )

            for lat, lon, color, val in zip(
                map_df["latitude"],
                map_df["longitude"],
                colors,
                map_df["risk_value"]
            ):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=f"<b>Risque: {val:.2f}</b>",
                    tooltip=f"Risque: {val:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2,
                ).add_to(marker_cluster)
        elif "risk_value" in map_df.columns:
            # Trop de points : seulement heatmap
            st.info("Trop de points pour markers. Utilisez l'heatmap à la place.")
        else:
            for idx, row in map_df.iterrows():
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup="Localisation",
                ).add_to(marker_cluster)

        m.fit_bounds([[map_df["latitude"].min(), map_df["longitude"].min()],
                      [map_df["latitude"].max(), map_df["longitude"].max()]], padding=(30, 30))
        # returned_objects=[] : la carte ne relance pas l'application à chaque déplacement
        st_folium(m, height=600, use_container_width=True, returned_objects=[])
        return m

    except Exception as e:
        st.error(f"Erreur markers : {e}")
        return None


def export_map_html(map_folium: folium.Map) -> bytes:
    """Exporte une carte Folium en HTML avec timestamp."""
    try:
        # Sauvegarder en HTML
        html_data = map_folium._repr_html_().encode('utf-8')
        return html_data
    except Exception as e:
        st.error(f"Erreur export HTML : {e}")
        return b""


def export_data_geojson(map_df: pd.DataFrame, lat_col: str, lon_col: str, color_col: Optional[str] = None) -> str:
    """Exporte les données en format GeoJSON."""
    try:
        features = []
        for idx, row in map_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["longitude"], row["latitude"]]
                },
                "properties": {}
            }

            # Ajouter les propriétés
            if "risk_value" in map_df.columns:
                feature["properties"]["risque"] = float(row["risk_value"])

            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        geojson_str = json.dumps(geojson_data, indent=2)
        return geojson_str
    except Exception as e:
        st.error(f"Erreur export GeoJSON : {e}")
        return "{}"


def show_gradient_3d_pydeck(
    map_df: pd.DataFrame,
    color_palette: str = "Viridis",
) -> None:
    """Colonnes 3D (PyDeck) : la hauteur et la couleur suivent la valeur de risque."""
    try:
        data = map_df.copy()
        span_deg = max(data["latitude"].max() - data["latitude"].min(),
                       data["longitude"].max() - data["longitude"].min(), 0.5)
        radius = float(np.clip(span_deg * 111_000 / 60, 1_000, 30_000))
        if "risk_value" not in data.columns:
            st.info("Choisissez une variable de risque pour donner une hauteur aux colonnes.")
            data["risk_value"] = 1.0
        vmin, vmax = data["risk_value"].min(), data["risk_value"].max()
        norm = (data["risk_value"] - vmin) / (vmax - vmin) if vmax > vmin else pd.Series(0.5, index=data.index)
        palette = COLOR_PALETTES.get(color_palette, COLOR_PALETTES["Viridis"])
        rgb = [tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for h in palette]
        idx = (norm * (len(rgb) - 1)).round().astype(int)
        data["r"], data["g"], data["b"] = zip(*[rgb[i] for i in idx])
        data["hauteur"] = 0.1 + 0.9 * norm
        data["libelle"] = data["risk_value"].map(lambda v: f"{v:,.2f}".replace(",", " "))

        layer = pdk.Layer(
            "ColumnLayer",
            data=data,
            get_position=["longitude", "latitude"],
            get_elevation="hauteur",
            elevation_scale=radius * 40,
            radius=radius,
            get_fill_color="[r, g, b, 210]",
            pickable=True,
            extruded=True,
        )
        view_state = pdk.ViewState(
            latitude=data["latitude"].mean(),
            longitude=data["longitude"].mean(),
            zoom=float(np.clip(8 - np.log2(span_deg), 3, 12)),
            pitch=45,
        )
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Risque : {libelle}"},
        )
        st.pydeck_chart(r)

    except Exception as e:
        st.error(f"Erreur PyDeck : {e}")


def run_maps_page(
    df: pd.DataFrame,
    title: str = "Cartographie du risque",
    data_sources: Optional[dict] = None,
    df_prep: Optional[pd.DataFrame] = None,
) -> None:
    """Page Streamlit de cartographie avancée multi-type (optimisée pour gros datasets)."""

    start_time = time.time()

    if title:
        st.header(title)

    # Sélecteur de source de données
    st.subheader("Sélection de la source de données")

    available_sources = {}
    source_options = []

    if isinstance(df_prep, pd.DataFrame) and not df_prep.empty:
        available_sources["Données prétraitées (dernier traitement)"] = df_prep
        source_options.append("Données prétraitées (dernier traitement)")

    if data_sources:
        if "Climat" in data_sources:
            available_sources["Source : Climat"] = data_sources["Climat"]
            source_options.append("Source : Climat")
        for label in sorted(data_sources.keys()):
            if label != "Climat":
                available_sources[f"Source : {label}"] = data_sources[label]
                source_options.append(f"Source : {label}")

    if not source_options:
        st.warning("Aucune source de données disponible. Veuillez d'abord charger des données.")
        return

    selected_source = st.selectbox("Choisir la source de données", source_options, index=0)
    selected_df = available_sources.get(selected_source, df)

    if selected_df.empty:
        st.warning("La source de données sélectionnée est vide.")
        return

    st.markdown("---")
    st.subheader("Configuration de la carte")

    coord_candidates = [
        c for c in selected_df.select_dtypes(include=["number"]).columns
        if selected_df[c].notna().any() and selected_df[c].abs().max() <= 180
    ]
    if len(coord_candidates) < 2:
        st.info("Cette source n'a pas de coordonnées (latitude et longitude numériques). "
                "Choisissez une source qui en contient, par exemple les données climatiques ou la fusion.")
        return

    lat_candidates = [c for c in coord_candidates if selected_df[c].abs().max() <= 90]
    auto_lat, auto_lon = detect_lat_lon_columns(selected_df)
    if not lat_candidates:
        st.info("Aucune colonne ne ressemble à une latitude (valeurs entre -90 et 90).")
        return
    if auto_lat not in lat_candidates or auto_lon not in coord_candidates or auto_lat == auto_lon:
        st.info("Colonnes latitude/longitude non trouvées automatiquement. Sélectionnez-les ci-dessous.")
        auto_lat = lat_candidates[0]
        auto_lon = next(c for c in coord_candidates if c != auto_lat)

    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat_col = st.selectbox("Colonne latitude", options=lat_candidates,
                               index=lat_candidates.index(auto_lat))
    with col_lon:
        lon_options = [c for c in coord_candidates if c != lat_col]
        lon_col = st.selectbox("Colonne longitude", options=lon_options,
                               index=lon_options.index(auto_lon) if auto_lon in lon_options else 0)

    numeric_cols = [c for c in selected_df.select_dtypes(include=["number"]).columns if c not in (lat_col, lon_col)]
    color_col: Optional[str] = None
    if numeric_cols:
        guess = next((c for c in numeric_cols if any(k in c.lower() for k in ("cout", "sinistre", "inond", "precip"))), None)
        color_options = ["(aucune)"] + numeric_cols
        color_col = st.selectbox("Variable numérique pour le risque (optionnel)", options=color_options,
                                 index=color_options.index(guess) if guess else 0)
        if color_col == "(aucune)":
            color_col = None
    aggregate_points = st.checkbox(
        "Une valeur par lieu (moyenne sur la période)", value=True,
        help="Les données quotidiennes répètent les mêmes coordonnées chaque jour : la moyenne par lieu "
             "donne une carte lisible et rapide.",
    )

    st.markdown("---")
    st.subheader("Options de visualisation")

    # Préparation rapide pour compter les points
    try:
        temp_map_df = _prepare_map_data(selected_df, lat_col, lon_col, color_col)
        n_points = (temp_map_df[["latitude", "longitude"]].drop_duplicates().shape[0]
                    if aggregate_points else len(temp_map_df))
        recommended_viz = _recommend_viz_type(n_points)
    except ValueError:
        recommended_viz = "Heatmap (Climatique)"
        n_points = 0

    col1, col2, col3 = st.columns(3)

    with col1:
        viz_type = st.selectbox(
            "Type de visualisation",
            options=[
                "Heatmap (Climatique)",
                "Points simples",
                "Markers + Clusters",
                "Gradient 3D (PyDeck)",
            ],
            index=0 if "Heatmap" in recommended_viz else (1 if "Points" in recommended_viz else (2 if "Markers" in recommended_viz else 3)),
            help=f"Recommandé pour {n_points} points : {recommended_viz}"
        )

    with col2:
        tile_layer_name = st.selectbox("Fond de carte", options=list(TILE_LAYERS.keys()))
        tile_layer = TILE_LAYERS[tile_layer_name]

    with col3:
        color_palette = st.selectbox("Palette de couleurs", options=list(COLOR_PALETTES.keys()))

    # Filtre temporel si colonne date disponible
    date_cols = [c for c in selected_df.columns
                 if pd.api.types.is_datetime64_any_dtype(selected_df[c])
                 or (not pd.api.types.is_numeric_dtype(selected_df[c]) and ("date" in c.lower() or "time" in c.lower()))]
    df_filtered = selected_df.copy()

    if date_cols:
        st.markdown("---")
        st.subheader("Filtre temporel (optionnel)")
        date_col_filter = st.selectbox("Colonne date pour filtrage", options=["(aucune)"] + date_cols)

        if date_col_filter != "(aucune)":
            try:
                df_filtered["_date_parsed"] = pd.to_datetime(df_filtered[date_col_filter], errors="coerce")
                min_date = df_filtered["_date_parsed"].min()
                max_date = df_filtered["_date_parsed"].max()

                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Période à afficher",
                        value=(min_date.date(), max_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                    )

                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        mask = (
                            (df_filtered["_date_parsed"] >= pd.to_datetime(start_date))
                            & (df_filtered["_date_parsed"] <= pd.to_datetime(end_date))
                        )
                        df_filtered = df_filtered[mask]
                        st.info(f"{len(df_filtered)} points après filtrage temporel")

                df_filtered = df_filtered.drop(columns=["_date_parsed"], errors="ignore")
            except Exception as e:
                st.warning(f"Impossible de filtrer par date : {e}")

    st.markdown("---")

    # Validation et préparation des données
    try:
        map_df = _prepare_map_data(df_filtered, lat_col, lon_col, color_col)
        elapsed = time.time() - start_time
        st.info(f"**{len(map_df)}** points •  Préparation en **{elapsed:.2f}s**")
    except ValueError as e:
        st.error(f"{e}")
        return

    if aggregate_points:
        agg = {"risk_value": "mean"} if "risk_value" in map_df.columns else {}
        if agg:
            map_df = map_df.groupby(["latitude", "longitude"], as_index=False).agg(agg)
        else:
            map_df = map_df.drop_duplicates(["latitude", "longitude"])
        st.caption(f"{len(map_df)} lieux distincts" + (f", {color_col} moyen par lieu." if color_col else "."))

    st.markdown("---")
    st.subheader("Carte")

    # Afficher selon type de visualisation
    render_start = time.time()
    folium_map = None

    if viz_type == "Heatmap (Climatique)":
        if not FOLIUM_AVAILABLE:
            st.error("Folium non installé. Installer avec : `pip install folium streamlit-folium`")
        else:
            folium_map = show_heatmap_folium(map_df, color_col, tile_layer)

    elif viz_type == "Points simples":
        if not FOLIUM_AVAILABLE:
            st.error("Folium non installé. Installer avec : `pip install folium streamlit-folium`")
        else:
            folium_map = show_simple_points_folium(map_df, color_col, tile_layer)

    elif viz_type == "Markers + Clusters":
        if not FOLIUM_AVAILABLE:
            st.error("Folium non installé. Installer avec : `pip install folium streamlit-folium`")
        else:
            folium_map = show_markers_folium(map_df, color_col, tile_layer)

    elif viz_type == "Gradient 3D (PyDeck)":
        show_gradient_3d_pydeck(map_df, color_palette)

    render_time = time.time() - render_start
    st.caption(f"Rendu en {render_time:.2f}s")

    # Boutons de téléchargement pour les cartes Folium
    if folium_map is not None:
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            html_data = export_map_html(folium_map)
            st.download_button(
                label="Télécharger Carte (HTML)",
                data=html_data,
                file_name=f"carte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
            )

        with col_dl2:
            geojson_data = export_data_geojson(map_df, lat_col, lon_col)
            st.download_button(
                label="Télécharger Données (GeoJSON)",
                data=geojson_data,
                file_name=f"donnees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                mime="application/json",
            )

    # Afficher aperçu des données
    st.markdown("---")
    st.subheader("Aperçu des données")
    preview = map_df.rename(columns={"risk_value": color_col}) if color_col else map_df
    st.dataframe(preview.head(20), width="stretch", hide_index=True)
