"""Visualisations cartographiques pour Data Tool Climatique.

On utilise les capacités de Streamlit / PyDeck pour afficher une carte
simple à partir de colonnes latitude / longitude et, optionnellement, une
variable de couleur (par exemple un score de risque).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st


def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Essaie de deviner les colonnes latitude / longitude.

    Recherche des variantes courantes de noms de colonnes.
    """

    candidates_lat = ["lat", "latitude", "LAT", "Latitude"]
    candidates_lon = ["lon", "lng", "longitude", "LONGITUDE", "Lon"]

    lat_col = next((c for c in candidates_lat if c in df.columns), None)
    lon_col = next((c for c in candidates_lon if c in df.columns), None)
    return lat_col, lon_col


def show_risk_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
    use_pydeck: bool = False,
) -> None:
    """Affiche une carte basée sur lat/lon.

    - Si `color_col` est fourni, elle est utilisée comme valeur de couleur / taille.
    - Si `use_pydeck` est True, utilise PyDeck pour une visualisation plus riche.
    """

    if lat_col not in df.columns or lon_col not in df.columns:
        st.warning("Colonnes latitude/longitude introuvables dans le DataFrame.")
        return

    map_df = df[[lat_col, lon_col]].copy()
    map_df = map_df.rename(columns={lat_col: "lat", lon_col: "lon"})

    if color_col and color_col in df.columns:
        map_df["risk_value"] = df[color_col]
        
        if use_pydeck:
            # Normaliser les valeurs pour la couleur (0-255)
            min_val = map_df["risk_value"].min()
            max_val = map_df["risk_value"].max()
            if max_val > min_val:
                map_df["color_norm"] = ((map_df["risk_value"] - min_val) / (max_val - min_val) * 255).astype(int)
            else:
                map_df["color_norm"] = 128
            
            # Créer une couche PyDeck avec gradient de couleur
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_color="[255 - color_norm, color_norm, 100, 160]",
                get_radius=5000,
                pickable=True,
            )
            
            view_state = pdk.ViewState(
                latitude=map_df["lat"].mean(),
                longitude=map_df["lon"].mean(),
                zoom=5,
                pitch=0,
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": f"Risque: {{risk_value:.2f}}"},
            )
            
            st.pydeck_chart(r)
        else:
            st.map(map_df, latitude="lat", longitude="lon")
        
        st.dataframe(df[[lat_col, lon_col, color_col]].head(), use_container_width=True)
    else:
        st.map(map_df, latitude="lat", longitude="lon")
        st.dataframe(df[[lat_col, lon_col]].head(), use_container_width=True)


def run_maps_page(df: pd.DataFrame, title: str = "🗺️ Cartographie du risque") -> None:
    """Page Streamlit de cartographie avancée.

    Laisse l'utilisateur :
    - choisir les colonnes latitude / longitude (avec détection automatique),
    - choisir une variable optionnelle pour colorer les points (score de risque),
    - activer PyDeck pour visualisation avancée,
    - filtrer par période temporelle si colonne date disponible.
    """

    if title:  # Afficher le titre seulement s'il est fourni
        st.header(title)

    if df.empty:
        st.warning("Le DataFrame fourni est vide.")
        return

    st.markdown("Sélectionnez les colonnes de localisation et, si souhaité, une variable de risque.")

    auto_lat, auto_lon = detect_lat_lon_columns(df)

    lat_col = st.selectbox("Colonne latitude", options=df.columns.tolist(), index=df.columns.get_loc(auto_lat) if auto_lat else 0)
    lon_col = st.selectbox("Colonne longitude", options=df.columns.tolist(), index=df.columns.get_loc(auto_lon) if auto_lon else 0)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    color_col: Optional[str] = None
    if numeric_cols:
        color_col = st.selectbox("Variable numérique pour le risque (optionnel)", options=["(aucune)"] + numeric_cols)
        if color_col == "(aucune)":
            color_col = None
    
    st.markdown("---")
    st.subheader("⚙️ Options avancées")
    
    use_pydeck = st.checkbox("Utiliser PyDeck (visualisation avancée avec gradient)", value=False)
    
    # Filtre temporel si colonne date disponible
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    date_col_filter = None
    df_filtered = df.copy()
    
    if date_cols:
        st.markdown("**Filtre temporel (optionnel)**")
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

    if st.button("Afficher la carte"):
        show_risk_map(df_filtered, lat_col=lat_col, lon_col=lon_col, color_col=color_col, use_pydeck=use_pydeck)
