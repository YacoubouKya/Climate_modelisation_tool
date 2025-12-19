"""
Module de visualisation cartographique pour l'analyse des risques climatiques.

Fournit des fonctions pour créer des cartes interactives avec différentes représentations.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

# Détection automatique des colonnes de coordonnées
def detect_lat_lon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Détecte automatiquement les noms de colonnes pour la latitude et la longitude.
    
    Args:
        df: DataFrame contenant les données
        
    Returns:
        Tuple contenant (nom_colonne_latitude, nom_colonne_longitude) ou (None, None) si non trouvés
    """
    candidates_lat = ["lat", "latitude", "LAT", "Latitude"]
    candidates_lon = ["lon", "lng", "longitude", "LONGITUDE", "Lon"]

    lat_col = next((c for c in candidates_lat if c in df.columns), None)
    lon_col = next((c for c in candidates_lon if c in df.columns), None)
    return lat_col, lon_col

def create_map(
    gdf: gpd.GeoDataFrame,
    value_col: Optional[str] = None,
    map_type: str = "points",
    **kwargs
) -> pdk.Deck:
    """
    Crée une visualisation cartographique interactive.
    
    Args:
        gdf: GeoDataFrame contenant les données géographiques
        value_col: Colonne à utiliser pour la coloration/échelle
        map_type: Type de visualisation ('points', 'heatmap', 'cluster')
        **kwargs: Arguments supplémentaires pour la personnalisation
        
    Returns:
        Objet PyDeck pour affichage
    """
    # Vérifier si c'est un GeoDataFrame et extraire les coordonnées si nécessaire
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("L'entrée doit être un GeoDataFrame")
    
    # Convertir en WGS84 si nécessaire
    if gdf.crs and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Extraire les coordonnées
    gdf = gdf.copy()
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y
    
    # Créer la carte selon le type demandé
    if map_type == "points":
        return _create_point_map(gdf, value_col, **kwargs)
    elif map_type == "heatmap":
        return _create_heatmap(gdf, value_col, **kwargs)
    elif map_type == "cluster":
        return _create_cluster_map(gdf, value_col, **kwargs)
    else:
        raise ValueError(f"Type de carte non supporté: {map_type}")

def _create_point_map(
    gdf: gpd.GeoDataFrame,
    value_col: Optional[str] = None,
    **kwargs
) -> pdk.Deck:
    """Crée une carte avec des points."""
    # Configuration de la vue initiale
    view_state = pdk.ViewState(
        latitude=gdf['latitude'].mean(),
        longitude=gdf['longitude'].mean(),
        zoom=5,
        pitch=0,
    )
    
    # Couche de points
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=gdf,
        get_position=['longitude', 'latitude'],
        get_radius=100,
        get_fill_color=[255, 0, 0, 160],
        pickable=True,
        auto_highlight=True,
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": f"Valeur: {{{value_col}}}"} if value_col else None
    )

def _create_heatmap(
    gdf: gpd.GeoDataFrame,
    value_col: Optional[str] = None,
    **kwargs
) -> pdk.Deck:
    """Crée une carte de chaleur."""
    # Configuration de la vue initiale
    view_state = pdk.ViewState(
        latitude=gdf['latitude'].mean(),
        longitude=gdf['longitude'].mean(),
        zoom=5,
        pitch=0,
    )
    
    # Couche de heatmap
    layer = pdk.Layer(
        'HeatmapLayer',
        data=gdf,
        get_position=['longitude', 'latitude'],
        get_weight=value_col or 1,
        radius_pixels=30,
        intensity=1,
        threshold=0.1,
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state
    )

def _create_cluster_map(
    gdf: gpd.GeoDataFrame,
    value_col: Optional[str] = None,
    n_clusters: int = 5,
    **kwargs
) -> pdk.Deck:
    """Crée une carte avec des clusters."""
    # Appliquer le clustering K-means
    coords = gdf[['longitude', 'latitude']].values
    kmeans = KMeans(n_clusters=min(n_clusters, len(coords)), random_state=42)
    gdf['cluster'] = kmeans.fit_predict(coords)
    
    # Calculer le centre et la taille de chaque cluster
    clusters = gdf.groupby('cluster').agg({
        'longitude': 'mean',
        'latitude': 'mean',
        'geometry': 'count'
    }).rename(columns={'geometry': 'size'}).reset_index()
    
    # Configuration de la vue initiale
    view_state = pdk.ViewState(
        latitude=clusters['latitude'].mean(),
        longitude=clusters['longitude'].mean(),
        zoom=5,
        pitch=0,
    )
    
    # Couche de clusters
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=clusters,
        get_position=['longitude', 'latitude'],
        get_radius='size*100',
        get_fill_color=[255, 0, 0, 160],
        pickable=True,
        auto_highlight=True,
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Taille du cluster: {size}"}
    )

def show_risk_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None,
    use_pydeck: bool = False,
) -> None:
    """
    Affiche une carte de risque à partir de coordonnées.
    
    Args:
        df: DataFrame contenant les données
        lat_col: Nom de la colonne de latitude
        lon_col: Nom de la colonne de longitude
        color_col: Colonne optionnelle pour la couleur des points
        use_pydeck: Si True, utilise PyDeck pour une visualisation plus riche
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        st.warning("Colonnes latitude/longitude introuvables dans le DataFrame.")
        return

    if use_pydeck:
        # Utiliser PyDeck pour une visualisation plus riche
        st.pydeck_chart(create_map(df, value_col=color_col))
    else:
        # Utiliser la carte simple de Streamlit
        st.map(df.rename(columns={lat_col: "lat", lon_col: "lon"}))

def run_maps_page(df: pd.DataFrame, title: str = "Carte des risques") -> None:
    """
    Affiche une page complète de cartographie interactive.
    
    Args:
        df: DataFrame contenant les données
        title: Titre de la page
    """
    st.header(title)
    
    # Détection automatique des colonnes
    lat_col, lon_col = detect_lat_lon_columns(df)
    
    # Sélection des colonnes
    col1, col2 = st.columns(2)
    with col1:
        lat_col = st.selectbox(
            "Colonne de latitude",
            options=df.columns,
            index=df.columns.get_loc(lat_col) if lat_col else 0
        )
    with col2:
        lon_col = st.selectbox(
            "Colonne de longitude",
            options=df.columns,
            index=df.columns.get_loc(lon_col) if lon_col else 0
        )
    
    # Options d'affichage
    color_col = st.selectbox(
        "Colonne pour la couleur des points (optionnel)",
        [""] + df.select_dtypes(include='number').columns.tolist(),
        format_func=lambda x: "Aucune" if x == "" else x
    )
    
    # Type de visualisation
    map_type = st.selectbox(
        "Type de visualisation",
        ["Points", "Heatmap", "Cluster"]
    ).lower()
    
    # Afficher la carte
    try:
        # Créer un GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col])
        )
        
        # Afficher la carte
        st.pydeck_chart(create_map(gdf, value_col=color_col, map_type=map_type))
        
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte : {e}")
        st.exception(e)
