"""
Module de visualisation avancée pour l'analyse des risques climatiques.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import pandas as pd
import numpy as np
import json

class RiskVisualizer:
    """
    Classe pour la visualisation des risques climatiques et des données d'assurance.
    """
    
    def __init__(self, mapbox_token: Optional[str] = None):
        """
        Initialise le visualiseur avec un token Mapbox optionnel.
        
        Args:
            mapbox_token: Token d'accès à l'API Mapbox (optionnel)
        """
        self.mapbox_token = mapbox_token
        if mapbox_token:
            px.set_mapbox_access_token(mapbox_token)
    
    def plot_risk_heatmap(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        title: str = "Carte de chaleur des risques",
        zoom: float = 10,
        opacity: float = 0.7,
        radius: int = 15,
        mapbox_style: str = "carto-positron",
        **kwargs
    ) -> go.Figure:
        """
        Affiche une carte de chaleur des risques avec Mapbox.
        
        Args:
            gdf: GeoDataFrame contenant les données
            value_col: Nom de la colonne à visualiser
            lat_col: Nom de la colonne de latitude
            lon_col: Nom de la colonne de longitude
            title: Titre de la carte
            zoom: Niveau de zoom initial
            opacity: Opacité des points de chaleur
            radius: Rayon des points de chaleur
            mapbox_style: Style de carte Mapbox
            **kwargs: Arguments supplémentaires pour px.density_mapbox
            
        Returns:
            Figure Plotly
        """
        if not all(col in gdf.columns for col in [lat_col, lon_col, value_col]):
            raise ValueError("Les colonnes de latitude, longitude et valeur sont requises.")
        
        # Créer une copie pour éviter les modifications sur l'original
        df = gdf.copy()
        
        # Si c'est un GeoDataFrame, extraire les coordonnées si nécessaire
        if isinstance(df, gpd.GeoDataFrame):
            if lat_col not in df.columns or lon_col not in df.columns:
                df[lon_col] = df.geometry.x
                df[lat_col] = df.geometry.y
        
        # Créer la carte de chaleur
        fig = px.density_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            z=value_col,
            radius=radius,
            zoom=zoom,
            opacity=opacity,
            mapbox_style=mapbox_style,
            title=title,
            **kwargs
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title=value_col,
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=300,
                yanchor="top",
                y=1,
                x=0,
                ticks="outside"
            )
        )
        
        return fig
    
    def plot_risk_comparison(
        self,
        gdf: gpd.GeoDataFrame,
        current_risk: str,
        future_risk: str,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        title: str = "Comparaison des risques actuels et futurs",
        mapbox_style: str = "carto-positron",
        **kwargs
    ) -> go.Figure:
        """
        Affiche une comparaison côte à côte des risques actuels et futurs.
        
        Args:
            gdf: GeoDataFrame contenant les données
            current_risk: Nom de la colonne de risque actuel
            future_risk: Nom de la colonne de risque futur
            lat_col: Nom de la colonne de latitude
            lon_col: Nom de la colonne de longitude
            title: Titre de la figure
            mapbox_style: Style de carte Mapbox
            **kwargs: Arguments supplémentaires pour make_subplots
            
        Returns:
            Figure Plotly avec deux sous-graphiques
        """
        required_cols = [lat_col, lon_col, current_risk, future_risk]
        if not all(col in gdf.columns for col in required_cols):
            missing = [col for col in required_cols if col not in gdf.columns]
            raise ValueError(f"Colonnes manquantes: {', '.join(missing)}")
        
        # Créer une copie pour éviter les modifications sur l'original
        df = gdf.copy()
        
        # Si c'est un GeoDataFrame, extraire les coordonnées si nécessaire
        if isinstance(df, gpd.GeoDataFrame):
            if lat_col not in df.columns or lon_col not in df.columns:
                df[lon_col] = df.geometry.x
                df[lat_col] = df.geometry.y
        
        # Créer une figure avec deux sous-graphiques
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Risque actuel", "Risque futur"),
            **kwargs
        )
        
        # Carte du risque actuel
        fig1 = px.scatter_mapbox(
            df, 
            lat=lat_col, 
            lon=lon_col, 
            color=current_risk,
            color_continuous_scale=px.colors.sequential.Viridis,
            mapbox_style=mapbox_style,
            zoom=10
        )
        
        # Carte du risque futur
        fig2 = px.scatter_mapbox(
            df, 
            lat=lat_col, 
            lon=lon_col, 
            color=future_risk,
            color_continuous_scale=px.colors.sequential.Viridis,
            mapbox_style=mapbox_style,
            zoom=10
        )
        
        # Ajouter les deux cartes à la figure
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Mettre à jour la mise en page
        fig.update_layout(
            title_text=title,
            showlegend=False,
            mapbox={
                'style': mapbox_style,
                'zoom': 9,
                'center': {
                    'lat': df[lat_col].mean(),
                    'lon': df[lon_col].mean()
                }
            },
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        
        # Mettre à jour les cartes pour utiliser le même zoom et centre
        fig.update_geos(fitbounds="locations")
        
        return fig
    
    def plot_time_series(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: List[str],
        title: str = "Série temporelle des risques",
        height: int = 500,
        width: int = 900,
        **kwargs
    ) -> go.Figure:
        """
        Affiche une série temporelle interactive des risques.
        
        Args:
            df: DataFrame contenant les données
            time_col: Nom de la colonne de temps
            value_cols: Liste des colonnes de valeurs à afficher
            title: Titre du graphique
            height: Hauteur de la figure
            width: Largeur de la figure
            **kwargs: Arguments supplémentaires pour px.line
            
        Returns:
            Figure Plotly
        """
        required_cols = [time_col] + value_cols
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Colonnes manquantes: {', '.join(missing)}")
        
        # Créer la figure
        fig = px.line(
            df, 
            x=time_col, 
            y=value_cols,
            title=title,
            height=height,
            width=width,
            **kwargs
        )
        
        # Personnaliser la mise en page
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Valeur",
            legend_title="Variables",
            hovermode="x unified"
        )
        
        return fig
    
    def plot_damage_curve(
        self,
        hazard_intensity: np.ndarray,
        damage_ratio: np.ndarray,
        title: str = "Courbe de dommage",
        x_label: str = "Intensité de l'aléa",
        y_label: str = "Taux de dommage (0-1)",
        height: int = 500,
        width: int = 800
    ) -> go.Figure:
        """
        Affiche une courbe de dommage (fonction de vulnérabilité).
        
        Args:
            hazard_intensity: Tableau des valeurs d'intensité de l'aléa
            damage_ratio: Tableau des taux de dommage correspondants (0-1)
            title: Titre du graphique
            x_label: Étiquette de l'axe des x
            y_label: Étiquette de l'axe des y
            height: Hauteur de la figure
            width: Largeur de la figure
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Ajouter la courbe de dommage
        fig.add_trace(
            go.Scatter(
                x=hazard_intensity,
                y=damage_ratio,
                mode='lines+markers',
                name='Courbe de dommage',
                line=dict(color='royalblue', width=2),
                marker=dict(size=8, color='royalblue')
            )
        )
        
        # Mise en forme
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=height,
            width=width,
            showlegend=True,
            template="plotly_white"
        )
        
        # Ajouter des lignes de grille
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        
        return fig
