"""
Module de base pour le traitement des données géospatiales.

Fournit la classe GeoProcessor pour les opérations géospatiales fondamentales.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from shapely.geometry import Point, Polygon, shape
import warnings

# Désactiver les avertissements
warnings.filterwarnings('ignore')

class GeoProcessor:
    """
    Classe pour le traitement des données géospatiales pour l'analyse des risques climatiques.
    """
    
    def __init__(self, crs: str = "EPSG:4326"):
        """
        Initialise le processeur géospatial.
        
        Args:
            crs: Système de coordonnées de référence (par défaut: EPSG:4326 - WGS84)
        """
        self.crs = crs
        self.hazard_data = None
        self.elevation_data = None
        
    def create_geodataframe(
        self, 
        df: pd.DataFrame, 
        lat_col: str = "latitude", 
        lon_col: str = "longitude"
    ) -> gpd.GeoDataFrame:
        """
        Convertit un DataFrame pandas en GeoDataFrame avec des géométries de points.
        
        Args:
            df: DataFrame contenant les données
            lat_col: Nom de la colonne de latitude
            lon_col: Nom de la colonne de longitude
            
        Returns:
            GeoDataFrame avec des géométries de points
        """
        geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
        return gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)
    
    def load_hazard_data(self, file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Charge des données d'aléas à partir d'un fichier.
        
        Args:
            file_path: Chemin vers le fichier de données d'aléas
            
        Returns:
            GeoDataFrame contenant les données d'aléas
        """
        file_path = Path(file_path)
        if file_path.suffix == '.shp':
            self.hazard_data = gpd.read_file(file_path)
        elif file_path.suffix == '.geojson':
            self.hazard_data = gpd.read_file(file_path)
        else:
            raise ValueError("Format de fichier non supporté. Utilisez .shp ou .geojson")
            
        return self.hazard_data
    
    def add_elevation(
        self, 
        gdf: gpd.GeoDataFrame, 
        dem_path: Union[str, Path],
        elevation_col: str = 'elevation'
    ) -> gpd.GeoDataFrame:
        """
        Ajoute des données d'élévation à partir d'un MNT.
        
        Args:
            gdf: GeoDataFrame d'entrée
            dem_path: Chemin vers le fichier MNT
            elevation_col: Nom de la colonne d'élévation de sortie
            
        Returns:
            GeoDataFrame avec la colonne d'élévation ajoutée
        """
        # Implémentation simplifiée - à adapter selon le format du MNT
        dem = rioxarray.open_rasterio(dem_path)
        
        def get_elevation(point):
            # Vérifier si le point est valide
            if not point.is_valid:
                return np.nan
                
            # Extraire la valeur d'élévation au point
            try:
                return float(dem.sel(
                    x=point.x, 
                    y=point.y, 
                    method='nearest'
                ).values[0])
            except:
                return np.nan
        
        # Appliquer à tous les points
        gdf[elevation_col] = gdf.geometry.apply(get_elevation)
        return gdf
