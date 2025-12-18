"""
Module de traitement géospatial pour l'analyse des risques climatiques en assurance dommages.
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
    
    def load_hazard_data(self, hazard_geojson: Union[str, Dict]) -> None:
        """
        Charge les données d'aléas à partir d'un fichier GeoJSON ou d'un dictionnaire.
        
        Args:
            hazard_geojson: Chemin vers le fichier GeoJSON ou dictionnaire GeoJSON
        """
        if isinstance(hazard_geojson, (str, Path)):
            self.hazard_data = gpd.read_file(hazard_geojson)
        else:
            self.hazard_data = gpd.GeoDataFrame.from_features(
                hazard_geojson["features"],
                crs=self.crs
            )
    
    def spatial_join_hazard(
        self,
        gdf: gpd.GeoDataFrame,
        hazard_name: str = "inondation",
        buffer_meters: float = 0.0,
        hazard_geojson: Optional[Union[str, Dict]] = None
    ) -> gpd.GeoDataFrame:
        """
        Effectue une jointure spatiale avec des données d'aléas.
        
        Args:
            gdf: GeoDataFrame contenant les points d'intérêt
            hazard_name: Nom de l'aléa (utilisé pour nommer les colonnes)
            buffer_meters: Rayon du tampon à appliquer aux points (en mètres)
            hazard_geojson: Optionnel, charge les données d'aléas si non déjà fait
            
        Returns:
            GeoDataFrame avec les colonnes d'aléas ajoutées
        """
        # Charger les données d'aléas si fournies
        if hazard_geojson is not None:
            self.load_hazard_data(hazard_geojson)
        
        if self.hazard_data is None:
            raise ValueError("Aucune donnée d'aléa chargée. Utilisez load_hazard_data() d'abord.")
        
        # Créer une copie pour éviter les modifications sur l'original
        result_gdf = gdf.copy()
        
        # Convertir en CRS projeté pour les calculs de distance (en mètres)
        original_crs = result_gdf.crs
        result_gdf = result_gdf.to_crs("EPSG:3857")  # Web Mercator pour les calculs en mètres
        
        # Appliquer un tampon si nécessaire
        if buffer_meters > 0:
            buffered = result_gdf.geometry.buffer(buffer_meters)
        else:
            buffered = result_gdf.geometry
        
        # Convertir les données d'aléas dans le même CRS projeté
        hazard_gdf = self.hazard_data.to_crs(result_gdf.crs)
        
        # Effectuer la jointure spatiale
        joined = gpd.sjoin(
            gpd.GeoDataFrame(geometry=buffered, crs=result_gdf.crs),
            hazard_gdf,
            how="left",
            predicate="intersects"
        )
        
        # Ajouter les colonnes d'aléas au résultat
        for col in hazard_gdf.columns:
            if col != 'geometry':
                result_gdf[f"{hazard_name}_{col}"] = joined[col]
        
        # Calculer la distance à l'aléa le plus proche
        if not hazard_gdf.empty:
            result_gdf[f"distance_to_{hazard_name}"] = result_gdf.geometry.apply(
                lambda x: hazard_gdf.distance(x).min() if not x.is_empty else np.nan
            )
        
        # Revenir au CRS d'origine
        return result_gdf.to_crs(original_crs)
    
    def add_elevation(
        self,
        gdf: gpd.GeoDataFrame,
        dem_path: str,
        elevation_col: str = "elevation"
    ) -> gpd.GeoDataFrame:
        """
        Ajoute une colonne d'élévation à partir d'un Modèle Numérique de Terrain (MNT).
        
        Args:
            gdf: GeoDataFrame contenant les points d'intérêt
            dem_path: Chemin vers le fichier raster du MNT
            elevation_col: Nom de la colonne d'élévation à créer
            
        Returns:
            GeoDataFrame avec la colonne d'élévation ajoutée
        """
        # Charger le MNT
        dem = rioxarray.open_rasterio(dem_path)
        
        # Convertir les coordonnées dans le même CRS que le MNT
        points = gdf.to_crs(dem.rio.crs)
        
        # Extraire les valeurs d'élévation pour chaque point
        elevations = []
        for x, y in zip(points.geometry.x, points.geometry.y):
            try:
                # Extraire la valeur du pixel correspondant aux coordonnées
                value = float(dem.sel(x=x, y=y, method="nearest").values[0])
                elevations.append(value)
            except Exception as e:
                print(f"Erreur lors de l'extraction de l'élévation: {e}")
                elevations.append(np.nan)
        
        # Ajouter la colonne d'élévation
        gdf[elevation_col] = elevations
        return gdf
    
    def calculate_water_proximity(
        self,
        gdf: gpd.GeoDataFrame,
        water_bodies_geojson: Union[str, Dict],
        max_distance: float = 5000.0
    ) -> gpd.GeoDataFrame:
        """
        Calcule la distance aux plans d'eau les plus proches.
        
        Args:
            gdf: GeoDataFrame contenant les points d'intérêt
            water_bodies_geojson: Données des plans d'eau (fichier ou dictionnaire GeoJSON)
            max_distance: Distance maximale pour le calcul (en mètres)
            
        Returns:
            GeoDataFrame avec les colonnes de distance aux plans d'eau ajoutées
        """
        # Charger les données des plans d'eau
        if isinstance(water_bodies_geojson, (str, Path)):
            water_bodies = gpd.read_file(water_bodies_geojson)
        else:
            water_bodies = gpd.GeoDataFrame.from_features(
                water_bodies_geojson["features"],
                crs=self.crs
            )
        
        # Convertir en CRS projeté pour les calculs de distance
        original_crs = gdf.crs
        gdf_proj = gdf.to_crs("EPSG:3857")
        water_bodies = water_bodies.to_crs("EPSG:3857")
        
        # Calculer la distance au plan d'eau le plus proche
        def min_distance_to_water(point):
            distances = water_bodies.distance(point)
            min_dist = distances.min()
            return min(min_dist, max_distance)  # Limiter à max_distance
        
        gdf["distance_to_water"] = gdf_proj.geometry.apply(min_distance_to_water)
        
        # Revenir au CRS d'origine
        return gdf.to_crs(original_crs)
    
    def add_climate_scenario(
        self,
        gdf: gpd.GeoDataFrame,
        scenario_name: str,
        rcp: str = "4.5",
        year: int = 2050,
        variables: List[str] = ["precipitation", "temperature"]
    ) -> gpd.GeoDataFrame:
        """
        Ajoute des projections climatiques au GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame contenant les points d'intérêt
            scenario_name: Nom du scénario (ex: 'RCP4.5', 'RCP8.5')
            rcp: Scénario RCP (4.5, 6.0, 8.5, etc.)
            year: Année de projection
            variables: Variables climatiques à ajouter
            
        Returns:
            GeoDataFrame avec les colonnes de projection climatique ajoutées
        """
        # Ici, vous intégreriez des données de modèles climatiques
        # Ceci est un exemple simplifié
        
        # Pour chaque variable, ajouter une colonne avec des valeurs simulées
        for var in variables:
            # Simulation d'un modèle simple pour l'exemple
            if var == "precipitation":
                # Augmentation des précipitations de 5% par décennie
                base_value = 100  # Valeur de base en mm
                years_from_now = year - 2023  # À partir de 2023
                change_factor = 1.05 ** (years_from_now / 10)
                gdf[f"{scenario_name}_{var}_{year}"] = base_value * change_factor
                
            elif var == "temperature":
                # Augmentation de la température de 0.2°C par décennie
                base_temp = 15.0  # Température moyenne de base en °C
                years_from_now = year - 2023
                temp_increase = 0.2 * (years_from_now / 10)
                gdf[f"{scenario_name}_{var}_{year}"] = base_temp + temp_increase
        
        return gdf
