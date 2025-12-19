"""
Module d'analyse spatiale avancée pour les risques climatiques.

Fournit des fonctions pour des analyses spatiales complexes.
"""

from typing import Optional, Union, List, Dict, Any
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point, Polygon
import numpy as np
from sklearn.cluster import DBSCAN
import warnings

# Désactiver les avertissements
warnings.filterwarnings('ignore')

def spatial_join_hazard(
    gdf: gpd.GeoDataFrame,
    hazard_data: gpd.GeoDataFrame,
    how: str = 'inner',
    op: str = 'intersects'
) -> gpd.GeoDataFrame:
    """
    Effectue une jointure spatiale entre des points et des données d'aléas.
    
    Args:
        gdf: GeoDataFrame des points d'intérêt
        hazard_data: GeoDataFrame des données d'aléas
        how: Type de jointure ('inner', 'left', 'right')
        op: Opération spatiale ('intersects', 'within', 'contains')
        
    Returns:
        GeoDataFrame résultant de la jointure
    """
    # Vérifier les CRS
    if gdf.crs != hazard_data.crs:
        hazard_data = hazard_data.to_crs(gdf.crs)
    
    # Effectuer la jointure spatiale
    return gpd.sjoin(gdf, hazard_data, how=how, op=op)

def calculate_water_proximity(
    gdf: gpd.GeoDataFrame,
    water_bodies: gpd.GeoDataFrame,
    max_distance: float = 1000,
    distance_col: str = 'distance_to_water'
) -> gpd.GeoDataFrame:
    """
    Calcule la distance aux plans d'eau les plus proches.
    
    Args:
        gdf: GeoDataFrame des points d'intérêt
        water_bodies: GeoDataFrame des plans d'eau
        max_distance: Distance maximale de recherche (mètres)
        distance_col: Nom de la colonne de sortie pour la distance
        
    Returns:
        GeoDataFrame avec la colonne de distance ajoutée
    """
    # Vérifier les CRS et convertir en projection métrique si nécessaire
    if gdf.crs != water_bodies.crs:
        water_bodies = water_bodies.to_crs(gdf.crs)
    
    # Créer une copie pour éviter les modifications sur l'original
    result = gdf.copy()
    
    # Calculer la distance au plan d'eau le plus proche
    def get_min_distance(point, water_geoms):
        distances = water_geoms.distance(point)
        return distances.min()
    
    # Convertir en projection métrique si nécessaire (EPSG:3857 pour les mètres)
    if not gdf.crs.is_projected:
        metric_crs = 'EPSG:3857'
        gdf_metric = gdf.to_crs(metric_crs)
        water_bodies_metric = water_bodies.to_crs(metric_crs)
    else:
        gdf_metric = gdf
        water_bodies_metric = water_bodies
    
    # Calculer les distances
    water_geoms = water_bodies_metric.geometry.unary_union
    result[distance_col] = gdf_metric.geometry.apply(
        lambda x: min(x.distance(water_geoms), max_distance)
    )
    
    return result

def add_climate_scenario(
    gdf: gpd.GeoDataFrame,
    scenario_data: Union[gpd.GeoDataFrame, str],
    scenario_name: str = 'rcp45',
    year: int = 2050,
    id_col: str = 'id'
) -> gpd.GeoDataFrame:
    """
    Ajoute des données de scénario climatique aux données spatiales.
    
    Args:
        gdf: GeoDataFrame des données spatiales
        scenario_data: Données du scénario (GeoDataFrame ou chemin vers fichier)
        scenario_name: Nom du scénario (ex: 'rcp45', 'rcp85')
        year: Année du scénario
        id_col: Nom de la colonne d'identification
        
    Returns:
        GeoDataFrame avec les données du scénario ajoutées
    """
    # Charger les données si nécessaire
    if isinstance(scenario_data, str):
        if scenario_data.endswith(('.shp', '.geojson')):
            scenario_data = gpd.read_file(scenario_data)
        else:
            raise ValueError("Format de fichier non supporté. Utilisez .shp ou .geojson")
    
    # Vérifier les CRS
    if gdf.crs != scenario_data.crs:
        scenario_data = scenario_data.to_crs(gdf.crs)
    
    # Effectuer la jointure spatiale
    joined = gpd.sjoin(gdf, scenario_data, how='left', op='intersects')
    
    # Filtrer les colonnes du scénario
    scenario_cols = [c for c in scenario_data.columns 
                    if scenario_name in c.lower() and str(year) in c]
    
    # Conserver uniquement les colonnes nécessaires
    keep_cols = [id_col, 'geometry'] + scenario_cols
    result = joined[keep_cols].drop_duplicates(subset=id_col)
    
    return result

def detect_clusters(
    gdf: gpd.GeoDataFrame,
    eps: float = 0.1,
    min_samples: int = 5,
    cluster_col: str = 'cluster'
) -> gpd.GeoDataFrame:
    """
    Détecte les clusters spatiaux avec DBSCAN.
    
    Args:
        gdf: GeoDataFrame des points
        eps: Distance maximale entre deux échantillons pour les regrouper
        min_samples: Nombre minimum d'échantillons dans un voisinage
        cluster_col: Nom de la colonne de sortie pour les clusters
        
    Returns:
        GeoDataFrame avec la colonne de cluster ajoutée
    """
    # Extraire les coordonnées
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    
    # Appliquer DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    clusters = dbscan.fit_predict(coords)
    
    # Ajouter les clusters au GeoDataFrame
    result = gdf.copy()
    result[cluster_col] = clusters
    
    return result
