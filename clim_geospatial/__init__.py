"""
Package pour l'analyse géospatiale avancée des risques climatiques.

Ce package fournit des fonctionnalités pour :
- Le traitement des données géospatiales (core.py)
- La visualisation cartographique (visualization.py)
- L'analyse spatiale avancée (analysis.py)
"""

from .core import GeoProcessor
from .visualization import (
    create_map,
    show_risk_map,
    run_maps_page,
    detect_lat_lon_columns
)
from .analysis import (
    spatial_join_hazard,
    calculate_water_proximity,
    add_climate_scenario
)

__all__ = [
    'GeoProcessor',
    'create_map',
    'show_risk_map',
    'run_maps_page',
    'detect_lat_lon_columns',
    'spatial_join_hazard',
    'calculate_water_proximity',
    'add_climate_scenario'
]
