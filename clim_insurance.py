"""
Module pour l'analyse des risques d'assurance et le calcul des primes.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceAnalyzer:
    """
    Classe pour l'analyse des risques d'assurance et le calcul des primes.
    """
    
    def __init__(self):
        """Initialise l'analyseur d'assurance avec des paramètres par défaut."""
        self.risk_categories = {
            'low': (0, 0.25),
            'medium': (0.25, 0.75),
            'high': (0.75, 0.95),
            'extreme': (0.95, 1.0)
        }
        
    def categorize_risk(
        self, 
        risk_scores: pd.Series
    ) -> pd.Series:
        """
        Catégorise les scores de risque en niveaux (faible, moyen, élevé, extrême).
        
        Args:
            risk_scores: Série pandas contenant les scores de risque (0-1)
            
        Returns:
            Série pandas avec les catégories de risque
        """
        return pd.cut(
            risk_scores,
            bins=[0, 0.25, 0.75, 0.95, 1.0],
            labels=['low', 'medium', 'high', 'extreme'],
            include_lowest=True
        )
    
    def calculate_premiums(
        self,
        base_premium: float,
        risk_categories: pd.Series
    ) -> pd.Series:
        """
        Calcule les primes d'assurance en fonction des catégories de risque.
        
        Args:
            base_premium: Prime de base pour le risque moyen
            risk_categories: Série pandas contenant les catégories de risque
            
        Returns:
            Série pandas avec les primes calculées
        """
        factors = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.5,
            'extreme': 2.5
        }
        return base_premium * risk_categories.map(factors)
    
    def estimate_claim_frequency(
        self,
        gdf: gpd.GeoDataFrame,
        hazard_col: str,
        exposure_col: str = 'exposure'
    ) -> pd.Series:
        """
        Estime la fréquence des sinistres en fonction de l'exposition et de l'aléa.
        
        Args:
            gdf: GeoDataFrame contenant les données d'exposition et d'aléa
            hazard_col: Nom de la colonne contenant les scores d'aléa
            exposure_col: Nom de la colonne contenant les valeurs d'exposition
            
        Returns:
            Série pandas avec les fréquences de sinistres estimées
        """
        # Modèle simple: fréquence = aléa * exposition normalisée
        exposure_norm = (gdf[exposure_col] / gdf[exposure_col].max())
        return gdf[hazard_col] * exposure_norm
    
    def calculate_technical_premium(
        self,
        frequency: pd.Series,
        severity: pd.Series,
        safety_load: float = 0.3,
        expense_ratio: float = 0.25
    ) -> pd.Series:
        """
        Calcule la prime technique en fonction de la fréquence et de la sévérité des sinistres.
        
        Args:
            frequency: Série pandas contenant les fréquences de sinistres
            severity: Série pandas contenant les coûts moyens des sinistres
            safety_load: Charge de sécurité (défaut: 0.3 pour 30%)
            expense_ratio: Ratio des frais généraux (défaut: 0.25 pour 25%)
            
        Returns:
            Série pandas avec les primes techniques calculées
        """
        pure_premium = frequency * severity
        risk_premium = pure_premium * (1 + safety_load)
        technical_premium = risk_premium / (1 - expense_ratio)
        return technical_premium
    
    def calculate_risk_aggregates(
        self,
        gdf: gpd.GeoDataFrame,
        loss_column: str,
        return_periods: List[int] = [10, 50, 100, 200, 500]
    ) -> Dict[str, float]:
        """
        Calcule les agrégats de risque (AAL, PML) pour différentes périodes de retour.
        
        Args:
            gdf: GeoDataFrame contenant les données de pertes
            loss_column: Nom de la colonne contenant les montants de pertes
            return_periods: Liste des périodes de retour à calculer
            
        Returns:
            Dictionnaire contenant les métriques de risque
        """
        # Trier les pertes par ordre décroissant
        losses = gdf[loss_column].sort_values(ascending=False).reset_index(drop=True)
        n = len(losses)
        
        # Calculer l'AAL (Average Annual Loss)
        aal = losses.mean()
        
        # Calculer la PML pour différentes périodes de retour
        pml_results = {}
        for rp in return_periods:
            rank = n / rp
            idx = min(int(round(rank)), n - 1)
            pml_results[f'PML_{rp}'] = losses.iloc[idx]
        
        # Retourner les résultats
        return {
            'AAL': aal,
            **pml_results
        }
    
    def plot_loss_exceedance_curve(
        self,
        gdf: gpd.GeoDataFrame,
        loss_column: str,
        title: str = "Courbe de dépassement des pertes",
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Trace la courbe de dépassement des pertes (Loss Exceedance Curve).
        
        Args:
            gdf: GeoDataFrame contenant les données de pertes
            loss_column: Nom de la colonne contenant les montants de pertes
            title: Titre du graphique
            figsize: Taille de la figure
            
        Returns:
            Figure matplotlib
        """
        # Trier les pertes par ordre décroissant
        losses = gdf[loss_column].sort_values(ascending=False).reset_index(drop=True)
        n = len(losses)
        exceedance_probs = (np.arange(n) + 1) / n
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer la courbe de dépassement
        ax.plot(exceedance_probs, losses, 'b-', linewidth=2)
        ax.set_xlabel('Probabilité de dépassement')
        ax.set_ylabel('Montant des pertes')
        ax.set_title(title)
        ax.grid(True)
        
        # Ajouter des lignes pour les périodes de retour courantes
        for rp in [10, 50, 100, 200, 500]:
            if rp <= n:
                idx = min(int(round(n / rp)), n - 1)
                ax.axvline(x=1/rp, color='r', linestyle='--', alpha=0.5)
                ax.text(1/rp * 1.1, losses.iloc[idx] * 0.9, f'1/{rp}', color='r')
        
        return fig
