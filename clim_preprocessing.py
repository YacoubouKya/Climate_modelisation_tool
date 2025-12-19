"""Prétraitement de base pour Data Tool Climatique.

Ce module fournit des fonctionnalités de prétraitement pour l'analyse des risques climatiques,
y compris la gestion des données géospatiales et temporelles.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Dict, List, Union, Any
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

# Désactiver les avertissements
warnings.filterwarnings('ignore')

# Type pour la fréquence d'agrégation
AggregationFreq = Literal["Aucune", "Jour", "Mois"]


def parse_datetime_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convertit une colonne de dates en datetime et la définit comme index.
    
    Args:
        df: DataFrame contenant les données
        date_col: Nom de la colonne de date à convertir
        
    Returns:
        DataFrame avec la colonne de date convertie et définie comme index
        
    Raises:
        ValueError: Si la colonne n'existe pas ou ne peut pas être convertie en date
    """
    if date_col not in df.columns:
        raise ValueError(f"La colonne de date '{date_col}' est introuvable dans les données")
    
    # Faire une copie pour éviter les effets de bord
    df = df.copy()
    
    # Vérifier si la colonne est déjà au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            # Essayer de convertir en datetime avec plusieurs formats courants
            df[date_col] = pd.to_datetime(
                df[date_col], 
                format='mixed',  # Permet de détecter automatiquement le format
                dayfirst=True,   # Important pour les dates au format européen (JJ/MM/AAAA)
                errors='coerce'  # Convertit les erreurs en NaT
            )
            
            # Vérifier si la conversion a réussi
            if df[date_col].isna().all():
                raise ValueError(f"Impossible de convertir la colonne '{date_col}' en format date/heure")
                
            # Avertir des éventuelles valeurs manquantes après conversion
            na_count = df[date_col].isna().sum()
            if na_count > 0:
                import warnings
                warnings.warn(
                    f"{na_count} valeurs n'ont pas pu être converties en date et ont été remplacées par des valeurs manquantes.",
                    UserWarning
                )
                
        except Exception as e:
            raise ValueError(f"Erreur lors de la conversion de la colonne de date : {str(e)}")
    
    # Trier par date
    df = df.sort_values(by=date_col)
    
    # Définir l'index temporel
    df = df.set_index(date_col)
    
    return df


def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    id_cols: Optional[List[str]] = None,
    agg_func: str = "mean"
) -> pd.DataFrame:
    """Agrège les données temporelles selon la fréquence spécifiée.
    
    Args:
        df: DataFrame contenant les données
        date_col: Nom de la colonne de date
        freq: Fréquence d'agrégation ("Jour", "Mois", etc.)
        id_cols: Colonnes d'identification pour le groupement
        agg_func: Fonction d'agrégation ('mean', 'sum', 'max', 'min')
        
    Returns:
        DataFrame agrégé selon la fréquence spécifiée
    """
    # Faire une copie pour éviter les effets de bord
    df_agg = df.copy()
    
    # S'assurer que la colonne de date est au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df_agg[date_col]):
        df_agg[date_col] = pd.to_datetime(df_agg[date_col], errors='coerce')
    
    # Grouper par période
    if freq.lower() == "jour":
        # Agrégation quotidienne (déjà au bon niveau)
        pass
    elif freq.lower() == "mois":
        # Agrégation mensuelle
        df_agg['_year'] = df_agg[date_col].dt.year
        df_agg['_month'] = df_agg[date_col].dt.month
        group_cols = ['_year', '_month']
    else:
        raise ValueError(f"Fréquence d'agrégation non supportée : {freq}")
    
    # Ajouter les colonnes d'identification au groupement
    if id_cols:
        group_cols = id_cols + group_cols if 'group_cols' in locals() else id_cols.copy()
    
    # Fonction d'agrégation
    if agg_func == "mean":
        agg_func = 'mean'
    elif agg_func == "sum":
        agg_func = 'sum'
    elif agg_func == "max":
        agg_func = 'max'
    elif agg_func == "min":
        agg_func = 'min'
    else:
        raise ValueError("Fonction d'agrégation non reconnue. Utilisez 'mean', 'sum', 'max' ou 'min'.")
    
    # Colonnes numériques à agréger
    numeric_cols = df_agg.select_dtypes(include=['number']).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
    
    # Créer un dictionnaire d'agrégation
    agg_dict = {col: agg_func for col in numeric_cols}
    
    # Grouper et agréger
    if 'group_cols' in locals() and group_cols:
        df_agg = df_agg.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Recréer la colonne de date pour les agrégations mensuelles
        if freq.lower() == "mois":
            df_agg[date_col] = pd.to_datetime(
                df_agg['_year'].astype(str) + '-' + 
                df_agg['_month'].astype(str) + '-01'
            )
            df_agg = df_agg.drop(columns=['_year', '_month'])
    
    # Trier par date
    df_agg = df_agg.sort_values(date_col)
    
    return df_agg

class DataPreprocessor:
    """
    Classe pour le prétraitement des données climatiques et d'assurance.
    """
    
    def __init__(self, date_col: str = "date", id_col: str = "id"):
        """
        Initialise le prétraiteur de données.
        
        Args:
            date_col: Nom de la colonne de date
            id_col: Nom de la colonne d'identifiant
        """
        self.date_col = date_col
        self.id_col = id_col
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Charge les données à partir d'un fichier CSV ou Excel.
        
        Args:
            file_path: Chemin vers le fichier de données
            
        Returns:
            DataFrame contenant les données chargées
        """
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path, parse_dates=[self.date_col])
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, parse_dates=[self.date_col])
        else:
            raise ValueError("Format de fichier non pris en charge. Utilisez CSV ou Excel.")
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = "drop", 
        fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans le DataFrame.
        
        Args:
            df: DataFrame d'entrée
            method: Méthode de traitement ('drop' pour supprimer, 'fill' pour remplacer)
            fill_value: Valeur de remplacement si method='fill'
            
        Returns:
            DataFrame avec les valeurs manquantes traitées
        """
        if method == "drop":
            return df.dropna()
        elif method == "fill":
            return df.fillna(fill_value)
        else:
            raise ValueError("Méthode non reconnue. Utilisez 'drop' ou 'fill'.")
    
    def aggregate_by_frequency(
        self, 
        df: pd.DataFrame, 
        freq: AggregationFreq = "Mois",
        id_cols: Optional[List[str]] = None,
        agg_func: str = "mean"
    ) -> pd.DataFrame:
        """
        Agrège les données par une fréquence temporelle spécifiée.
        
        Args:
            df: DataFrame d'entrée contenant une colonne de date
            freq: Fréquence d'agrégation ('Jour', 'Mois', 'Aucune')
            id_cols: Colonnes d'identification pour le groupement
            agg_func: Fonction d'agrégation ('mean', 'sum', 'max', 'min')
            
        Returns:
            DataFrame agrégé selon la fréquence spécifiée
        """
        if freq == "Aucune" or self.date_col not in df.columns:
            return df
        
        new_df = df.copy()
        new_df[self.date_col] = pd.to_datetime(new_df[self.date_col], errors="coerce")

        if freq == "Jour":
            new_df["_dt_group"] = new_df[self.date_col].dt.date
        elif freq == "Mois":
            new_df["_dt_group"] = new_df[self.date_col].dt.to_period("M").dt.to_timestamp()
        else:
            return df

        group_cols = ["_dt_group"]
        if id_cols:
            group_cols.extend([c for c in id_cols if c in new_df.columns])

        # Exclure les colonnes de groupement de l'agrégation
        num_cols = new_df.select_dtypes(include=["number"]).columns.tolist()
        num_cols = [c for c in num_cols if c not in group_cols and c != "_dt_group"]
        
        # Créer le dictionnaire d'agrégation
        if agg_func == "mean":
            agg_dict = {col: "mean" for col in num_cols}
        elif agg_func == "sum":
            agg_dict = {col: "sum" for col in num_cols}
        elif agg_func == "max":
            agg_dict = {col: "max" for col in num_cols}
        elif agg_func == "min":
            agg_dict = {col: "min" for col in num_cols}
        else:
            raise ValueError("Fonction d'agrégation non reconnue. Utilisez 'mean', 'sum', 'max' ou 'min'.")
        
        # Grouper et agréger
        grouped = new_df.groupby(group_cols).agg(agg_dict).reset_index()
        grouped = grouped.rename(columns={"_dt_group": self.date_col})
        
        return grouped


def add_rolling_features(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Ajoute des moyennes glissantes pour des colonnes numériques.

    - `windows` est une liste de tailles de fenêtre (en nombre de lignes, supposées
      déjà triées dans l'ordre temporel).
    - Les nouvelles colonnes sont nommées `<col>_roll_<window>`.
    """

    if windows is None:
        windows = [3, 7]

    if date_col not in df.columns:
        return df

    new_df = df.sort_values(date_col).copy()

    if value_cols is None:
        value_cols = new_df.select_dtypes(include=["number"]).columns.tolist()

    # Vectorisation : créer toutes les colonnes rolling en une seule passe
    valid_cols = [col for col in value_cols if col in new_df.columns]
    for w in windows:
        # Appliquer rolling sur toutes les colonnes en une fois
        rolled = new_df[valid_cols].rolling(window=w, min_periods=1).mean()
        # Renommer les colonnes
        rolled.columns = [f"{col}_roll_{w}" for col in valid_cols]
        # Ajouter au DataFrame
        new_df = pd.concat([new_df, rolled], axis=1)

    return new_df


def detect_zscore_anomalies(
    df: pd.DataFrame,
    value_cols: Optional[List[str]] = None,
    threshold: float = 3.0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Détecte les anomalies simples par z-score.

    Retourne un DataFrame de flags (True si outlier) et un résumé par colonne.
    """
    if value_cols is None:
        value_cols = df.select_dtypes(include=["number"]).columns.tolist()

    flags = pd.DataFrame(index=df.index)
    summary = {}

    for col in value_cols:
        if col not in df.columns:
            continue
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val == 0:
            continue
        z = np.abs((df[col] - mean_val) / std_val)
        outliers = z > threshold
        flags[f"{col}_outlier"] = outliers

        nb_outliers = outliers.sum()
        pct_outliers = 100.0 * nb_outliers / len(df) if len(df) > 0 else 0.0
        summary[col] = {"nb_outliers": nb_outliers, "pct_outliers": round(pct_outliers, 2)}

    return flags, summary


def add_cumulative_features(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """Ajoute des cumuls glissants sur N jours pour des variables climatiques.
    
    Utile pour : précipitations cumulées, degrés-jours cumulés, etc.
    """
    df_out = df.copy()
    df_out = df_out.sort_values(date_col)
    
    # Vectorisation : traiter toutes les colonnes en une passe
    valid_cols = [col for col in value_cols if col in df_out.columns]
    for window in windows:
        # Appliquer rolling sum sur toutes les colonnes en une fois
        cumul = df_out[valid_cols].rolling(window=window, min_periods=1).sum()
        # Renommer les colonnes
        cumul.columns = [f"{col}_cumul_{window}j" for col in valid_cols]
        # Ajouter au DataFrame
        df_out = pd.concat([df_out, cumul], axis=1)
    
    return df_out


def add_threshold_exceedance_features(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    thresholds: Dict[str, float],
    windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """Compte le nombre de jours dépassant un seuil sur une fenêtre glissante.
    
    Exemple : nombre de jours > 35°C sur les 30 derniers jours.
    thresholds = {"temperature": 35.0, "precipitation": 50.0}
    """
    df_out = df.copy()
    df_out = df_out.sort_values(date_col)
    
    for col in value_cols:
        if col not in df_out.columns or col not in thresholds:
            continue
        threshold = thresholds[col]
        exceed = (df_out[col] > threshold).astype(int)
        
        for window in windows:
            df_out[f"{col}_days_above_{threshold}_{window}j"] = (
                exceed.rolling(window=window, min_periods=1).sum()
            )
    
    return df_out


def add_reference_anomaly_features(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    reference_start: str,
    reference_end: str,
) -> pd.DataFrame:
    """Calcule les anomalies par rapport à une période de référence climatologique.
    
    Exemple : écart à la moyenne 1990-2020 pour chaque mois de l'année.
    """
    df_out = df.copy()
    df_out = df_out.sort_values(date_col)
    
    # Extraire le mois pour calculer les moyennes de référence par mois
    df_out["_month"] = pd.to_datetime(df_out[date_col]).dt.month
    
    # Filtrer la période de référence
    ref_mask = (
        (pd.to_datetime(df_out[date_col]) >= pd.to_datetime(reference_start))
        & (pd.to_datetime(df_out[date_col]) <= pd.to_datetime(reference_end))
    )
    df_ref = df_out[ref_mask]
    
    # Calculer les moyennes mensuelles de référence
    for col in value_cols:
        if col not in df_out.columns:
            continue
        
        monthly_ref = df_ref.groupby("_month")[col].mean().to_dict()
        
        # Calculer l'anomalie pour chaque ligne
        df_out[f"{col}_anomaly_vs_ref"] = df_out.apply(
            lambda row: row[col] - monthly_ref.get(row["_month"], row[col])
            if pd.notna(row[col]) and row["_month"] in monthly_ref
            else np.nan,
            axis=1,
        )
    
    df_out = df_out.drop(columns=["_month"])
    return df_out


def add_extreme_features(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    windows: List[int] = [7, 30],
) -> pd.DataFrame:
    """Ajoute les min/max glissants sur des fenêtres temporelles.
    
    Utile pour identifier les extrêmes récents.
    """
    df_out = df.copy()
    df_out = df_out.sort_values(date_col)
    
    # Vectorisation : traiter toutes les colonnes en une passe
    valid_cols = [col for col in value_cols if col in df_out.columns]
    for window in windows:
        # Appliquer rolling max/min sur toutes les colonnes en une fois
        max_vals = df_out[valid_cols].rolling(window=window, min_periods=1).max()
        min_vals = df_out[valid_cols].rolling(window=window, min_periods=1).min()
        
        # Renommer les colonnes
        max_vals.columns = [f"{col}_max_{window}j" for col in valid_cols]
        min_vals.columns = [f"{col}_min_{window}j" for col in valid_cols]
        
        # Ajouter au DataFrame
        df_out = pd.concat([df_out, max_vals, min_vals], axis=1)
    
    return df_out


def basic_climate_preprocessing(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    freq: str = "Aucune",
    id_cols: Optional[list[str]] = None,
    add_rolling: bool = False,
    rolling_cols: Optional[List[str]] = None,
    detect_anomalies: bool = False,
    anomaly_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Pipeline minimal de prétraitement climatique.

    Args:
        df: DataFrame contenant les données à prétraiter
        date_col: Nom de la colonne de date (optionnel)
        freq: Fréquence d'agrégation ("Aucune", "Jour", "Mois")
        id_cols: Liste des colonnes d'identification
        add_rolling: Si True, ajoute des moyennes mobiles
        rolling_cols: Colonnes pour le calcul des moyennes mobiles
        detect_anomalies: Si True, détecte les anomalies
        anomaly_cols: Colonnes à analyser pour la détection d'anomalies

    Returns:
        Tuple contenant:
        - df_prep: DataFrame prétraité
        - info: Dictionnaire récapitulatif pour l'affichage
    """
    import streamlit as st
    from typing import Dict, Any
    
    info: Dict[str, Any] = {}
    df_prep = df.copy()

    # 1) Gestion de la date
    if date_col and date_col != "(aucune)":
        try:
            df_prep = parse_datetime_column(df_prep, date_col)
            info["date_col"] = date_col
            info["date_range"] = {
                "debut": df_prep.index.min().strftime("%Y-%m-%d"),
                "fin": df_prep.index.max().strftime("%Y-%m-%d")
            }
        except Exception as e:
            st.error(f"Erreur lors du traitement des dates : {str(e)}")
            st.warning("Le prétraitement continue sans utiliser la colonne de date.")
            date_col = None

    # 2) Agrégation temporelle
    if freq != "Aucune" and date_col:
        try:
            df_prep = aggregate_time_series(df_prep, date_col=date_col, freq=freq, id_cols=id_cols)
            info["freq"] = freq
        except Exception as e:
            st.error(f"Erreur lors de l'agrégation temporelle : {str(e)}")
            st.warning("Le prétraitement continue sans agrégation temporelle.")

    # 3) Moyennes mobiles
    if add_rolling and date_col and rolling_cols:
        try:
            df_prep = add_rolling_features(df_prep, date_col=date_col, value_cols=rolling_cols)
            info["rolling"] = True
            info["rolling_cols"] = rolling_cols
        except Exception as e:
            st.error(f"Erreur lors du calcul des moyennes mobiles : {str(e)}")
            st.warning("Le prétraitement continue sans moyennes mobiles.")

    # 4) Détection d'anomalies
    if detect_anomalies:
        try:
            flags, summary = detect_zscore_anomalies(
                df_prep, 
                value_cols=anomaly_cols or [col for col in df_prep.select_dtypes(include=['number']).columns 
                                          if col != date_col]
            )
            info["anomaly_summary"] = summary
            info["anomaly_columns"] = anomaly_cols
        except Exception as e:
            st.error(f"Erreur lors de la détection d'anomalies : {str(e)}")
            st.warning("Le prétraitement continue sans détection d'anomalies.")

    info["shape"] = df_prep.shape
    info["columns"] = df_prep.columns.tolist()
    
    return df_prep, info
