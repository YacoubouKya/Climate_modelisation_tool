"""Utilitaires pour la manipulation de données - Data Tool Climatique.

Ce module centralise les fonctions de fusion, nettoyage et transformation
de données pour éviter la duplication.
"""

from __future__ import annotations

import pandas as pd


def merge_dataframes(dfs: list[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    """Fusionne intelligemment plusieurs DataFrames.
    
    Stratégie :
    - Si colonnes communes : merge sur ces colonnes
    - Sinon : concaténation horizontale avec renommage des doublons
    
    Parameters
    ----------
    dfs : list[pd.DataFrame]
        Liste de DataFrames à fusionner
    how : str, default="outer"
        Type de merge ('inner', 'outer', 'left', 'right')
        
    Returns
    -------
    pd.DataFrame
        DataFrame fusionné
        
    Raises
    ------
    ValueError
        Si la liste est vide ou contient des DataFrames invalides
    TypeError
        Si dfs ne contient pas des DataFrames
    """
    if not dfs:
        raise ValueError("La liste de DataFrames est vide")
    
    # Valider que tous les éléments sont des DataFrames
    for i, df in enumerate(dfs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"L'élément {i} n'est pas un DataFrame (type: {type(df)})")
        if df.empty:
            import warnings
            warnings.warn(f"Le DataFrame {i} est vide et sera ignoré")
    
    # Filtrer les DataFrames vides
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        raise ValueError("Tous les DataFrames sont vides")
    
    if len(dfs) == 1:
        return dfs[0].copy()
    
    df = dfs[0].copy()
    
    for i, other_df in enumerate(dfs[1:], 1):
        common_cols = list(set(df.columns) & set(other_df.columns))
        
        if common_cols:
            # Merge sur colonnes communes
            df = pd.merge(df, other_df, on=common_cols, how=how, suffixes=("", f"_dup{i}"))
            # Supprimer les colonnes dupliquées
            dup_cols = [c for c in df.columns if f"_dup{i}" in c]
            if dup_cols:
                df = df.drop(columns=dup_cols)
        else:
            # Concaténation avec renommage des conflits
            conflicting_cols = [col for col in other_df.columns if col in df.columns]
            if conflicting_cols:
                rename_dict = {col: f"{col}_src{i+1}" for col in conflicting_cols}
                other_df_renamed = other_df.rename(columns=rename_dict)
            else:
                other_df_renamed = other_df
            df = pd.concat([df, other_df_renamed], axis=1)
    
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les noms de colonnes d'un DataFrame.
    
    - Supprime les espaces
    - Remplace les caractères spéciaux
    - Convertit en minuscules
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à nettoyer
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec noms de colonnes nettoyés
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('[^a-z0-9_]', '', regex=True)
    )
    return df


def detect_date_columns(df: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    """Détecte automatiquement les colonnes de dates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser
    threshold : float, default=0.8
        Seuil de conversion réussie pour considérer une colonne comme date
        
    Returns
    -------
    list[str]
        Liste des noms de colonnes détectées comme dates
    """
    date_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_cols.append(col)
            continue
        
        # Essayer de convertir en date
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            success_rate = converted.notna().sum() / len(df)
            if success_rate >= threshold:
                date_cols.append(col)
        except Exception:
            continue
    
    return date_cols


def get_numeric_columns(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """Retourne les colonnes numériques d'un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser
    exclude : list[str], optional
        Colonnes à exclure
        
    Returns
    -------
    list[str]
        Liste des colonnes numériques
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if exclude:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """Retourne les colonnes catégorielles d'un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser
    exclude : list[str], optional
        Colonnes à exclure
        
    Returns
    -------
    list[str]
        Liste des colonnes catégorielles
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if exclude:
        cat_cols = [col for col in cat_cols if col not in exclude]
    
    return cat_cols


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes dupliquées d'un DataFrame.
    
    Garde la première occurrence de chaque colonne unique.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à nettoyer
        
    Returns
    -------
    pd.DataFrame
        DataFrame sans colonnes dupliquées
    """
    return df.loc[:, ~df.columns.duplicated()]


def get_memory_usage(df: pd.DataFrame) -> dict[str, float]:
    """Calcule l'utilisation mémoire d'un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser
        
    Returns
    -------
    dict[str, float]
        Dictionnaire avec 'total_mb', 'per_column_mb'
    """
    total_bytes = df.memory_usage(deep=True).sum()
    total_mb = total_bytes / 1024 / 1024
    
    per_column = df.memory_usage(deep=True) / 1024 / 1024
    
    return {
        'total_mb': total_mb,
        'per_column_mb': per_column.to_dict()
    }
