"""Prétraitement de base pour Data Tool Climatique.

Cette première version fournit des opérations simples mais utiles pour un
hackathon :
- sélection des colonnes date / identifiant / variables climatiques,
- mise en forme de la date et tri chronologique,
- agrégation temporelle basique (par jour ou par mois).

L’objectif est de fournir un pipeline minimal, facile à étendre ensuite.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


AggregationFreq = Literal["Aucune", "Jour", "Mois"]


def parse_datetime_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convertit une colonne en datetime et trie le DataFrame.

    Si la conversion échoue, la fonction renvoie le DataFrame d’origine.
    """

    new_df = df.copy()
    try:
        new_df[date_col] = pd.to_datetime(new_df[date_col])
        new_df = new_df.sort_values(date_col)
    except Exception:
        # On ne crash pas : le reste de l’app pourra quand même fonctionner.
        pass
    return new_df


def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    freq: AggregationFreq = "Aucune",
    id_cols: Optional[list[str]] = None,
    sum_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Agrège des séries temporelles à une fréquence donnée (par identifiant).

    Règles d'agrégation :
    - colonnes de `sum_cols` (pluie, coûts...) : somme sur la période ;
    - colonnes binaires 0/1 (événement observé) : maximum, soit « au moins un
      événement sur la période » ;
    - autres colonnes numériques : moyenne ;
    - colonnes texte constantes dans le groupe (type de zone...) : conservées.
    """

    if freq == "Aucune" or date_col not in df.columns:
        return df

    new_df = df.copy()
    new_df[date_col] = pd.to_datetime(new_df[date_col], errors="coerce")

    if freq == "Jour":
        new_df["_dt_group"] = new_df[date_col].dt.normalize()
    elif freq == "Mois":
        new_df["_dt_group"] = new_df[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        return df

    group_cols = ["_dt_group"] + [c for c in (id_cols or []) if c in new_df.columns]
    sum_cols = [c for c in (sum_cols or []) if c in new_df.columns and c not in group_cols]

    agg_dict = {}
    for col in new_df.columns:
        if col in group_cols or col == date_col:
            continue
        series = new_df[col]
        if pd.api.types.is_numeric_dtype(series):
            if col in sum_cols:
                agg_dict[col] = "sum"
            elif set(series.dropna().unique()) <= {0, 1}:
                agg_dict[col] = "max"
            else:
                agg_dict[col] = "mean"
        elif (new_df.groupby(group_cols[1:] or ["_dt_group"])[col].nunique(dropna=True) <= 1).all():
            agg_dict[col] = "first"

    grouped = new_df.groupby(group_cols).agg(agg_dict).reset_index()
    return grouped.rename(columns={"_dt_group": date_col})


def _group_rolling(df: pd.DataFrame, cols: List[str], id_cols: Optional[List[str]], window: int, how: str) -> pd.DataFrame:
    """Statistique glissante calculée séparément pour chaque identifiant (zone, station)."""
    ids = [c for c in (id_cols or []) if c in df.columns]
    if ids:
        rolled = df.groupby(ids, sort=False)[cols].rolling(window=window, min_periods=1).agg(how)
        rolled = rolled.reset_index(level=list(range(len(ids))), drop=True)
        return rolled.reindex(df.index)
    return df[cols].rolling(window=window, min_periods=1).agg(how)


def add_rolling_features(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ajoute des moyennes glissantes `<col>_roll_<n>` (n observations), par identifiant."""
    windows = windows or [3, 7]
    if date_col not in df.columns:
        return df
    new_df = df.sort_values([c for c in (id_cols or []) if c in df.columns] + [date_col]).copy()
    if value_cols is None:
        value_cols = new_df.select_dtypes(include=["number"]).columns.tolist()
    valid_cols = [col for col in value_cols if col in new_df.columns]
    for w in windows:
        rolled = _group_rolling(new_df, valid_cols, id_cols, w, "mean")
        rolled.columns = [f"{col}_roll_{w}" for col in valid_cols]
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
    windows: Optional[List[int]] = None,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ajoute des cumuls glissants `<col>_cumul_<n>` (pluie cumulée...), par identifiant."""
    windows = windows or [3, 7]
    if date_col not in df.columns:
        return df
    df_out = df.sort_values([c for c in (id_cols or []) if c in df.columns] + [date_col]).copy()
    valid_cols = [col for col in value_cols if col in df_out.columns]
    for window in windows:
        cumul = _group_rolling(df_out, valid_cols, id_cols, window, "sum")
        cumul.columns = [f"{col}_cumul_{window}" for col in valid_cols]
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
    freq: AggregationFreq = "Aucune",
    id_cols: Optional[list[str]] = None,
    add_rolling: bool = False,
    rolling_cols: Optional[List[str]] = None,
    detect_anomalies: bool = False,
    anomaly_cols: Optional[List[str]] = None,
    sum_cols: Optional[List[str]] = None,
    add_cumul: bool = False,
    cumul_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Pipeline de prétraitement climatique.

    Ordre : conversion de la date, agrégation, cumuls et moyennes glissantes
    (calculés séparément pour chaque identifiant), détection d'anomalies.
    Les fenêtres sont exprimées en nombre de périodes (jours ou mois selon
    l'agrégation).

    Renvoie le DataFrame prétraité et un résumé pour l'affichage.
    """
    info: dict = {}
    df_prep = df.copy()

    if date_col:
        df_prep = parse_datetime_column(df_prep, date_col)
        info["date_col"] = date_col

    if freq != "Aucune" and date_col:
        df_prep = aggregate_time_series(df_prep, date_col=date_col, freq=freq, id_cols=id_cols, sum_cols=sum_cols)
        info["freq"] = freq

    if add_cumul and date_col and cumul_cols:
        df_prep = add_cumulative_features(df_prep, date_col=date_col, value_cols=cumul_cols, windows=windows, id_cols=id_cols)
        info["cumul"] = cumul_cols

    if add_rolling and date_col and rolling_cols:
        df_prep = add_rolling_features(df_prep, date_col=date_col, value_cols=rolling_cols, windows=windows, id_cols=id_cols)
        info["rolling"] = rolling_cols

    if detect_anomalies:
        _, summary = detect_zscore_anomalies(df_prep, value_cols=anomaly_cols)
        info["anomaly_summary"] = summary

    if date_col and date_col in df_prep.columns:
        df_prep = df_prep.sort_values([c for c in (id_cols or []) if c in df_prep.columns] + [date_col]).reset_index(drop=True)

    info["shape"] = df_prep.shape
    return df_prep, info
