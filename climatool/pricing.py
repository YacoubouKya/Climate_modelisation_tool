"""Tarification du risque climatique : de la probabilité d'événement à la prime.

Approche fréquence x coût moyen :
- fréquence : probabilité d'événement prédite par le modèle pour chaque observation
  (par exemple une zone un jour donné) ;
- coût moyen : coût moyen observé d'un sinistre, global ou par segment ;
- charge attendue annuelle = somme des (probabilité x coût moyen) / nombre d'années ;
- prime pure par unité exposée = charge attendue annuelle / exposition (nombre de biens) ;
- prime commerciale = prime pure x (1 + chargement de sécurité) / (1 - frais - marge).

Les fonctions de ce module sont indépendantes de l'interface Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Loadings:
    """Chargements appliqués à la prime pure (en proportion, 0.10 = 10 %)."""

    security: float = 0.10
    fees: float = 0.15
    margin: float = 0.05

    def commercial_factor(self) -> float:
        if self.fees + self.margin >= 1:
            raise ValueError("Frais et marge doivent totaliser moins de 100 %.")
        return (1 + self.security) / (1 - self.fees - self.margin)


def observed_severity(
    df: pd.DataFrame,
    event_col: str,
    cost_col: str,
    segment_col: str | None = None,
    credibility_k: float = 0.0,
) -> pd.Series | float:
    """Coût moyen d'un sinistre, calculé sur les observations où l'événement a eu lieu.

    Par segment si `segment_col` est fourni. Avec credibility_k > 0, le coût moyen
    de chaque segment est pondéré avec le coût moyen global :
        Z = n / (n + k),  coût = Z x coût du segment + (1 - Z) x coût global,
    où n est le nombre de sinistres du segment (crédibilité de type Bühlmann).
    Les segments sans sinistre reçoivent le coût moyen global.
    """
    events = df[df[event_col] == 1]
    if events.empty:
        raise ValueError("Aucun événement observé : impossible d'estimer le coût moyen.")
    overall = float(events[cost_col].mean())
    if not segment_col:
        return overall
    stats = events.groupby(segment_col)[cost_col].agg(["mean", "size"])
    z = stats["size"] / (stats["size"] + credibility_k) if credibility_k > 0 else 1.0
    credible = z * stats["mean"] + (1 - z) * overall
    segments = pd.Index(df[segment_col].unique())
    return credible.reindex(segments).fillna(overall)


def credibility_weights(df: pd.DataFrame, event_col: str, segment_col: str, credibility_k: float) -> pd.Series:
    """Poids de crédibilité Z = n / (n + k) par segment (k > 0)."""
    n = df[df[event_col] == 1].groupby(segment_col).size().reindex(df[segment_col].unique()).fillna(0)
    return n / (n + credibility_k)


def number_of_years(df: pd.DataFrame, date_col: str | None) -> float:
    """Durée couverte par les données, en années (1 si aucune date)."""
    if not date_col or date_col not in df.columns:
        return 1.0
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return 1.0
    return max((dates.max() - dates.min()).days + 1, 1) / 365.25


def price_portfolio(
    df: pd.DataFrame,
    proba: np.ndarray,
    event_col: str,
    cost_col: str | None = None,
    average_cost: float | None = None,
    segment_col: str | None = None,
    exposure_col: str | None = None,
    date_col: str | None = None,
    severity_by_segment: bool = False,
    loadings: Loadings = Loadings(),
    recalibrate: bool = True,
    credibility_k: float = 0.0,
) -> pd.DataFrame:
    """Tableau de tarification par segment (ou global si aucun segment).

    Colonnes : observations, événements observés, fréquence observée et modélisée,
    coût moyen, charge annuelle observée et attendue, ratio observé / attendu,
    exposition, prime pure et prime commerciale par unité exposée.

    recalibrate : ramène la fréquence modélisée moyenne à la fréquence observée
    sur le portefeuille (calage global). Indispensable si le modèle a été entraîné
    avec une pondération des classes, qui surestime les probabilités.
    """
    if cost_col is None and average_cost is None:
        raise ValueError("Indiquez une colonne de coût ou un coût moyen.")

    data = df.copy()
    data["_proba"] = np.asarray(proba, dtype=float)
    if recalibrate and data["_proba"].mean() > 0:
        data["_proba"] = (data["_proba"] * data[event_col].mean() / data["_proba"].mean()).clip(0, 1)
    data["_segment"] = data[segment_col].astype(str) if segment_col else "Portefeuille"

    if cost_col is not None:
        severity = observed_severity(
            data, event_col, cost_col, "_segment" if severity_by_segment else None, credibility_k
        )
    else:
        severity = float(average_cost)
    if isinstance(severity, pd.Series):
        data["_severity"] = data["_segment"].map(severity)
    else:
        data["_severity"] = severity

    data["_expected"] = data["_proba"] * data["_severity"]
    years = number_of_years(data, date_col)

    grouped = data.groupby("_segment")
    table = pd.DataFrame({
        "Observations": grouped.size(),
        "Événements observés": grouped[event_col].sum(),
        "Fréquence observée": grouped[event_col].mean(),
        "Fréquence modélisée": grouped["_proba"].mean(),
        "Coût moyen d'un sinistre": grouped["_severity"].mean(),
        "Charge annuelle attendue": grouped["_expected"].sum() / years,
    })
    if cost_col is not None and severity_by_segment and credibility_k > 0:
        table.insert(5, "Crédibilité Z", credibility_weights(data, event_col, "_segment", credibility_k))
    if cost_col is not None:
        table["Charge annuelle observée"] = grouped[cost_col].sum() / years
        table["Ratio observé / attendu"] = table["Charge annuelle observée"] / table["Charge annuelle attendue"]

    if exposure_col:
        table["Exposition"] = grouped[exposure_col].mean()
        table["Prime pure par unité"] = table["Charge annuelle attendue"] / table["Exposition"]
    else:
        table["Prime pure par unité"] = table["Charge annuelle attendue"]
    table["Prime commerciale par unité"] = table["Prime pure par unité"] * loadings.commercial_factor()

    table.index.name = segment_col or "Segment"
    return table.sort_values("Charge annuelle attendue", ascending=False)


def portfolio_totals(table: pd.DataFrame) -> dict:
    """Totaux du portefeuille pour l'affichage des indicateurs clés."""
    totals = {
        "charge_attendue": float(table["Charge annuelle attendue"].sum()),
        "charge_observee": float(table["Charge annuelle observée"].sum()) if "Charge annuelle observée" in table else None,
    }
    if "Exposition" in table:
        exposure = float(table["Exposition"].sum())
        totals["exposition"] = exposure
        totals["prime_pure_moyenne"] = totals["charge_attendue"] / exposure if exposure else None
        factor = (table["Prime commerciale par unité"] / table["Prime pure par unité"]).iloc[0]
        totals["prime_commerciale_moyenne"] = totals["prime_pure_moyenne"] * factor if exposure else None
    return totals
