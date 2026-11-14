"""Hyperparamètres ajustables pour l'affinage du meilleur modèle.

Chaque modèle déclare ses hyperparamètres réglables ; les valeurs choisies sont
appliquées à une copie non entraînée du meilleur modèle (sklearn.base.clone),
ce qui conserve ses autres réglages (graine, poids de classes, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st
from sklearn.base import clone


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str  # "int", "float" ou "depth" (entier, 0 = profondeur illimitée)
    min_value: float
    max_value: float
    step: float
    default: float


_TREES = [
    Param("n_estimators", "Nombre d'arbres", "int", 10, 1000, 10, 100),
    Param("max_depth", "Profondeur maximale (0 = illimitée)", "depth", 0, 50, 1, 0),
    Param("min_samples_split", "Observations minimales pour diviser un noeud", "int", 2, 50, 1, 2),
]

TUNING_SPECS: dict[str, list[Param]] = {
    "Random Forest": _TREES,
    "Extra Trees": _TREES,
    "Gradient Boosting": [
        Param("n_estimators", "Nombre d'arbres", "int", 10, 1000, 10, 100),
        Param("learning_rate", "Taux d'apprentissage", "float", 0.01, 1.0, 0.01, 0.1),
        Param("max_depth", "Profondeur maximale", "int", 1, 20, 1, 3),
    ],
    "Decision Tree": [
        Param("max_depth", "Profondeur maximale (0 = illimitée)", "depth", 0, 50, 1, 0),
        Param("min_samples_split", "Observations minimales pour diviser un noeud", "int", 2, 50, 1, 2),
    ],
    "AdaBoost": [
        Param("n_estimators", "Nombre d'estimateurs", "int", 10, 1000, 10, 50),
        Param("learning_rate", "Taux d'apprentissage", "float", 0.01, 2.0, 0.01, 1.0),
    ],
    "K-Nearest Neighbors": [
        Param("n_neighbors", "Nombre de voisins", "int", 1, 100, 1, 5),
    ],
    "Logistic Regression": [
        Param("C", "Inverse de la régularisation (C)", "float", 0.001, 100.0, 0.01, 1.0),
    ],
    "Ridge": [
        Param("alpha", "Régularisation (alpha)", "float", 0.001, 100.0, 0.01, 1.0),
    ],
    "Lasso": [
        Param("alpha", "Régularisation (alpha)", "float", 0.001, 100.0, 0.01, 1.0),
    ],
}


def is_tunable(model_name: str) -> bool:
    return model_name in TUNING_SPECS


def tuning_widgets(model_name: str, current_model: Any) -> dict[str, Any]:
    """Affiche un champ par hyperparamètre, pré-rempli avec la valeur du meilleur modèle."""
    params = TUNING_SPECS[model_name]
    current = current_model.get_params() if current_model is not None else {}
    values: dict[str, Any] = {}
    cols = st.columns(len(params))
    for col, p in zip(cols, params):
        start = current.get(p.name, p.default)
        if p.kind == "depth":
            start = 0 if start is None else start
        with col:
            if p.kind == "float":
                value = st.number_input(
                    p.label, min_value=float(p.min_value), max_value=float(p.max_value),
                    value=float(min(max(start, p.min_value), p.max_value)), step=float(p.step),
                    format="%.3f", key=f"tune_{model_name}_{p.name}",
                )
            else:
                value = st.number_input(
                    p.label, min_value=int(p.min_value), max_value=int(p.max_value),
                    value=int(min(max(start, p.min_value), p.max_value)), step=int(p.step),
                    key=f"tune_{model_name}_{p.name}",
                )
        values[p.name] = None if (p.kind == "depth" and value == 0) else value
    return values


def build_tuned_estimator(current_model: Any, values: dict[str, Any]) -> Any:
    """Copie non entraînée du meilleur modèle avec les hyperparamètres choisis."""
    return clone(current_model).set_params(**values)
