"""Modélisation de base pour Data Tool Climatique.

Ce module propose une fonction `run_climate_modeling` qui entraîne un modèle
simple (classification ou régression) à partir d’un DataFrame prétraité.

L’objectif est d’avoir quelque chose de **fonctionnel rapidement** pour un
hackathon, quitte à étendre ensuite le catalogue de modèles.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

# Importer les utilitaires communs
from clim_model_utils import detect_task_type, build_preprocessor, get_feature_importance, TaskType


ModelName = Literal["RandomForest", "GradientBoosting", "Linear/Logistic"]


def run_climate_modeling(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    model_name: ModelName = "RandomForest",
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    handle_imbalance: bool = False,
    use_time_validation: bool = False,
    date_col: str | None = None,
) -> Tuple[Pipeline, Dict[str, object]]:
    """Entraîne un modèle simple sur les données climatiques.

    Paramètres
    ----------
    df : DataFrame prétraité
    target_col : nom de la colonne cible
    test_size : proportion de test
    random_state : graine aléatoire

    Retourne
    --------
    pipeline : pipeline sklearn (préprocesseur + modèle)
    info : dict contenant X_train, X_test, y_train, y_test, metrics, task_type
    """

    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' absente du DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task_type: TaskType = detect_task_type(y)

    # Split train/test : on privilégie un split stratifié pour la classification,
    # mais on retombe automatiquement sur un split non stratifié si les classes
    # sont trop déséquilibrées (ce qui provoquerait une ValueError).
    stratify_arg = y if task_type == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )
        used_stratify = stratify_arg is not None
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        used_stratify = False

    preprocessor = build_preprocessor(X_train)

    # Gestion du déséquilibre de classes
    class_weight_arg = "balanced" if (handle_imbalance and task_type == "classification") else None
    
    if model_name == "RandomForest":
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state,
                class_weight=class_weight_arg
            )
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    elif model_name == "GradientBoosting":
        if task_type == "classification":
            model = GradientBoostingClassifier(random_state=random_state)
        else:
            model = GradientBoostingRegressor(random_state=random_state)
    else:  # "Linear/Logistic"
        if task_type == "classification":
            model = LogisticRegression(max_iter=1000, class_weight=class_weight_arg)
        else:
            model = LinearRegression()

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    # Probabilités pour classification (utile pour courbes PR/ROC)
    y_proba = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            pass

    if task_type == "classification":
        metric_value = accuracy_score(y_test, y_pred)
        metric_name = "accuracy"
        # Ajouter F1-score pour mieux évaluer le déséquilibre
        f1 = f1_score(y_test, y_pred, average="weighted")
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metric_value = rmse
        metric_name = "rmse"
        f1 = None
    
    # Validation temporelle si demandée et date disponible
    cv_scores = None
    if use_time_validation and date_col and date_col in df.columns:
        try:
            # Trier par date pour validation temporelle
            df_sorted = df.sort_values(date_col)
            X_sorted = df_sorted.drop(columns=[target_col])
            y_sorted = df_sorted[target_col]
            
            tscv = TimeSeriesSplit(n_splits=3)
            scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
            cv_scores = cross_val_score(pipeline, X_sorted, y_sorted, cv=tscv, scoring=scoring)
        except Exception:
            cv_scores = None
    
    # Feature importance (si modèle tree-based)
    feature_info = get_feature_importance(pipeline)
    if feature_info:
        feature_names, feature_importance = feature_info
    else:
        feature_names = None
        feature_importance = None

    info: Dict[str, object] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": task_type,
        "used_stratify": used_stratify,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "f1_score": f1,
        "cv_scores": cv_scores,
        "feature_importance": feature_importance,
        "feature_names": feature_names,
        "handle_imbalance": handle_imbalance,
        "use_time_validation": use_time_validation,
    }

    return pipeline, info
