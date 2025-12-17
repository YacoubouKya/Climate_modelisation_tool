"""Utilitaires communs pour la modélisation - Data Tool Climatique.

Ce module centralise les fonctions partagées entre clim_modeling et clim_model_comparison
pour éviter la duplication de code.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TaskType = Literal["classification", "regression"]


def detect_task_type(y: pd.Series) -> TaskType:
    """Détecte automatiquement le type de tâche à partir de la cible.
    
    Règles :
    - Si dtype est object/category OU si nombre de valeurs uniques <= 20 
      ET ratio unique/total < 0.1 → classification
    - Sinon → régression
    
    Parameters
    ----------
    y : pd.Series
        Série cible
        
    Returns
    -------
    TaskType
        "classification" ou "regression"
        
    Raises
    ------
    ValueError
        Si y est vide ou invalide
    """
    if y is None or len(y) == 0:
        raise ValueError("La série cible est vide")
    
    if y.isna().all():
        raise ValueError("La série cible ne contient que des valeurs manquantes")
    
    if y.dtype == "O" or y.dtype.name == "category":
        return "classification"
    
    n_unique = y.nunique(dropna=True)
    n_total = len(y)
    
    if n_total == 0:
        raise ValueError("La série cible ne contient aucune valeur valide")
    
    # Classification si peu de valeurs uniques ET faible ratio
    if n_unique <= 20 and (n_unique / n_total) < 0.1:
        return "classification"
    
    return "regression"


def build_preprocessor(X: pd.DataFrame, do_scale: bool = True, handle_high_cardinality: bool = True) -> ColumnTransformer:
    """Construit un préprocesseur robuste pour les données.
    
    Features :
    - Imputation des valeurs manquantes
    - Scaling optionnel des numériques
    - OneHotEncoding des catégorielles
    - Gestion optionnelle de la haute cardinalité (>100 valeurs)
    
    Parameters
    ----------
    X : pd.DataFrame
        Features à prétraiter
    do_scale : bool, default=True
        Appliquer StandardScaler sur les numériques
    handle_high_cardinality : bool, default=True
        Si True, limite le nombre de catégories encodées
        
    Returns
    -------
    ColumnTransformer
        Préprocesseur sklearn
        
    Raises
    ------
    ValueError
        Si X est vide ou invalide
    """
    if X is None or X.empty:
        raise ValueError("Le DataFrame X est vide")
    
    if X.shape[1] == 0:
        raise ValueError("Le DataFrame X n'a aucune colonne")
    
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []

    # Transformer numérique
    if numeric_features:
        if do_scale:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
        else:
            numeric_transformer = SimpleImputer(strategy="median")
        transformers.append(("num", numeric_transformer, numeric_features))

    # Transformer catégoriel
    if categorical_features:
        # Filtrer les colonnes à haute cardinalité si demandé
        if handle_high_cardinality:
            low_card_features = [col for col in categorical_features if X[col].nunique() <= 100]
            if low_card_features:
                categorical_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
                    ]
                )
                transformers.append(("cat", categorical_transformer, low_card_features))
        else:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extrait les noms de features d'un pipeline sklearn.
    
    Utile pour la feature importance.
    
    Parameters
    ----------
    pipeline : Pipeline
        Pipeline sklearn avec un preprocessor
        
    Returns
    -------
    list[str]
        Liste des noms de features après preprocessing
    """
    try:
        if not hasattr(pipeline, 'named_steps'):
            return []
        
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
            return preprocessor.get_feature_names_out().tolist()
    except Exception as e:
        # Log l'erreur mais ne pas crasher
        import warnings
        warnings.warn(f"Impossible d'extraire les noms de features : {e}")
    return []


def get_feature_importance(pipeline: Pipeline) -> tuple[list[str], list[float]] | None:
    """Extrait la feature importance d'un pipeline avec modèle tree-based.
    
    Parameters
    ----------
    pipeline : Pipeline
        Pipeline sklearn avec modèle
        
    Returns
    -------
    tuple[list[str], list[float]] | None
        (noms_features, importances) ou None si non disponible
    """
    try:
        if not hasattr(pipeline, 'named_steps'):
            return None
        
        model = pipeline.named_steps.get("model")
        if not model or not hasattr(model, "feature_importances_"):
            return None
        
        feature_names = get_feature_names_from_pipeline(pipeline)
        feature_importance = model.feature_importances_
        
        if not feature_names:
            return None
        
        if len(feature_names) != len(feature_importance):
            import warnings
            warnings.warn(f"Incompatibilité : {len(feature_names)} features vs {len(feature_importance)} importances")
            return None
        
        return feature_names, feature_importance.tolist()
    except Exception as e:
        import warnings
        warnings.warn(f"Impossible d'extraire la feature importance : {e}")
    return None
