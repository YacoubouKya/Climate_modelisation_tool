"""Module de comparaison de modèles pour Data Tool Climatique.

Permet d'entraîner et comparer plusieurs modèles ML pour le risque climatique.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Importer les utilitaires communs
from clim_model_utils import detect_task_type, build_preprocessor


def get_available_models(task: str, fast_mode: bool = False) -> Dict[str, Any]:
    """Retourne les modèles disponibles selon la tâche."""
    if task == "classification":
        if fast_mode:
            return {
                "Random Forest": RandomForestClassifier(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=500, random_state=42, n_jobs=-1
                ),
                "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
                "Extra Trees": ExtraTreesClassifier(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
            }
        else:
            return {
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42, n_jobs=-1
                ),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
                "Naive Bayes": GaussianNB(),
            }
    else:  # regression
        if fast_mode:
            return {
                "Random Forest": RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=42
                ),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "Ridge": Ridge(random_state=42),
                "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
                "Extra Trees": ExtraTreesRegressor(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                ),
            }
        else:
            return {
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
            }


def train_and_evaluate_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    task: str,
    use_cv: bool = False,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Entraîne et évalue un modèle."""
    start_time = time.time()

    try:
        # Créer le pipeline
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # Entraîner
        pipe.fit(X_train, y_train)

        # Prédictions
        y_pred = pipe.predict(X_test)
        y_train_pred = pipe.predict(X_train)
        
        # Probabilités pour classification
        y_proba = None
        if task == "classification" and hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None

        # Métriques
        if task == "classification":
            test_score = accuracy_score(y_test, y_pred)
            train_score = accuracy_score(y_train, y_train_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            rmse = None
            metric_name = "Accuracy"
        else:
            test_score = r2_score(y_test, y_pred)
            train_score = r2_score(y_train, y_train_pred)
            f1 = None
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            metric_name = "R²"

        # Cross-validation
        cv_scores = None
        if use_cv:
            try:
                scoring = "accuracy" if task == "classification" else "r2"
                cv_scores = cross_val_score(
                    pipe, X_train, y_train, cv=cv_folds, scoring=scoring
                )
            except Exception:
                cv_scores = None

        result = {
            "model_name": model_name,
            "pipeline": pipe,
            "test_score": test_score,
            "train_score": train_score,
            "f1_score": f1,
            "rmse": rmse,
            "cv_scores": cv_scores,
            "training_time": time.time() - start_time,
            "metric_name": metric_name,
            "success": True,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    except Exception as e:
        result = {
            "model_name": model_name,
            "pipeline": None,
            "test_score": None,
            "train_score": None,
            "f1_score": None,
            "rmse": None,
            "cv_scores": None,
            "training_time": time.time() - start_time,
            "metric_name": None,
            "success": False,
            "error": str(e),
        }

    return result


def validate_data_for_modeling(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str], List[str]]:
    """Valide les données avant modélisation."""
    errors = []
    warnings = []
    
    # Vérifications critiques
    if X.shape[0] == 0:
        errors.append("❌ DataFrame X vide (0 lignes)")
    if X.shape[1] == 0:
        errors.append("❌ DataFrame X sans features")
    if len(y) == 0:
        errors.append("❌ Variable cible y vide")
    if len(X) != len(y):
        errors.append(f"❌ Incompatibilité : X={len(X)} lignes, y={len(y)} valeurs")
    
    # Vérifications non-bloquantes
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        if len(nan_cols) <= 5:
            warnings.append(f"⚠️ {len(nan_cols)} colonne(s) avec NaN : {', '.join(nan_cols)}")
        else:
            warnings.append(f"⚠️ {len(nan_cols)} colonnes avec NaN")
    
    # Haute cardinalité
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    high_card = [col for col in cat_cols if X[col].nunique() > 100]
    if high_card:
        warnings.append(f"⚠️ {len(high_card)} colonne(s) à haute cardinalité (> 100 valeurs)")
    
    # Taille du dataset
    total_size_mb = (X.memory_usage(deep=True).sum() + y.memory_usage(deep=True)) / 1024 / 1024
    if total_size_mb > 500:
        warnings.append(f"⚠️ Dataset volumineux ({total_size_mb:.1f} MB)")
    
    return len(errors) == 0, errors, warnings


def compare_models(
    df: pd.DataFrame,
    target_col: str,
    task: str = "auto",
    test_size: float = 0.2,
    selected_models: List[str] = None,
    fast_mode: bool = False,
    use_cv: bool = False,
    cv_folds: int = 5,
    handle_imbalance: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Compare plusieurs modèles ML avec validation robuste."""

    # Préparer les données
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Validation des données
    is_valid, errors, warnings = validate_data_for_modeling(X, y)
    
    if not is_valid:
        st.error("❌ **Validation échouée**")
        for error in errors:
            st.error(error)
        st.stop()
    
    if warnings:
        st.markdown("---")
        show_warnings = st.checkbox("⚠️ Afficher les avertissements de validation", value=False, key="show_validation_warnings")
        if show_warnings:
            for warning in warnings:
                st.warning(warning)
            st.info("💡 Ces avertissements n'empêchent pas l'entraînement")
        st.markdown("---")

    # Détecter le type de tâche
    if task == "auto":
        task = detect_task_type(y)

    # Détection de gros dataset
    dataset_size_mb = (X.memory_usage(deep=True).sum() + y.memory_usage(deep=True)) / 1024 / 1024
    n_rows = len(X)
    
    if dataset_size_mb > 5 or n_rows > 10000:
        st.warning(f"⚠️ Dataset volumineux : {n_rows:,} lignes, {dataset_size_mb:.1f} MB")
        st.info(f"💡 Entraînement de {len(selected_models) if selected_models else 'plusieurs'} modèles - Peut prendre 2-5 minutes")
        
        # Désactiver CV pour gros datasets
        if use_cv:
            st.warning("⚠️ Validation croisée désactivée pour éviter les timeouts")
            use_cv = False

    # Split
    try:
        if task == "classification" and not handle_imbalance:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Construire le preprocesseur
    preprocessor = build_preprocessor(X_train)

    # Obtenir les modèles disponibles
    available_models = get_available_models(task, fast_mode)

    # Filtrer les modèles sélectionnés
    if selected_models:
        available_models = {k: v for k, v in available_models.items() if k in selected_models}

    # Gérer le déséquilibre si demandé
    if handle_imbalance and task == "classification":
        for model_name, model in available_models.items():
            if hasattr(model, "class_weight"):
                model.class_weight = "balanced"

    # Entraîner et évaluer chaque modèle
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    start_total = time.time()

    for i, (model_name, model) in enumerate(available_models.items()):
        status_text.text(f"🔄 Entraînement : {model_name}... ({i+1}/{len(available_models)})")
        
        model_start = time.time()
        result = train_and_evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            task,
            use_cv,
            cv_folds,
        )
        model_time = time.time() - model_start
        
        results.append(result)
        
        if result["success"]:
            time_text.text(f"⏱️ {model_name} : {model_time:.1f}s")
        else:
            time_text.text(f"❌ {model_name} : Échec")
        
        progress_bar.progress((i + 1) / len(available_models))
    
    total_time = time.time() - start_total
    status_text.text(f"✅ Entraînement terminé en {total_time:.1f}s!")
    time_text.empty()
    progress_bar.empty()

    return results, task


def display_comparison_results(results: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Affiche les résultats de comparaison avec graphiques."""

    # Filtrer les modèles réussis
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    if not successful_results:
        st.error("❌ Aucun modèle n'a pu être entraîné avec succès.")
        return None

    # Créer un DataFrame des résultats
    results_data = []
    for r in successful_results:
        row = {
            "Modèle": r["model_name"],
            "Score Test": r["test_score"],
            "Score Train": r["train_score"],
            "Temps (s)": r["training_time"],
        }
        
        if task == "classification" and r["f1_score"] is not None:
            row["F1-Score"] = r["f1_score"]
        elif task == "regression" and r["rmse"] is not None:
            row["RMSE"] = r["rmse"]
        
        if r["cv_scores"] is not None:
            row["CV Mean"] = r["cv_scores"].mean()
            row["CV Std"] = r["cv_scores"].std()
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values("Score Test", ascending=False).reset_index(drop=True)

    # Identifier le meilleur modèle
    best_result = successful_results[np.argmax([r["test_score"] for r in successful_results])]
    
    # Afficher le meilleur modèle en haut
    st.success(
        f"🏆 **Meilleur modèle** : {best_result['model_name']} "
        f"({best_result['metric_name']} = {best_result['test_score']:.4f})"
    )

    # Afficher le tableau
    st.subheader("📊 Tableau de comparaison")
    st.dataframe(results_df.style.format({
        "Score Test": "{:.4f}",
        "Score Train": "{:.4f}",
        "Temps (s)": "{:.2f}",
        "F1-Score": "{:.4f}" if "F1-Score" in results_df.columns else None,
        "RMSE": "{:.4f}" if "RMSE" in results_df.columns else None,
        "CV Mean": "{:.4f}" if "CV Mean" in results_df.columns else None,
        "CV Std": "{:.4f}" if "CV Std" in results_df.columns else None,
    }), use_container_width=True)

    # Graphiques de comparaison
    st.markdown("---")
    st.subheader("📈 Visualisations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique 1: Score Test
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if model == best_result['model_name'] else '#3498db' 
                  for model in results_df["Modèle"]]
        ax1.barh(results_df["Modèle"], results_df["Score Test"], color=colors)
        ax1.set_xlabel(best_result['metric_name'])
        ax1.set_title(f"Comparaison - {best_result['metric_name']}")
        ax1.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        # Graphique 2: Temps d'entraînement
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(results_df["Modèle"], results_df["Temps (s)"], color='#e74c3c')
        ax2.set_xlabel("Temps (secondes)")
        ax2.set_title("Temps d'Entraînement")
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    # Heatmap des métriques (si plusieurs métriques disponibles)
    metric_cols = [col for col in results_df.columns if col not in ["Modèle", "Temps (s)"]]
    if len(metric_cols) >= 2:
        st.markdown("---")
        st.subheader("🔥 Heatmap des métriques")
        
        fig3, ax3 = plt.subplots(figsize=(10, len(results_df) * 0.5 + 2))
        heatmap_data = results_df[["Modèle"] + metric_cols].set_index("Modèle")
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=heatmap_data.mean().mean(),
            ax=ax3,
            cbar_kws={'label': 'Score'}
        )
        ax3.set_title("Heatmap des Métriques par Modèle")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # Afficher les erreurs si présentes
    if failed_results:
        st.markdown("---")
        show_errors = st.checkbox(f"⚠️ Afficher les {len(failed_results)} modèle(s) échoué(s)", value=False, key="show_failed_models")
        if show_errors:
            for r in failed_results:
                st.error(f"**{r['model_name']}** : {r['error']}")

    return best_result
