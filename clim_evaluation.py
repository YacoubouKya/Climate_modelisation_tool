"""Évaluation simple pour Data Tool Climatique.

Ce module fournit une fonction `show_evaluation` qui affiche de manière
synthétique les performances du modèle entraîné (métrique principale +
quelques graphiques basiques).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report,
    precision_score,
    recall_score,
)


def analyze_by_segment(y_test: pd.Series, y_pred: pd.Series, segment_col: pd.Series, task_type: str) -> pd.DataFrame:
    """Analyse les performances par segment."""
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    
    df_analysis = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred,
        'segment': segment_col
    })
    
    results = []
    for segment in df_analysis['segment'].unique():
        segment_data = df_analysis[df_analysis['segment'] == segment]
        y_true_seg = segment_data['y_test']
        y_pred_seg = segment_data['y_pred']
        
        if task_type == "classification":
            acc = accuracy_score(y_true_seg, y_pred_seg)
            f1 = f1_score(y_true_seg, y_pred_seg, average='weighted', zero_division=0)
            results.append({
                'Segment': segment,
                'Nombre': len(segment_data),
                'Accuracy': acc,
                'F1-Score': f1
            })
        else:
            mse = mean_squared_error(y_true_seg, y_pred_seg)
            r2 = r2_score(y_true_seg, y_pred_seg)
            results.append({
                'Segment': segment,
                'Nombre': len(segment_data),
                'RMSE': np.sqrt(mse),
                'R²': r2
            })
    
    return pd.DataFrame(results)


def show_evaluation(info: dict) -> None:
    """Affiche l’évaluation à partir du dictionnaire retourné par clim_modeling.

    `info` doit contenir au minimum :
    - task_type : "classification" ou "regression"
    - metric_name, metric_value
    - y_test, y_pred
    """

    task_type = info.get("task_type", "regression")
    metric_name = info.get("metric_name", "score")
    metric_value = info.get("metric_value", None)
    y_test = info.get("y_test")
    y_pred = info.get("y_pred")

    st.subheader("📊 Résultats globaux")
    
    # Afficher les métriques principales
    col1, col2, col3 = st.columns(3)
    with col1:
        if metric_value is not None:
            st.metric(label=metric_name.upper(), value=f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else metric_value)
    
    with col2:
        model_name = info.get("model_name", "N/A")
        st.metric("Modèle", model_name)
    
    with col3:
        cv_scores = info.get("cv_scores")
        if cv_scores is not None and len(cv_scores) > 0:
            st.metric("CV Score", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if y_test is None or y_pred is None:
        return

    y_test = pd.Series(y_test)
    y_pred = pd.Series(y_pred, index=y_test.index)

    if task_type == "classification":
        st.subheader("🧩 Matrice de confusion")
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Valeurs réelles")
        st.pyplot(fig)
        
        # Métriques détaillées
        st.subheader("📊 Métriques détaillées")
        try:
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Precision (weighted)", f"{precision:.4f}")
            with col2:
                st.metric("Recall (weighted)", f"{recall:.4f}")
            
            # Rapport de classification
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de calculer les métriques détaillées : {e}")
        
        # Courbes PR et ROC (si classification binaire et probabilités disponibles)
        if len(labels) == 2 and "y_proba" in info:
            y_proba = info["y_proba"]
            if y_proba is not None and len(y_proba.shape) == 2:
                st.subheader("📈 Courbes PR et ROC")
                
                # Courbe Precision-Recall
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1])
                pr_auc = auc(recall_curve, precision_curve)
                
                # Courbe ROC
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # PR Curve
                ax1.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.3f}")
                ax1.set_xlabel("Recall")
                ax1.set_ylabel("Precision")
                ax1.set_title("Courbe Precision-Recall")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # ROC Curve
                ax2.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
                ax2.plot([0, 1], [0, 1], "k--", label="Random")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("Courbe ROC")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        # Analyse par segment (si X_test disponible)
        if "X_test" in info:
            X_test = info["X_test"]
            if isinstance(X_test, pd.DataFrame):
                cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_cols:
                    st.subheader("🔍 Analyse par segment")
                    segment_col = st.selectbox("Colonne de segmentation", options=cat_cols)
                    
                    if segment_col:
                        # Créer un DataFrame avec les résultats
                        eval_df = pd.DataFrame({
                            "segment": X_test[segment_col],
                            "y_test": y_test,
                            "y_pred": y_pred
                        })
                        
                        # Calculer l'accuracy par segment
                        segment_acc = eval_df.groupby("segment").apply(
                            lambda g: (g["y_test"] == g["y_pred"]).mean()
                        ).reset_index(name="accuracy")
                        
                        st.dataframe(segment_acc, use_container_width=True)
                        
                        # Graphique
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(segment_acc["segment"].astype(str), segment_acc["accuracy"])
                        ax.set_xlabel(segment_col)
                        ax.set_ylabel("Accuracy")
                        ax.set_title(f"Performance par {segment_col}")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)

    else:
        st.subheader("📈 Prédictions vs valeurs réelles")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.6)
        min_val = float(min(y_test.min(), y_pred.min()))
        max_val = float(max(y_test.max(), y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Prédictions")
        st.pyplot(fig)

        st.subheader("📉 Résidus")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_xlabel("Résidus (réel - prédit)")
        st.pyplot(fig2)
        
        # Métriques orientées risque pour régression
        st.subheader("⚠️ Métriques orientées risque")
        mae = np.abs(residuals).mean()
        rmse = np.sqrt((residuals ** 2).mean())
        mape = (np.abs(residuals / y_test) * 100).mean() if (y_test != 0).all() else np.nan
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{mae:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            if not np.isnan(mape):
                st.metric("MAPE (%)", f"{mape:.2f}")
        
        # Analyse par segment pour régression
        if "X_test" in info:
            X_test = info["X_test"]
            if isinstance(X_test, pd.DataFrame):
                cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
                if cat_cols:
                    st.subheader("🔍 Analyse par segment")
                    segment_col = st.selectbox("Colonne de segmentation", options=cat_cols)
                    
                    if segment_col:
                        eval_df = pd.DataFrame({
                            "segment": X_test[segment_col],
                            "y_test": y_test,
                            "y_pred": y_pred
                        })
                        
                        # Calculer RMSE par segment
                        segment_rmse = eval_df.groupby("segment").apply(
                            lambda g: np.sqrt(((g["y_test"] - g["y_pred"]) ** 2).mean())
                        ).reset_index(name="rmse")
                        
                        st.dataframe(segment_rmse, use_container_width=True)
                        
                        # Graphique
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(segment_rmse["segment"].astype(str), segment_rmse["rmse"])
                        ax.set_xlabel(segment_col)
                        ax.set_ylabel("RMSE")
                        ax.set_title(f"Erreur par {segment_col}")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)
