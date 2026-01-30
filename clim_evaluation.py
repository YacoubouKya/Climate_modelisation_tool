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
    
    # Afficher les 4 métriques principales selon le type de tâche avec un design amélioré
    if task_type == "classification":
        col1, col2, col3, col4 = st.columns(4)
        
        # Accuracy
        with col1:
            if metric_value is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">ACCURACY</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{metric_value:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Performance globale</div>
                </div>
                """, unsafe_allow_html=True)
        
        # F1-Score
        with col2:
            f1_score = info.get("f1_score")
            if f1_score is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">F1-SCORE</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{f1_score:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Balance précision/rappel</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Precision
        with col3:
            precision = info.get("precision")
            if precision is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">PRECISION</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{precision:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Prédictions positives</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recall
        with col4:
            recall = info.get("recall")
            if recall is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">RECALL</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{recall:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Détection positive</div>
                </div>
                """, unsafe_allow_html=True)
    
    else:  # Régression
        col1, col2, col3, col4 = st.columns(4)
        
        # RMSE (métrique principale)
        with col1:
            if metric_value is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">RMSE</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{metric_value:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Erreur quadratique</div>
                </div>
                """, unsafe_allow_html=True)
        
        # MSE
        with col2:
            mse = info.get("mse")
            if mse is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">MSE</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{mse:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Erreur moyenne</div>
                </div>
                """, unsafe_allow_html=True)
        
        # MAE
        with col3:
            mae = info.get("mae")
            if mae is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">MAE</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{mae:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Erreur absolue</div>
                </div>
                """, unsafe_allow_html=True)
        
        # R²
        with col4:
            r2 = info.get("r2")
            if r2 is not None:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; color: #ffffff; margin-bottom: 8px; font-weight: 600;">R²</div>
                    <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{r2:.4f}</div>
                    <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px;">Qualité d'ajustement</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Afficher le nom du modèle avec un design spectaculaire
    st.markdown("---")
    model_name = info.get("model_name", "N/A")
    cv_scores = info.get("cv_scores")
    
    # Section principale du modèle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="display: inline-block; padding: 8px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 20px; font-size: 14px; font-weight: 600; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);">
            🤖 INFORMATIONS DU MODÈLE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Carte principale du modèle
        st.markdown(f"""
        <div style="padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50%; right: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);"></div>
            <div style="position: relative; z-index: 1;">
                <div style="font-size: 18px; color: rgba(255,255,255,0.9); margin-bottom: 12px; font-weight: 600; letter-spacing: 1px;">ALGORITHME UTILISÉ</div>
                <div style="font-size: 32px; font-weight: bold; color: #ffffff; margin-bottom: 15px; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">{model_name}</div>
                <div style="display: flex; align-items: center; justify-content: flex-start; margin-top: 20px;">
                    <div style="width: 12px; height: 12px; background: #4ade80; border-radius: 50%; margin-right: 10px; animation: pulse 2s infinite;"></div>
                    <div style="font-size: 14px; color: rgba(255,255,255,0.8);">Modèle entraîné et prêt</div>
                </div>
            </div>
        </div>
        <style>
        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.1); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    with col2:
        if cv_scores is not None and len(cv_scores) > 0:
            # Carte des scores de validation
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            cv_min = cv_scores.min()
            cv_max = cv_scores.max()
            
            st.markdown(f"""
            <div style="padding: 25px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 20px; box-shadow: 0 8px 25px rgba(67, 233, 123, 0.3); position: relative; overflow: hidden;">
                <div style="position: absolute; top: -30%; left: -30%; width: 160%; height: 160%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);"></div>
                <div style="position: relative; z-index: 1;">
                    <div style="font-size: 16px; color: rgba(255,255,255,0.9); margin-bottom: 15px; font-weight: 600; letter-spacing: 1px;">VALIDATION CROISÉE</div>
                    
                    <div style="background: rgba(255,255,255,0.2); border-radius: 12px; padding: 15px; margin-bottom: 15px; backdrop-filter: blur(10px);">
                        <div style="font-size: 28px; font-weight: bold; color: #ffffff; text-align: center; margin-bottom: 8px;">{cv_mean:.4f}</div>
                        <div style="font-size: 12px; color: rgba(255,255,255,0.8); text-align: center;">Score moyen</div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px; color: rgba(255,255,255,0.9);">
                        <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 8px; text-align: center;">
                            <div style="font-weight: 600;">± {cv_std:.4f}</div>
                            <div style="font-size: 10px; opacity: 0.8;">Écart-type</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 8px; text-align: center;">
                            <div style="font-weight: 600;">{len(cv_scores)} folds</div>
                            <div style="font-size: 10px; opacity: 0.8;">Validations</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 12px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 8px; font-size: 11px; color: rgba(255,255,255,0.8); text-align: center;">
                        Plage: {cv_min:.4f} - {cv_max:.4f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Carte alternative si pas de CV
            st.markdown(f"""
            <div style="padding: 25px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3); position: relative; overflow: hidden;">
                <div style="position: absolute; top: -30%; left: -30%; width: 160%; height: 160%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);"></div>
                <div style="position: relative; z-index: 1; text-align: center;">
                    <div style="font-size: 16px; color: rgba(255,255,255,0.9); margin-bottom: 15px; font-weight: 600;">STATUT</div>
                    <div style="font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 10px;">Entraînement simple</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.8);">Pas de validation croisée</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
        
        # Métriques détaillées avec design amélioré
        st.subheader("📊 Métriques détaillées")
        try:
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            
            # Affichage des métriques avec un design moderne
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 16px; color: #ffffff; margin-bottom: 10px; font-weight: 600;">PRECISION (WEIGHTED)</div>
                    <div style="font-size: 32px; font-weight: bold; color: #ffffff;">{precision:.4f}</div>
                    <div style="font-size: 14px; color: #e0e0e0; margin-top: 8px;">Qualité des prédictions positives</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <div style="font-size: 16px; color: #ffffff; margin-bottom: 10px; font-weight: 600;">RECALL (WEIGHTED)</div>
                    <div style="font-size: 32px; font-weight: bold; color: #ffffff;">{recall:.4f}</div>
                    <div style="font-size: 14px; color: #e0e0e0; margin-top: 8px;">Taux de détection positif</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Rapport de classification avec un meilleur affichage
            st.markdown("### 📋 Rapport de Classification Détaillé")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            
            # Améliorer l'affichage du dataframe
            st.dataframe(
                report_df.round(4), 
                use_container_width=True,
                height=400,
                hide_index=True
            )
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
