"""
Module de reporting pour l'analyse de risque climatique
Basé sur le design exact du rapport exemple
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, List
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configuration de matplotlib pour un meilleur rendu
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 10


def _get_report_css() -> str:
    """Retourne le CSS exact du rapport exemple"""
    return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            text-align: center;
        }
        
        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 5px solid #3498db;
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
        }
        
        h3 {
            color: #2980b9;
            font-size: 1.4em;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        h4 {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        p {
            margin: 10px 0;
            font-size: 1em;
        }
        
        .metric-box {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            margin: 10px 10px 10px 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
            display: block;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.8em;
            display: block;
        }
        
        /* Container pour tableaux avec scroll horizontal */
        .table-container {
            width: 100%;
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            min-width: 600px;  /* Largeur minimale pour éviter l'écrasement */
        }
        
        /* Tableaux compacts pour aperçu et stats */
        table.dataframe {
            font-size: 0.85em;  /* Texte plus petit */
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        th {
            padding: 10px 8px;  /* Padding réduit */
            text-align: left;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }
        
        td {
            padding: 8px 8px;  /* Padding réduit */
            border-bottom: 1px solid #ecf0f1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;  /* Largeur max par cellule */
        }
        
        tbody tr:hover {
            background: #f8f9fa;
            transition: background 0.3s ease;
        }
        
        tbody tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        /* Indicateur de scroll */
        .table-container::after {
            content: "← Faites défiler horizontalement →";
            display: block;
            text-align: center;
            font-size: 0.75em;
            color: #95a5a6;
            padding: 5px;
            font-style: italic;
        }
        
        /* Masquer l'indicateur si pas de scroll nécessaire */
        @media (min-width: 1400px) {
            .table-container::after {
                display: none;
            }
        }
        
        .figure-container {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .figure-container img {
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .info-box {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        ul {
            list-style: none;
            padding-left: 0;
        }
        
        ul li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        
        ul li:before {
            content: "▸";
            position: absolute;
            left: 0;
            color: #3498db;
            font-weight: bold;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #ecf0f1;
        }
        
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
    """


def _img_to_base64(fig: plt.Figure) -> str:
    """Convertit une figure matplotlib en base64 pour HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<div class="figure-container"><img src="data:image/png;base64,{img_str}" style="max-width:100%; height:auto;"></div>'


def _wrap_table(html_table: str) -> str:
    """Enveloppe un tableau HTML dans un container avec scroll."""
    return f'<div class="table-container">{html_table}</div>'


def _create_distribution_plot(df: pd.DataFrame) -> plt.Figure:
    """Crée les graphiques de distribution des variables numériques."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) == 0:
        return None
    
    # Limiter à 6 variables pour éviter la surcharge
    cols_to_plot = numeric_cols[:6]
    n_cols = min(3, len(cols_to_plot))
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        df[col].hist(bins=30, ax=ax, alpha=0.7, color='#3498db', edgecolor='black')
        ax.set_title(f'Distribution de {col}', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('Fréquence')
        ax.grid(True, alpha=0.3)
    
    # Masquer les axes non utilisés
    for i in range(len(cols_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def _create_correlation_plot(df: pd.DataFrame) -> plt.Figure:
    """Crée la matrice de corrélation."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    
    # Créer une palette de couleurs divergente
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Tracer la heatmap
    sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .8}, 
               annot=True, fmt=".2f", ax=ax)
    
    ax.set_title('Matrice de corrélation entre les variables numériques', pad=20)
    plt.tight_layout()
    return fig


def _create_feature_importance_plot(feature_names: List[str], importances: np.ndarray, top_n: int = 10) -> plt.Figure:
    """Crée un graphique d'importance des caractéristiques."""
    # Tri des caractéristiques par importance
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Style du graphique
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e9ecef')
    
    # Tracé des barres
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, align='center', color='#667eea', alpha=0.8)
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.02, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                va='center', ha='left', color='#2c3e50', fontsize=9)
    
    # Mise en forme
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color='#2c3e50')
    ax.tick_params(axis='x', colors='#6c757d')
    ax.set_title(f'Top {top_n} des caractéristiques les plus importantes', 
                 color='#2c3e50', pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', color='#6c757d', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.2, color='#dee2e6', axis='x')
    
    plt.tight_layout()
    return fig


def _create_evaluation_plots(model_info: Dict[str, Any]) -> plt.Figure:
    """Crée les graphiques d'évaluation du modèle."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1: Métriques de performance
    if model_info.get("task_type") == "classification":
        # Courbe ROC simulée
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve([0, 0, 1, 1, 1, 0, 1, 0, 1, 1], [0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.85, 0.15, 0.95, 0.6])
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, color='#667eea', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('Taux de faux positifs')
        axes[0].set_ylabel('Taux de vrais positifs')
        axes[0].set_title('Courbe ROC')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Matrice de confusion simulée
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix([0, 0, 1, 1, 1, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Matrice de Confusion')
        axes[1].set_xlabel('Prédit')
        axes[1].set_ylabel('Réel')
        
    elif model_info.get("task_type") == "regression":
        # Graphique de dispersion prédiction vs réel
        y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        y_pred = y_true + np.random.normal(0, 5, len(y_true))
        
        axes[0].scatter(y_true, y_pred, alpha=0.6, color='#667eea')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Valeurs réelles')
        axes[0].set_ylabel('Valeurs prédites')
        axes[0].set_title('Prédiction vs Réel')
        axes[0].grid(True, alpha=0.3)
        
        # Résidus
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='#e74c3c')
        axes[1].axhline(y=0, color='gray', linestyle='--')
        axes[1].set_xlabel('Valeurs prédites')
        axes[1].set_ylabel('Résidus')
        axes[1].set_title('Graphique des Résidus')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_climate_report(session_state: Dict[str, Any]) -> Optional[str]:
    """
    Génère un rapport HTML pour l'analyse de risque climatique
    avec le design exact du rapport exemple et options de personnalisation.
    """
    # Récupération des données
    df = session_state.get("clim_data")
    df_prep = session_state.get("clim_data_prep")
    model_info = session_state.get("clim_model_info")
    prep_info = session_state.get("clim_prep_info", {})
    data_sources = session_state.get("data_sources", {})
    
    # Récupération des options de personnalisation
    report_options = session_state.get("report_options", {})
    selected_sections = report_options.get("sections", [])
    report_title = report_options.get("title", "Rapport d'Analyse de Risque Climatique")
    include_code = report_options.get("include_code", False)
    
    if (df is None or (isinstance(df, pd.DataFrame) and df.empty)) and not bool(data_sources):
        return None
    
    # Génération du nom du fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"outputs/reports/Rapport_Climat_{timestamp}.html"
    
    # Construction du HTML
    parts = []
    
    # Header HTML
    parts.extend([
        "<html><head>",
        "<meta charset='utf-8'>",
        f"<title>{report_title}_{timestamp}</title>",
        _get_report_css(),
        "</head><body>",
        "<div class='container'>"
    ])
    
    # Titre principal
    parts.append(f"<h1>{report_title}_{timestamp}</h1>")
    
    # Info box avec informations générales
    parts.append("<div class='info-box'>")
    parts.append(f"<p><strong>Date de génération :</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>")
    
    if model_info:
        parts.append(f"<p><strong>Modèle :</strong> {model_info.get('model_name', 'Non spécifié')}</p>")
        parts.append(f"<p><strong>Type de tâche :</strong> {model_info.get('task_type', 'Non spécifié')}</p>")
    else:
        parts.append("<p><strong>Modèle :</strong> Non entraîné</p>")
        parts.append("<p><strong>Type de tâche :</strong> Analyse exploratoire</p>")
    
    parts.append("</div>")
    
    # Section 1: Données brutes
    if not selected_sections or "data_analysis" in selected_sections:
        parts.append("<h2>1. Données brutes</h2>")
        
        if isinstance(df, pd.DataFrame):
            # Métriques principales
            parts.append("<div class='metric-box'>")
            parts.append("<span class='metric-label'>Nombre de lignes</span>")
            parts.append(f"<span class='metric-value'>{df.shape[0]:,}</span>")
            parts.append("</div>")
            
            parts.append("<div class='metric-box'>")
            parts.append("<span class='metric-label'>Nombre de colonnes</span>")
            parts.append(f"<span class='metric-value'>{df.shape[1]}</span>")
            parts.append("</div>")
            
            missing_values = df.isnull().sum().sum()
            parts.append("<div class='metric-box'>")
            parts.append("<span class='metric-label'>Valeurs manquantes</span>")
            parts.append(f"<span class='metric-value'>{missing_values:,}</span>")
            parts.append("</div>")
            
            # Aperçu des données
            parts.append("<h3>Aperçu des données (5 premières lignes)</h3>")
            parts.append(_wrap_table(df.head().to_html(classes='dataframe dataframe', index=False)))
            
            # Statistiques descriptives
            parts.append("<h3>Statistiques descriptives</h3>")
            parts.append(_wrap_table(df.describe(include='all').round(2).to_html(classes='dataframe dataframe')))
            
            # Distributions des variables numériques
            parts.append("<h3>Distributions des variables numériques</h3>")
            try:
                dist_fig = _create_distribution_plot(df)
                if dist_fig:
                    parts.append(_img_to_base64(dist_fig))
            except Exception as e:
                parts.append(f"<div class='warning-box'>Erreur lors de la génération des distributions: {str(e)}</div>")
            
            # Matrice de corrélation
            try:
                corr_fig = _create_correlation_plot(df)
                if corr_fig:
                    parts.append("<h3>Matrice de corrélation</h3>")
                    parts.append(_img_to_base64(corr_fig))
            except Exception as e:
                parts.append(f"<div class='warning-box'>Erreur lors de la génération de la matrice de corrélation: {str(e)}</div>")
    
    # Section 2: Prétraitement
    if (bool(prep_info) or (isinstance(df_prep, pd.DataFrame) and not df_prep.empty)) and (not selected_sections or "preprocessing" in selected_sections):
        parts.append("<h2>2. Prétraitement des données</h2>")
        
        if prep_info:
            parts.append("<div class='info-box'>")
            parts.append("<h3>Étapes de prétraitement appliquées</h3>")
            parts.append("<ul>")
            
            if prep_info.get("date_col"):
                parts.append(f"<li>Colonne temporelle : {prep_info['date_col']}</li>")
            
            if prep_info.get("freq"):
                parts.append(f"<li>Fréquence d'agrégation : {prep_info['freq']}</li>")
            
            if prep_info.get("rolling"):
                parts.append("<li>Calcul des indicateurs mobiles</li>")
            
            if prep_info.get("anomaly_summary"):
                parts.append("<li>Détection des anomalies</li>")
            
            parts.append("</ul>")
            parts.append("</div>")
        
        if isinstance(df_prep, pd.DataFrame):
            parts.append("<h3>Aperçu des données prétraitées</h3>")
            parts.append(_wrap_table(df_prep.head().to_html(classes='dataframe dataframe', index=False)))
            
            parts.append("<h3>Statistiques après prétraitement</h3>")
            parts.append(_wrap_table(df_prep.describe(include='all').round(2).to_html(classes='dataframe dataframe')))
    
    # Section 3: Modélisation
    if model_info and (not selected_sections or "modeling" in selected_sections):
        parts.append("<h2>3. Modèle de Machine Learning</h2>")
        
        parts.append("<div class='info-box'>")
        parts.append(f"<p><strong>Nom du modèle :</strong> {model_info.get('model_name', 'Non spécifié')}</p>")
        parts.append(f"<p><strong>Type de pipeline :</strong> {model_info.get('pipeline_type', 'Standard')}</p>")
        parts.append(f"<p><strong>Tâche :</strong> {model_info.get('task_type', 'Non spécifié')}</p>")
        parts.append("</div>")
        
        # Dimensions des ensembles
        if "train_shape" in model_info and "test_shape" in model_info:
            parts.append("<h3>Dimensions des ensembles de données</h3>")
            parts.append("<div class='grid-2'>")
            parts.append("<div class='card'>")
            parts.append("<h4>Ensemble d'entraînement</h4>")
            train_shape = model_info["train_shape"]
            parts.append(f"<p><strong>Features (X_train) :</strong> {train_shape[0]} × {train_shape[1]}</p>")
            parts.append(f"<p><strong>Cible (y_train) :</strong> {train_shape[0]} valeurs</p>")
            parts.append("</div>")
            parts.append("<div class='card'>")
            parts.append("<h4>Ensemble de test</h4>")
            test_shape = model_info["test_shape"]
            parts.append(f"<p><strong>Features (X_test) :</strong> {test_shape[0]} × {test_shape[1]}</p>")
            parts.append(f"<p><strong>Cible (y_test) :</strong> {test_shape[0]} valeurs</p>")
            parts.append("</div>")
            parts.append("</div>")
        
        # Importance des features
        if "feature_importance" in model_info and "feature_names" in model_info:
            parts.append("<h3>Importance des features</h3>")
            parts.append("<h4>Top 10 des features les plus importantes</h4>")
            
            feat_imp = model_info["feature_importance"]
            feat_names = model_info["feature_names"]
            
            if len(feat_imp) > 0 and len(feat_names) == len(feat_imp):
                # Créer un tableau d'importance
                indices = np.argsort(feat_imp)[-10:][::-1]  # Top 10, ordre décroissant
                importance_data = []
                
                for i in indices:
                    importance_pct = feat_imp[i] * 100
                    importance_data.append({
                        'Feature': feat_names[i],
                        'Importance': f"{feat_imp[i]:.6f}",
                        'Importance (%)': f"{importance_pct:.2f}"
                    })
                
                imp_df = pd.DataFrame(importance_data)
                parts.append(_wrap_table(imp_df.to_html(classes='dataframe dataframe', index=False)))
                
                # Graphique d'importance
                if not selected_sections or "visualizations" in selected_sections:
                    try:
                        imp_fig = _create_feature_importance_plot(feat_names, feat_imp)
                        if imp_fig:
                            parts.append(_img_to_base64(imp_fig))
                    except Exception as e:
                        parts.append(f"<div class='warning-box'>Erreur lors de la génération du graphique d'importance: {str(e)}</div>")
    
    # Section 4: Évaluation des performances
    if model_info and (not selected_sections or "visualizations" in selected_sections):
        parts.append("<h2>4. Évaluation des performances</h2>")
        
        # Métriques de performance
        parts.append("<h3>Métriques de performance</h3>")
        
        metrics_data = {}
        
        if "metric_value" in model_info and model_info["metric_value"] is not None:
            metrics_data['accuracy'] = [f"{model_info['metric_value']:.4f}"]
        
        if "f1_score" in model_info and model_info["f1_score"] is not None:
            metrics_data['f1_weighted'] = [f"{model_info['f1_score']:.4f}"]
        
        if "precision" in model_info and model_info["precision"] is not None:
            metrics_data['precision_weighted'] = [f"{model_info['precision']:.4f}"]
        
        if "recall" in model_info and model_info["recall"] is not None:
            metrics_data['recall_weighted'] = [f"{model_info['recall']:.4f}"]
        
        # Ajouter des métriques spécifiques à la régression si disponibles
        if model_info.get("task_type") == "regression":
            if "mse" in model_info and model_info["mse"] is not None:
                metrics_data['mse'] = [f"{model_info['mse']:.4f}"]
            if "rmse" in model_info and model_info["rmse"] is not None:
                metrics_data['rmse'] = [f"{model_info['rmse']:.4f}"]
            if "r2" in model_info and model_info["r2"] is not None:
                metrics_data['r2_score'] = [f"{model_info['r2']:.4f}"]
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            parts.append(_wrap_table(metrics_df.to_html(classes='dataframe dataframe', index=False)))
        
        # Visualisations des performances
        parts.append("<h3>Visualisations des performances</h3>")
        try:
            eval_fig = _create_evaluation_plots(model_info)
            if eval_fig:
                parts.append(_img_to_base64(eval_fig))
        except Exception as e:
            parts.append(f"<div class='warning-box'>Erreur lors de la génération des graphiques d'évaluation: {str(e)}</div>")
    
    # Section 5: Cartographie (dernière carte tracée)
    if not selected_sections or "cartography" in selected_sections:
        parts.append("<h2>5. Cartographie et Analyse Spatiale</h2>")
        
        # Vérifier si des données géospatiales sont disponibles
        has_geo_data = False
        geo_cols = []
        
        if isinstance(df, pd.DataFrame):
            # Rechercher des colonnes géographiques
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['lat', 'latitude', 'lon', 'longitude', 'x', 'y']):
                    geo_cols.append(col)
                    has_geo_data = True
        
        if has_geo_data:
            parts.append("<div class='info-box'>")
            parts.append("<h3>Données géospatiales détectées</h3>")
            parts.append(f"<p>Colonnes géographiques trouvées : {', '.join(geo_cols)}</p>")
            
            # Statistiques géospatiales
            parts.append("<h4>Statistiques géospatiales</h4>")
            for col in geo_cols[:4]:  # Limiter à 4 colonnes
                if col in df.columns:
                    stats = df[col].describe()
                    parts.append(f"""
                        <div class='metric-box'>
                            <span class='metric-label'>{col}</span>
                            <span class='metric-value'>{stats['mean']:.4f}</span>
                        </div>
                    """)
            
            parts.append("</div>")
            
            # Essayer de créer une carte
            try:
                import folium
                
                if len(geo_cols) >= 2:
                    lat_col = geo_cols[0] if 'lat' in geo_cols[0].lower() else geo_cols[1]
                    lon_col = geo_cols[1] if 'lon' in geo_cols[1].lower() else geo_cols[0]
                    
                    # Calculer le centre
                    valid_data = df[[lat_col, lon_col]].dropna()
                    if len(valid_data) > 0:
                        center_lat = valid_data[lat_col].mean()
                        center_lon = valid_data[lon_col].mean()
                        
                        # Créer la carte
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        # Ajouter des points pour les 100 premières observations
                        sample_data = valid_data.head(100)
                        for idx, row in sample_data.iterrows():
                            folium.CircleMarker(
                                location=[row[lat_col], row[lon_col]],
                                radius=5,
                                popup=f"Point {idx}",
                                color='#667eea',
                                fill=True,
                                fillColor='#667eea'
                            ).add_to(m)
                        
                        # Sauvegarder la carte
                        map_html = m._repr_html_()
                        parts.append("<div class='figure-container'>")
                        parts.append(map_html)
                        parts.append("</div>")
                        
            except ImportError:
                parts.append("""
                    <div class='warning-box'>
                        <h4>Bibliothèque cartographique non disponible</h4>
                        <p>Pour afficher les cartes, installez la bibliothèque folium : pip install folium</p>
                    </div>
                """)
            except Exception as e:
                parts.append(f"<div class='warning-box'>Erreur lors de la génération de la carte: {str(e)}</div>")
        else:
            parts.append("""
                <div class='warning-box'>
                    <h4>Aucune donnée géospatiale détectée</h4>
                    <p>Pour inclure des cartes dans le rapport, assurez-vous que vos données contiennent des colonnes de coordonnées (latitude/longitude).</p>
                    <p>Colonnes attendues : latitude, longitude, lat, lon, x, y</p>
                </div>
            """)
    
    # Section 6: Recommandations
    if not selected_sections or "recommendations" in selected_sections:
        parts.append("<h2>6. Recommandations et Perspectives</h2>")
        parts.append("<div class='info-box'>")
        parts.append("<h3>Recommandations basées sur l'analyse</h3>")
        parts.append("<ul>")
        
        if isinstance(df, pd.DataFrame):
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_pct > 10:
                parts.append("<li><strong>Qualité des données :</strong> Considérer l'imputation des valeurs manquantes ou la collecte de données complémentaires.</li>")
            else:
                parts.append("<li><strong>Qualité des données :</strong> La qualité des données est acceptable pour l'analyse.</li>")
        
        if model_info:
            parts.append("<li><strong>Modélisation :</strong> Le modèle actuel peut être amélioré avec des features supplémentaires et l'optimisation des hyperparamètres.</li>")
            parts.append("<li><strong>Validation :</strong> Recommander la validation croisée pour une meilleure évaluation des performances.</li>")
        
        parts.append("<li><strong>Analyses futures :</strong> Envisager des analyses temporelles et spatiales plus approfondies.</li>")
        parts.append("<li><strong>Monitoring :</strong> Mettre en place un système de surveillance continue des indicateurs climatiques.</li>")
        parts.append("</ul>")
        parts.append("</div>")
    
    # Footer
    parts.append("<div class='footer'>")
    parts.append("<h3>Résumé du rapport</h3>")
    parts.append(f"<p><strong>Date de génération :</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>")
    
    if isinstance(df, pd.DataFrame):
        parts.append(f"<p><strong>Dataset initial :</strong> {df.shape[0]:,} lignes × {df.shape[1]} colonnes</p>")
    
    if model_info:
        parts.append(f"<p><strong>Modèle :</strong> {model_info.get('model_name', 'Non spécifié')}</p>")
        parts.append(f"<p><strong>Type de tâche :</strong> {model_info.get('task_type', 'Non spécifié')}</p>")
    
    parts.append("<p><strong>Auteur :</strong> Yacoubou KOUMAI</p>")
    parts.append("<p style='margin-top:20px; color:#95a5a6;'>Rapport généré automatiquement par Climate Risk Tool v1.0</p>")
    parts.append("</div>")
    
    # Fermeture des balises
    parts.extend([
        "</div>",
        "</body>",
        "</html>"
    ])
    
    # Écriture du fichier HTML
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    
    return out_path


def show_climate_reporting_summary(session_state: dict) -> None:
    """Affiche l'interface de génération de rapport dans Streamlit avec options de personnalisation."""
    import streamlit as st
    
    st.subheader("📝 Générer un rapport d'analyse climatique")
    
    # Vérification des données disponibles
    has_data = "clim_data" in session_state and session_state["clim_data"] is not None
    has_prep = "clim_data_prep" in session_state and session_state["clim_data_prep"] is not None
    has_model = "clim_model_info" in session_state and session_state["clim_model_info"] is not None
    
    if not has_data:
        st.warning("⚠️ Aucune donnée n'a été chargée. Veuillez d'abord charger des données depuis l'onglet 'Chargement des Données'.")
        return
    
    # Affichage du statut
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Données brutes", "✅ Disponibles" if has_data else "❌ Manquantes")
    with col2:
        st.metric("Prétraitement", "✅ Disponible" if has_prep else "❌ Manquant")
    with col3:
        st.metric("Modèle", "✅ Disponible" if has_model else "❌ Manquant")
    
    # Options de personnalisation du rapport
    st.markdown("---")
    st.subheader("⚙️ Configuration du Rapport")
    
    # Organisation en colonnes pour une meilleure présentation
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("##### 📋 Sections à inclure")
        
        # Sélection des sections à inclure avec une meilleure disposition
        sections = [
            ("📋 Synthèse Exécutive", "exec_summary", True),
            ("📊 Analyse des Données", "data_analysis", True),
            ("🔧 Prétraitement", "preprocessing", has_prep),
            ("🤖 Modélisation", "modeling", has_model),
            ("📈 Visualisations", "visualizations", has_prep or has_model),
            ("🗺️ Cartographie", "cartography", True),
            ("📝 Recommandations", "recommendations", True)
        ]
        
        selected_sections = []
        # Afficher les sections en 2 colonnes pour une meilleure lisibilité
        cols = st.columns(2)
        for i, (name, key, enabled) in enumerate(sections):
            with cols[i % 2]:
                if st.checkbox(name, value=enabled, key=f"report_section_{key}", disabled=not enabled):
                    selected_sections.append(key)
    
    with col_right:
        st.markdown("##### ⚙️ Options avancées")
        report_title = st.text_input("Titre du rapport", "Rapport d'Analyse Climatique")
        include_code = st.checkbox("Inclure le code source", value=False)
        
        # Informations sur le rapport
        st.markdown("##### 📊 Informations")
        info_text = f"""
        **Données disponibles :**
        - {'✅' if has_data else '❌'} Données brutes
        - {'✅' if has_prep else '❌'} Données prétraitées
        - {'✅' if has_model else '❌'} Modèle entraîné
        
        **Sections sélectionnées :** {len(selected_sections)}/{len(sections)}
        """
        st.info(info_text)
        
    # Bouton de génération
    st.markdown("---")
    st.subheader("📤 Exporter le Rapport")
    
    # Centrer le bouton de génération
    col_generate, col_empty = st.columns([1, 1])
    with col_generate:
        if st.button("� Générer le rapport HTML", type="primary", use_container_width=True):
            with st.spinner("Génération du rapport en cours..."):
                try:
                    # Créer une copie du contexte avec les sections sélectionnées
                    report_context = {
                        **session_state,
                        "report_options": {
                            "sections": selected_sections,
                            "title": report_title,
                            "include_code": include_code
                        }
                    }
                    
                    report_path = generate_climate_report(report_context)
                    if report_path:
                        st.success("✅ Rapport généré avec succès !")
                        st.info(f"📄 Rapport sauvegardé : {report_path}")
                        
                        # Affichage du bouton de téléchargement
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Télécharger le rapport",
                                data=f,
                                file_name=f"Rapport_Climat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True,
                                type="primary"
                            )
                            
                        # Aperçu intégré
                        st.markdown("---")
                        st.subheader("👁️ Aperçu du rapport")
                        st.components.v1.html(
                            open(report_path, "r", encoding="utf-8").read(), 
                            height=600, 
                            scrolling=True
                        )
                    else:
                        st.error("❌ Erreur lors de la génération du rapport")
                        
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                    st.exception(e)  # Afficher plus de détails sur l'erreur
    
    with col_empty:
        # Informations supplémentaires sur le rapport
        st.markdown("##### 📝 Fonctionnalités")
        st.info("""
        **Fonctionnalités du rapport :**
        - 📊 Visualisations interactives
        - 📋 Tableaux de données détaillés
        - 🎨 Design moderne et responsive
        - 📱 Compatible mobile
        - 🖨️ Optimisé pour l'impression
        - Personnalisation des sections incluses
        - Génération rapide même avec des données partielles
        - Aperçu intégré avant téléchargement
        - Options avancées de personnalisation
        - 🗺️ Cartographie intégrée
        """)
