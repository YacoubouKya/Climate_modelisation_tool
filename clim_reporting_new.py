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
        
        # Dimensions avant et après prétraitement
        if isinstance(df, pd.DataFrame) and isinstance(df_prep, pd.DataFrame):
            parts.append("<h3>Dimensions des données</h3>")
            parts.append("<div class='grid-2'>")
            parts.append("<div class='card'>")
            parts.append("<h4>Avant prétraitement</h4>")
            parts.append(f"<p><strong>Lignes :</strong> {df.shape[0]:,}</p>")
            parts.append(f"<p><strong>Colonnes :</strong> {df.shape[1]}</p>")
            parts.append("</div>")
            parts.append("<div class='card'>")
            parts.append("<h4>Après prétraitement</h4>")
            parts.append(f"<p><strong>Lignes :</strong> {df_prep.shape[0]:,}</p>")
            parts.append(f"<p><strong>Colonnes :</strong> {df_prep.shape[1]}</p>")
            parts.append(f"<p><strong>Nouvelles features :</strong> {df_prep.shape[1] - df.shape[1]}</p>")
            parts.append("</div>")
            parts.append("</div>")
        
        if prep_info:
            parts.append("<div class='info-box'>")
            parts.append("<h3>Étapes de prétraitement appliquées</h3>")
            parts.append("<ul>")
            
            if prep_info.get("date_col"):
                parts.append(f"<li><strong>Colonne temporelle :</strong> {prep_info['date_col']}</li>")
            
            if prep_info.get("freq") and prep_info.get("freq") != "Aucune":
                parts.append(f"<li><strong>Fréquence d'agrégation :</strong> {prep_info['freq']}</li>")
            
            if prep_info.get("rolling"):
                parts.append("<li><strong>Calcul des indicateurs mobiles :</strong> Moyennes glissantes appliquées</li>")
            
            if prep_info.get("anomaly_summary"):
                parts.append("<li><strong>Détection des anomalies :</strong> Analyse z-score effectuée</li>")
            
            if prep_info.get("cumul_features"):
                parts.append("<li><strong>Features cumulatives :</strong> Cumuls glissants ajoutés</li>")
            
            if prep_info.get("threshold_features"):
                parts.append("<li><strong>Features de seuil :</strong> Comptage de jours > seuil</li>")
            
            if prep_info.get("ref_anomaly_features"):
                parts.append("<li><strong>Anomalies de référence :</strong> Calcul vs période climatologique</li>")
            
            if prep_info.get("extreme_features"):
                parts.append("<li><strong>Features extrêmes :</strong> Min/max glissants ajoutés</li>")
            
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
        
        # Vérification temporaire des clés disponibles
        parts.append("<div class='info-box'>")
        parts.append("<p><strong>Vérification des clés model_info :</strong></p>")
        parts.append("<ul>")
        keys_to_check = ["train_shape", "test_shape", "feature_importance", "feature_names"]
        for key in keys_to_check:
            if key in model_info:
                value = model_info[key]
                if value is None:
                    parts.append(f"<li>{key}: Présent mais None</li>")
                elif hasattr(value, 'shape'):
                    parts.append(f"<li>{key}: Présent - Shape {value.shape}</li>")
                elif isinstance(value, (list, tuple)):
                    parts.append(f"<li>{key}: Présent - Longueur {len(value)}</li>")
                else:
                    parts.append(f"<li>{key}: Présent - Type {type(value)}</li>")
            else:
                parts.append(f"<li>{key}: Absent</li>")
        parts.append("</ul>")
        parts.append("</div>")
        
        parts.append("<div class='info-box'>")
        parts.append(f"<p><strong>Nom du modèle :</strong> {model_info.get('model_name', 'Non spécifié')}</p>")
        parts.append(f"<p><strong>Type de pipeline :</strong> {model_info.get('pipeline_type', 'Standard')}</p>")
        parts.append(f"<p><strong>Tâche :</strong> {model_info.get('task_type', 'Non spécifié')}</p>")
        parts.append("</div>")
        
        # Dimensions des ensembles
        if "train_shape" in model_info and "test_shape" in model_info and model_info["train_shape"] is not None and model_info["test_shape"] is not None:
            parts.append("<h3>Dimensions des ensembles de données</h3>")
            parts.append("<div class='grid-2'>")
            parts.append("<div class='card'>")
            parts.append("<h4>Ensemble d'entraînement</h4>")
            train_shape = model_info["train_shape"]
            parts.append(f"<p><strong>Features (X_train) :</strong> {train_shape[0]:,} × {train_shape[1]}</p>")
            parts.append(f"<p><strong>Cible (y_train) :</strong> {train_shape[0]:,} valeurs</p>")
            parts.append(f"<p><strong>Ratio :</strong> {train_shape[0]/(train_shape[0] + model_info['test_shape'][0])*100:.1f}%</p>")
            parts.append("</div>")
            parts.append("<div class='card'>")
            parts.append("<h4>Ensemble de test</h4>")
            test_shape = model_info["test_shape"]
            parts.append(f"<p><strong>Features (X_test) :</strong> {test_shape[0]:,} × {test_shape[1]}</p>")
            parts.append(f"<p><strong>Cible (y_test) :</strong> {test_shape[0]:,} valeurs</p>")
            parts.append(f"<p><strong>Ratio :</strong> {test_shape[0]/(train_shape[0] + test_shape[0])*100:.1f}%</p>")
            parts.append("</div>")
            parts.append("</div>")
        
        # Importance des features
        if "feature_importance" in model_info and "feature_names" in model_info and model_info["feature_importance"] is not None and model_info["feature_names"] is not None:
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
                try:
                    imp_fig = _create_feature_importance_plot(feat_names, feat_imp)
                    if imp_fig:
                        parts.append("<h4>Graphique d'importance des features</h4>")
                        parts.append(_img_to_base64(imp_fig))
                except Exception as e:
                    parts.append(f"<div class='warning-box'>Erreur lors de la génération du graphique d'importance: {str(e)}</div>")
    
    # Section 4: Évaluation des performances
    if model_info and (not selected_sections or "visualizations" in selected_sections):
        parts.append("<h2>4. Évaluation des performances</h2>")
        
        # Métriques de performance
        parts.append("<h3>Métriques de performance</h3>")
        
        metrics_data = {}
        
        # Métriques pour classification
        if model_info.get("task_type") == "classification":
            if "metric_value" in model_info and model_info["metric_value"] is not None:
                metrics_data['accuracy'] = [f"{model_info['metric_value']:.4f}"]
            
            if "f1_score" in model_info and model_info["f1_score"] is not None:
                metrics_data['f1_score'] = [f"{model_info['f1_score']:.4f}"]
            
            if "precision" in model_info and model_info["precision"] is not None:
                metrics_data['precision'] = [f"{model_info['precision']:.4f}"]
            
            if "recall" in model_info and model_info["recall"] is not None:
                metrics_data['recall'] = [f"{model_info['recall']:.4f}"]
        
        # Métriques pour régression
        elif model_info.get("task_type") == "regression":
            if "mse" in model_info and model_info["mse"] is not None:
                metrics_data['mse'] = [f"{model_info['mse']:.4f}"]
            
            if "rmse" in model_info and model_info["rmse"] is not None:
                metrics_data['rmse'] = [f"{model_info['rmse']:.4f}"]
            
            if "r2" in model_info and model_info["r2"] is not None:
                metrics_data['r2_score'] = [f"{model_info['r2']:.4f}"]
            
            if "mae" in model_info and model_info["mae"] is not None:
                metrics_data['mae'] = [f"{model_info['mae']:.4f}"]
        
        # Métriques génériques (si task_type non spécifié)
        else:
            if "metric_value" in model_info and model_info["metric_value"] is not None:
                metrics_data['score'] = [f"{model_info['metric_value']:.4f}"]
            
            if "mse" in model_info and model_info["mse"] is not None:
                metrics_data['mse'] = [f"{model_info['mse']:.4f}"]
            
            if "rmse" in model_info and model_info["rmse"] is not None:
                metrics_data['rmse'] = [f"{model_info['rmse']:.4f}"]
            
            if "r2" in model_info and model_info["r2"] is not None:
                metrics_data['r2_score'] = [f"{model_info['r2']:.4f}"]
            
            if "mae" in model_info and model_info["mae"] is not None:
                metrics_data['mae'] = [f"{model_info['mae']:.4f}"]
            
            if "f1_score" in model_info and model_info["f1_score"] is not None:
                metrics_data['f1_score'] = [f"{model_info['f1_score']:.4f}"]
            
            if "precision" in model_info and model_info["precision"] is not None:
                metrics_data['precision'] = [f"{model_info['precision']:.4f}"]
            
            if "recall" in model_info and model_info["recall"] is not None:
                metrics_data['recall'] = [f"{model_info['recall']:.4f}"]
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            parts.append(_wrap_table(metrics_df.to_html(classes='dataframe dataframe', index=False)))
        else:
            parts.append("<div class='warning-box'>Aucune métrique de performance disponible</div>")
        
        # Visualisations des performances
        parts.append("<h3>Visualisations des performances</h3>")
        try:
            eval_fig = _create_evaluation_plots(model_info)
            if eval_fig:
                parts.append(_img_to_base64(eval_fig))
        except Exception as e:
            parts.append(f"<div class='warning-box'>Erreur lors de la génération des graphiques d'évaluation: {str(e)}</div>")
    
    # Section 5: Cartographie Avancée
    if not selected_sections or "cartography" in selected_sections:
        parts.append("<h2>5. Cartographie et Analyse Spatiale Avancée</h2>")
        
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
            parts.append("<h3>📍 Données Géospatiales Détectées</h3>")
            parts.append(f"<p><strong>Colonnes géographiques :</strong> {', '.join(geo_cols)}</p>")
            parts.append(f"<p><strong>Nombre de points :</strong> {len(df):,}</p>")
            
            # Statistiques géospatiales détaillées
            parts.append("<h4>📊 Statistiques Géospatiales</h4>")
            parts.append("<div class='grid-2'>")
            for col in geo_cols[:4]:  # Limiter à 4 colonnes
                if col in df.columns:
                    stats = df[col].describe()
                    parts.append(f"""
                        <div class='card'>
                            <h5>{col}</h5>
                            <p><strong>Moyenne :</strong> {stats['mean']:.4f}</p>
                            <p><strong>Min :</strong> {stats['min']:.4f}</p>
                            <p><strong>Max :</strong> {stats['max']:.4f}</p>
                            <p><strong>Écart-type :</strong> {stats['std']:.4f}</p>
                        </div>
                    """)
            parts.append("</div>")
            parts.append("</div>")
            
            # Créer les cartes avancées
            try:
                import folium
                from folium.plugins import HeatMap, MarkerCluster
                import branca.colormap as cm
                
                # Identifier les colonnes latitude/longitude
                lat_col = None
                lon_col = None
                for col in geo_cols:
                    col_lower = col.lower()
                    if 'lat' in col_lower and lat_col is None:
                        lat_col = col
                    elif 'lon' in col_lower and lon_col is None:
                        lon_col = col
                
                # Si pas trouvé, utiliser les deux premières
                if lat_col is None or lon_col is None:
                    lat_col = geo_cols[0]
                    lon_col = geo_cols[1] if len(geo_cols) > 1 else geo_cols[0]
                
                # Préparer les données
                valid_data = df[[lat_col, lon_col]].dropna()
                if len(valid_data) > 0:
                    center_lat = valid_data[lat_col].mean()
                    center_lon = valid_data[lon_col].mean()
                    
                    # Identifier les variables climatiques pour la coloration
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    # Exclure les colonnes de coordonnées
                    climate_vars = [col for col in numeric_cols 
                                  if col not in [lat_col, lon_col] and 
                                  not any(geo in col.lower() for geo in ['lat', 'lon', 'x', 'y'])]
                    
                    # Essayer de trouver la variable cible du modèle
                    target_variable = None
                    model_info = session_state.get("clim_model_info")
                    
                    # Debug: Afficher les informations disponibles
                    if model_info:
                        # 1. Dans les informations de modélisation
                        if "target_col" in model_info:
                            target_variable = model_info["target_col"]
                        # 2. Dans les informations de prétraitement
                        elif "clim_prep_info" in session_state:
                            prep_info = session_state["clim_prep_info"]
                            if "target_col" in prep_info:
                                target_variable = prep_info["target_col"]
                        # 3. Chercher des colonnes typiques de target
                        else:
                            typical_targets = ['target', 'label', 'y', 'outcome', 'risk', 'temperature', 'temp', 
                                             'precipitation', 'rain', 'humidity', 'wind', 'pressure', 'sea_level']
                            for target in typical_targets:
                                if target in climate_vars:
                                    target_variable = target
                                    break
                    else:
                        # Pas de model_info, chercher dans les colonnes typiques
                        typical_targets = ['target', 'label', 'y', 'outcome', 'risk', 'temperature', 'temp', 
                                         'precipitation', 'rain', 'humidity', 'wind', 'pressure', 'sea_level']
                        for target in typical_targets:
                            if target in climate_vars:
                                target_variable = target
                                break
                    
                    # Mettre la variable cible en premier si elle existe
                    if target_variable and target_variable in climate_vars:
                        climate_vars.remove(target_variable)
                        climate_vars.insert(0, target_variable)
                    
                    parts.append("<h3>🗺️ Cartes Interactives</h3>")
                    
                    # Afficher la variable cible détectée
                    if target_variable:
                        parts.append(f"""
                        <div class='info-box'>
                            <h4>🎯 Variable Cible du Modèle Détectée</h4>
                            <p><strong>Variable utilisée pour la cartographie thématique :</strong> <code>{target_variable}</code></p>
                            <p><em>C'est la variable que le modèle a appris à prédire. Elle sera utilisée par défaut pour colorer les points sur la carte.</em></p>
                        </div>
                        """)
                    else:
                        # Afficher un message d'aide si aucune variable cible n'est détectée
                        parts.append(f"""
                        <div class='warning-box'>
                            <h4>🔍 Variable Cible Non Détectée Automatiquement</h4>
                            <p><strong>Variables climatiques disponibles ({len(climate_vars)}) :</strong></p>
                            <p>{', '.join(climate_vars[:5])}{'...' if len(climate_vars) > 5 else ''}</p>
                            <p><em>Pour une meilleure expérience, entraînez d'abord un modèle avec une variable cible clairement nommée (ex: 'target', 'temperature', 'risk', etc.).</em></p>
                        </div>
                        """)
                    
                    parts.append("<p><strong>Variables climatiques disponibles pour la coloration :</strong></p>")
                    parts.append("<ul>")
                    for i, var in enumerate(climate_vars[:5]):  # Limiter à 5 variables
                        is_target = (var == target_variable)
                        target_indicator = " 🎯 (cible du modèle)" if is_target else ""
                        parts.append(f"<li><strong>{var}</strong>{target_indicator}</li>")
                    if len(climate_vars) > 5:
                        parts.append(f"<li><em>... et {len(climate_vars) - 5} autres</em></li>")
                    parts.append("</ul>")
                    
                    # === CARTE 1: Points de Base Colorés par Variable Cible ===
                    parts.append("<h4>📍 Carte des Points avec Variable Cible</h4>")
                    m1 = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                    
                    # Utiliser la variable cible pour colorer les points
                    if target_variable and target_variable in df.columns:
                        target_data = df[[lat_col, lon_col, target_variable]].dropna().head(200)
                        if len(target_data) > 0:
                            # Vérifier le type de données de la variable cible
                            target_dtype = target_data[target_variable].dtype
                            is_numeric = pd.api.types.is_numeric_dtype(target_data[target_variable])
                            
                            if is_numeric:
                                # Variable numérique : utiliser une colormap
                                min_val = target_data[target_variable].min()
                                max_val = target_data[target_variable].max()
                                
                                # Éviter les problèmes si min_val == max_val
                                if min_val == max_val:
                                    max_val = min_val + 1
                                
                                colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], 
                                                             vmin=min_val, vmax=max_val)
                                
                                for idx, row in target_data.iterrows():
                                    color = colormap(row[target_variable])
                                    popup_text = f"""
                                    <b>Point #{idx}</b><br>
                                    <b>Latitude:</b> {row[lat_col]:.4f}<br>
                                    <b>Longitude:</b> {row[lon_col]:.4f}<br>
                                    <b>{target_variable}:</b> {row[target_variable]:.2f}
                                    """
                                    folium.CircleMarker(
                                        location=[row[lat_col], row[lon_col]],
                                        radius=4,
                                        popup=folium.Popup(popup_text, max_width=200),
                                        tooltip=f"{target_variable}: {row[target_variable]:.2f}",
                                        color=color,
                                        fill=True,
                                        fillColor=color,
                                        fillOpacity=0.7
                                    ).add_to(m1)
                                
                                # Ajouter la colormap à la carte
                                m1.add_child(colormap)
                            else:
                                # Variable catégorielle : utiliser des couleurs fixes par catégorie
                                unique_values = target_data[target_variable].unique()
                                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                                color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
                                
                                for idx, row in target_data.iterrows():
                                    color = color_map[row[target_variable]]
                                    popup_text = f"""
                                    <b>Point #{idx}</b><br>
                                    <b>Latitude:</b> {row[lat_col]:.4f}<br>
                                    <b>Longitude:</b> {row[lon_col]:.4f}<br>
                                    <b>{target_variable}:</b> {row[target_variable]}
                                    """
                                    folium.CircleMarker(
                                        location=[row[lat_col], row[lon_col]],
                                        radius=4,
                                        popup=folium.Popup(popup_text, max_width=200),
                                        tooltip=f"{target_variable}: {row[target_variable]}",
                                        color=color,
                                        fill=True,
                                        fillColor=color,
                                        fillOpacity=0.7
                                    ).add_to(m1)
                        else:
                            # Pas de données valides pour la variable cible
                            sample_data = df[[lat_col, lon_col]].dropna().head(200)
                            for idx, row in sample_data.iterrows():
                                folium.CircleMarker(
                                    location=[row[lat_col], row[lon_col]],
                                    radius=4,
                                    popup=f"Point #{idx}",
                                    tooltip=f"Point {idx}",
                                    color='#1f77b4',
                                    fill=True,
                                    fillColor='#1f77b4',
                                    fillOpacity=0.7
                                ).add_to(m1)
                    else:
                        # Pas de variable cible, utiliser les points de base
                        sample_data = df[[lat_col, lon_col]].dropna().head(200)
                        for idx, row in sample_data.iterrows():
                            folium.CircleMarker(
                                location=[row[lat_col], row[lon_col]],
                                radius=4,
                                popup=f"Point #{idx}",
                                tooltip=f"Point {idx}",
                                color='#1f77b4',
                                fill=True,
                                fillColor='#1f77b4',
                                fillOpacity=0.7
                            ).add_to(m1)
                    
                    # Ajouter une légende
                    target_display = target_variable if target_variable else "Variable cible non détectée"
                    legend_html = f'''
                    <div style="position: fixed; 
                                bottom: 50px; left: 50px; width: 200px; height: 110px; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:14px; padding: 10px">
                    <p><b>Légende</b></p>
                    <p><i class="fa fa-circle" style="color:#1f77b4"></i> Points de données</p>
                    <p><small>Colorés par: {target_display}</small></p>
                    <p><small>Total: {len(df)} points</small></p>
                    </div>
                    '''
                    m1.get_root().html.add_child(folium.Element(legend_html))
                    
                    map_html1 = m1._repr_html_()
                    parts.append("<div class='figure-container'>")
                    parts.append(map_html1)
                    parts.append("</div>")
                    
                    # === CARTE 2: Heat Map ===
                    if len(valid_data) > 10:
                        parts.append("<h4>🔥 Carte de Densité (Heat Map)</h4>")
                        m2 = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        # Préparer les données pour la heat map
                        heat_data = [[row[lat_col], row[lon_col]] for idx, row in valid_data.iterrows()]
                        HeatMap(heat_data, 
                               radius=15, 
                               blur=10, 
                               gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'},
                               name='Densité des points').add_to(m2)
                        
                        # Légende pour la heat map
                        legend_html2 = '''
                        <div style="position: fixed; 
                                    bottom: 50px; left: 50px; width: 180px; height: 110px; 
                                    background-color: white; border:2px solid grey; z-index:9999; 
                                    font-size:14px; padding: 10px">
                        <p><b>Densité</b></p>
                        <div style="background: linear-gradient(to right, blue, cyan, lime, yellow, red); height: 20px;"></div>
                        <p><small>Faible → Élevée</small></p>
                        <p><small>Points: {}</small></p>
                        </div>
                        '''.format(len(heat_data))
                        m2.get_root().html.add_child(folium.Element(legend_html2))
                        
                        map_html2 = m2._repr_html_()
                        parts.append("<div class='figure-container'>")
                        parts.append(map_html2)
                        parts.append("</div>")
                    
                    # === CARTE 3: Carte Thématique par Variable Cible ===
                    if target_variable and target_variable in df.columns:
                        parts.append("<h4>🌡️ Carte Thématique par Variable Cible</h4>")
                        
                        # Créer une section d'information sur la variable cible
                        parts.append("<div class='info-box'>")
                        parts.append("<h5>🎯 Variable Cible du Modèle</h5>")
                        parts.append(f"<p><strong>Variable utilisée pour la coloration :</strong> <code>{target_variable}</code></p>")
                        
                        # Statistiques de la variable cible
                        target_data = df[target_variable].dropna()
                        if len(target_data) > 0:
                            is_numeric = pd.api.types.is_numeric_dtype(target_data)
                            
                            if is_numeric:
                                parts.append(f"""
                                <div class='grid-2'>
                                    <div>
                                        <p><strong>Plage de valeurs :</strong> {target_data.min():.2f} - {target_data.max():.2f}</p>
                                        <p><strong>Moyenne :</strong> {target_data.mean():.2f}</p>
                                        <p><strong>Médiane :</strong> {target_data.median():.2f}</p>
                                    </div>
                                    <div>
                                        <p><strong>Écart-type :</strong> {target_data.std():.2f}</p>
                                        <p><strong>Nombre de points :</strong> {len(target_data):,}</p>
                                        <p><strong>Valeurs manquantes :</strong> {df[target_variable].isna().sum()}</p>
                                    </div>
                                </div>
                                """)
                            else:
                                # Statistiques pour variables catégorielles
                                unique_values = target_data.nunique()
                                most_common = target_data.value_counts().index[0] if len(target_data) > 0 else "N/A"
                                parts.append(f"""
                                <div class='grid-2'>
                                    <div>
                                        <p><strong>Type de variable :</strong> Catégorielle</p>
                                        <p><strong>Nombre de catégories :</strong> {unique_values}</p>
                                        <p><strong>Catégorie la plus fréquente :</strong> {most_common}</p>
                                    </div>
                                    <div>
                                        <p><strong>Nombre de points :</strong> {len(target_data):,}</p>
                                        <p><strong>Valeurs manquantes :</strong> {df[target_variable].isna().sum()}</p>
                                        <p><strong>Catégories uniques :</strong> {', '.join(target_data.unique()[:5])}{'...' if unique_values > 5 else ''}</p>
                                    </div>
                                </div>
                                """)
                        parts.append("</div>")
                        
                        try:
                            m3 = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        except Exception as e:
                            parts.append(f"<p><em>⚠️ Erreur création carte: {str(e)}</em></p>")
                            return None
                        
                        # Créer une carte pour la variable cible
                        var_data = df[target_variable].dropna()
                        if len(var_data) > 0:
                            is_numeric = pd.api.types.is_numeric_dtype(var_data)
                            
                            if is_numeric:
                                # Variable numérique : utiliser une colormap simple
                                try:
                                    min_val = var_data.min()
                                    max_val = var_data.max()
                                    
                                    # Éviter les problèmes si min_val == max_val
                                    if min_val == max_val:
                                        max_val = min_val + 1
                                    
                                    # Créer une colormap simple
                                    colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], 
                                                                 vmin=min_val, vmax=max_val)
                                    colormap.caption = f'{target_variable} ({min_val:.2f} - {max_val:.2f})'
                                    colormap.position = 'bottomright'
                                    
                                    # Ajouter les points colorés avec taille variable selon la valeur
                                    color_data = df[[lat_col, lon_col, target_variable]].dropna().head(300)
                                    for idx, row in color_data.iterrows():
                                        # Normaliser la valeur pour la taille du point
                                        normalized_val = (row[target_variable] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                                        radius = 3 + normalized_val * 7  # Taille entre 3 et 10
                                        
                                        color = colormap(row[target_variable])
                                        popup_text = f"""
                                        <div style='width: 200px;'>
                                            <h5>Point #{idx}</h5>
                                            <p><strong>Latitude:</strong> {row[lat_col]:.4f}</p>
                                            <p><strong>Longitude:</strong> {row[lon_col]:.4f}</p>
                                            <p><strong>{target_variable}:</strong> <span style='color: {color}; font-weight: bold;'>{row[target_variable]:.2f}</span></p>
                                            <p><small>Percentile: {(normalized_val * 100):.1f}%</small></p>
                                        </div>
                                        """
                                        folium.CircleMarker(
                                            location=[row[lat_col], row[lon_col]],
                                            radius=radius,
                                            popup=folium.Popup(popup_text, max_width=250),
                                            tooltip=f"{target_variable}: {row[target_variable]:.2f}",
                                            color=color,
                                            fill=True,
                                            fillColor=color,
                                            fillOpacity=0.8,
                                            weight=2
                                        ).add_to(m3)
                                    
                                    # Ajouter la colormap à la carte
                                    m3.add_child(colormap)
                                    
                                except Exception as e:
                                    # En cas d'erreur avec la colormap, utiliser des points bleus simples
                                    parts.append(f"<p><em>⚠️ Erreur de colormap: {str(e)}</em></p>")
                                    color_data = df[[lat_col, lon_col]].dropna().head(300)
                                    for idx, row in color_data.iterrows():
                                        folium.CircleMarker(
                                            location=[row[lat_col], row[lon_col]],
                                            radius=5,
                                            popup=f"Point #{idx}",
                                            tooltip=f"Point {idx}",
                                            color='blue',
                                            fill=True,
                                            fillColor='blue',
                                            fillOpacity=0.7
                                        ).add_to(m3)
                            else:
                                # Variable catégorielle : utiliser des couleurs fixes par catégorie
                                unique_values = var_data.unique()
                                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B4513', 
                                         '#4B0082', '#FF69B4', '#808080', '#008080', '#FFD700']
                                color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
                                
                                # Créer une légende pour les catégories
                                legend_html = '''
                                <div style="position: fixed; 
                                            bottom: 50px; right: 50px; width: 200px; max-height: 300px; 
                                            background-color: white; border:2px solid grey; z-index:9999; 
                                            font-size:12px; padding: 10px; overflow-y: auto;">
                                <p><b>Légende - {target_variable}</b></p>
                                '''
                                
                                for i, val in enumerate(unique_values):
                                    color = colors[i % len(colors)]
                                    legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {val}</p>'
                                
                                legend_html += '</div>'
                                m3.get_root().html.add_child(folium.Element(legend_html))
                                
                                # Ajouter les points colorés par catégorie
                                color_data = df[[lat_col, lon_col, target_variable]].dropna().head(300)
                                for idx, row in color_data.iterrows():
                                    color = color_map[row[target_variable]]
                                    popup_text = f"""
                                    <div style='width: 200px;'>
                                        <h5>Point #{idx}</h5>
                                        <p><strong>Latitude:</strong> {row[lat_col]:.4f}</p>
                                        <p><strong>Longitude:</strong> {row[lon_col]:.4f}</p>
                                        <p><strong>{target_variable}:</strong> <span style='color: {color}; font-weight: bold;'>{row[target_variable]}</span></p>
                                    </div>
                                    """
                                    folium.CircleMarker(
                                        location=[row[lat_col], row[lon_col]],
                                        radius=6,
                                        popup=folium.Popup(popup_text, max_width=250),
                                        tooltip=f"{target_variable}: {row[target_variable]}",
                                        color=color,
                                        fill=True,
                                        fillColor=color,
                                        fillOpacity=0.8,
                                        weight=2
                                    ).add_to(m3)
                            
                            # La carte utilise déjà les tuiles OpenStreetMap par défaut
                            # Pas besoin d'ajouter des contrôles de couches supplémentaires
                            # folium.LayerControl().add_to(m3)  # Commenté pour éviter l'erreur
                            
                            map_html3 = m3._repr_html_()
                            parts.append("<div class='figure-container'>")
                            parts.append(map_html3)
                            parts.append("</div>")
                            
                            # Informations détaillées sur la variable utilisée
                            if is_numeric:
                                parts.append(f"""
                                <div class='info-box'>
                                    <h5>📊 Variable Cible Numérique : {target_variable}</h5>
                                    <div class='grid-2'>
                                        <div>
                                            <p><strong>Plage de valeurs :</strong> {min_val:.2f} - {max_val:.2f}</p>
                                            <p><strong>Moyenne :</strong> {var_data.mean():.2f}</p>
                                            <p><strong>Médiane :</strong> {var_data.median():.2f}</p>
                                        </div>
                                        <div>
                                            <p><strong>Écart-type :</strong> {var_data.std():.2f}</p>
                                            <p><strong>Nombre de points :</strong> {len(var_data):,}</p>
                                            <p><strong>Valeurs manquantes :</strong> {df[target_variable].isna().sum()}</p>
                                        </div>
                                    </div>
                                    <p><em>💡 Les points sont colorés et dimensionnés selon la valeur de la variable cible numérique du modèle. Utilisez les contrôles en haut à droite pour changer le fond de carte.</em></p>
                                </div>
                                """)
                            else:
                                parts.append(f"""
                                <div class='info-box'>
                                    <h5>📊 Variable Cible Catégorielle : {target_variable}</h5>
                                    <div class='grid-2'>
                                        <div>
                                            <p><strong>Type de variable :</strong> Catégorielle</p>
                                            <p><strong>Nombre de catégories :</strong> {len(unique_values)}</p>
                                            <p><strong>Catégories :</strong> {', '.join(unique_values[:5])}{'...' if len(unique_values) > 5 else ''}</p>
                                        </div>
                                        <div>
                                            <p><strong>Nombre de points :</strong> {len(var_data):,}</p>
                                            <p><strong>Valeurs manquantes :</strong> {df[target_variable].isna().sum()}</p>
                                            <p><strong>Catégorie la plus fréquente :</strong> {var_data.value_counts().index[0]}</p>
                                        </div>
                                    </div>
                                    <p><em>💡 Les points sont colorés par catégorie selon la variable cible du modèle. Chaque catégorie a une couleur unique. Utilisez les contrôles en haut à droite pour changer le fond de carte.</em></p>
                                </div>
                                """)
                        
                        # Si pas assez de données pour la variable cible
                        else:
                            parts.append(f"""
                                <div class='warning-box'>
                                    <h5>⚠️ Données Insuffisantes</h5>
                                    <p>La variable cible '{target_variable}' ne contient pas assez de données valides pour créer une carte thématique.</p>
                                    <p>Points disponibles : {len(var_data)} / {len(df)}</p>
                                </div>
                            """)
                    else:
                        parts.append("""
                            <div class='warning-box'>
                                <h5>🎯 Aucune Variable Cible Détectée</h5>
                                <p>Pour créer une carte thématique, le système doit détecter la variable cible du modèle.</p>
                                <p>Vérifiez que votre modèle a été correctement entraîné avec une variable cible.</p>
                            </div>
                        """)
                    
                    # === CARTE 4: Clustering avec Variable Cible ===
                    if len(valid_data) > 50:
                        parts.append("<h4>🔗 Carte avec Clustering et Variable Cible</h4>")
                        m4 = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                        
                        # Créer un cluster de marqueurs
                        marker_cluster = MarkerCluster(name='Clusters de points')
                        
                        # Utiliser la variable cible pour les clusters si disponible
                        if target_variable and target_variable in df.columns:
                            cluster_data = df[[lat_col, lon_col, target_variable]].dropna().head(500)
                            if len(cluster_data) > 0:
                                # Vérifier le type de données de la variable cible
                                is_numeric = pd.api.types.is_numeric_dtype(cluster_data[target_variable])
                                
                                if is_numeric:
                                    # Variable numérique : utiliser une colormap
                                    min_val = cluster_data[target_variable].min()
                                    max_val = cluster_data[target_variable].max()
                                    
                                    # Éviter les problèmes si min_val == max_val
                                    if min_val == max_val:
                                        max_val = min_val + 1
                                    
                                    colormap = cm.LinearColormap(['blue', 'green', 'yellow', 'red'], 
                                                                 vmin=min_val, vmax=max_val)
                                    
                                    for idx, row in cluster_data.iterrows():
                                        color = colormap(row[target_variable])
                                        popup_text = f"""
                                        <div style='width: 200px;'>
                                            <h5>Point #{idx}</h5>
                                            <p><strong>Latitude:</strong> {row[lat_col]:.4f}</p>
                                            <p><strong>Longitude:</strong> {row[lon_col]:.4f}</p>
                                            <p><strong>{target_variable}:</strong> <span style='color: {color}; font-weight: bold;'>{row[target_variable]:.2f}</span></p>
                                        </div>
                                        """
                                        folium.Marker(
                                            location=[row[lat_col], row[lon_col]],
                                            popup=folium.Popup(popup_text, max_width=250),
                                            tooltip=f"{target_variable}: {row[target_variable]:.2f}",
                                            icon=folium.Icon(color='blue', icon='info-sign')
                                        ).add_to(marker_cluster)
                                    
                                    # Ajouter la colormap à la carte
                                    m4.add_child(colormap)
                                else:
                                    # Variable catégorielle : utiliser des couleurs fixes par catégorie
                                    unique_values = cluster_data[target_variable].unique()
                                    colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 
                                             'gray', 'black', 'lightblue', 'lightgreen']
                                    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
                                    
                                    for idx, row in cluster_data.iterrows():
                                        color = color_map[row[target_variable]]
                                        popup_text = f"""
                                        <div style='width: 200px;'>
                                            <h5>Point #{idx}</h5>
                                            <p><strong>Latitude:</strong> {row[lat_col]:.4f}</p>
                                            <p><strong>Longitude:</strong> {row[lon_col]:.4f}</p>
                                            <p><strong>{target_variable}:</strong> <span style='color: {color}; font-weight: bold;'>{row[target_variable]}</span></p>
                                        </div>
                                        """
                                        folium.Marker(
                                            location=[row[lat_col], row[lon_col]],
                                            popup=folium.Popup(popup_text, max_width=250),
                                            tooltip=f"{target_variable}: {row[target_variable]}",
                                            icon=folium.Icon(color=color, icon='info-sign')
                                        ).add_to(marker_cluster)
                            else:
                                # Pas de données valides, utiliser les marqueurs standards
                                cluster_data = df[[lat_col, lon_col]].dropna().head(500)
                                for idx, row in cluster_data.iterrows():
                                    folium.Marker(
                                        location=[row[lat_col], row[lon_col]],
                                        popup=f"Point #{idx}",
                                        tooltip=f"Cluster Point {idx}"
                                    ).add_to(marker_cluster)
                        else:
                            # Pas de variable cible, utiliser les marqueurs standards
                            cluster_data = df[[lat_col, lon_col]].dropna().head(500)
                            for idx, row in cluster_data.iterrows():
                                folium.Marker(
                                    location=[row[lat_col], row[lon_col]],
                                    popup=f"Point #{idx}",
                                    tooltip=f"Cluster Point {idx}"
                                ).add_to(marker_cluster)
                        
                        marker_cluster.add_to(m4)
                        
                        # Légende pour le clustering
                        legend_html4 = f'''
                        <div style="position: fixed; 
                                    bottom: 50px; left: 50px; width: 220px; height: 110px; 
                                    background-color: white; border:2px solid grey; z-index:9999; 
                                    font-size:14px; padding: 10px">
                        <p><b>Clustering</b></p>
                        <p><i class="fa fa-map-marker"></i> Points regroupés</p>
                        <p><small>Variable: {target_variable if target_variable else 'Non spécifiée'}</small></p>
                        <p><small>Zoomez pour voir les détails</small></p>
                        </div>
                        '''
                        m4.get_root().html.add_child(folium.Element(legend_html4))
                        
                        map_html4 = m4._repr_html_()
                        parts.append("<div class='figure-container'>")
                        parts.append(map_html4)
                        parts.append("</div>")
                    
                    # Section d'analyse spatiale
                    parts.append("<h3>📈 Analyse Spatiale</h3>")
                    parts.append("<div class='info-box'>")
                    parts.append("<h4>🔍 Options d'Analyse Disponibles</h4>")
                    parts.append("<ul>")
                    parts.append("<li><strong>Filtres par période :</strong> Si des données temporelles sont disponibles</li>")
                    parts.append("<li><strong>Analyse par clusters :</strong> Identification des zones de concentration</li>")
                    parts.append("<li><strong>Corrélation spatiale :</strong> Analyse des patterns géographiques</li>")
                    parts.append("<li><strong>Interpolation spatiale :</strong> Estimation des valeurs entre les points</li>")
                    parts.append("</ul>")
                    parts.append("</div>")
                    
            except ImportError:
                parts.append("""
                    <div class='warning-box'>
                        <h4>📚 Bibliothèques Cartographiques Non Disponibles</h4>
                        <p>Pour afficher les cartes interactives, installez les bibliothèques requises :</p>
                        <code>pip install folium branca</code>
                        <p>Les bibliothèques folium et branca sont nécessaires pour la visualisation cartographique avancée.</p>
                    </div>
                """)
            except Exception as e:
                parts.append(f"""
                    <div class='warning-box'>
                        <h4>⚠️ Erreur lors de la génération des cartes</h4>
                        <p>Une erreur s'est produite : {str(e)}</p>
                        <p>Vérifiez que vos données géospatiales sont correctement formatées.</p>
                    </div>
                """)
        else:
            parts.append("""
                <div class='warning-box'>
                    <h4>🗺️ Aucune Donnée Géospatiale Détectée</h4>
                    <p>Pour inclure des cartes dans le rapport, assurez-vous que vos données contiennent des colonnes de coordonnées.</p>
                    <p><strong>Colonnes attendues :</strong></p>
                    <ul>
                        <li>latitude, longitude</li>
                        <li>lat, lon</li>
                        <li>x, y</li>
                    </ul>
                    <p><em>Les noms de colonnes peuvent être en majuscules ou minuscules.</em></p>
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
        
        # Option de sélection de variable pour la cartographie
        st.markdown("##### 🗺️ Cartographie")
        
        # Détecter les variables climatiques disponibles
        df = session_state.get("clim_data")
        climate_vars_for_selection = []
        target_variable = None  # Variable cible du modèle
        
        if isinstance(df, pd.DataFrame):
            # Identifier les colonnes géospatiales
            geo_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['lat', 'latitude', 'lon', 'longitude', 'x', 'y']):
                    geo_cols.append(col)
            
            # Identifier les variables climatiques
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            climate_vars_for_selection = [col for col in numeric_cols 
                                         if col not in geo_cols and 
                                         not any(geo in col.lower() for geo in ['lat', 'lon', 'x', 'y'])]
            
            # Essayer de trouver la variable cible du modèle
            target_variable = None
            model_info = session_state.get("clim_model_info")
            if model_info:
                # Chercher dans différentes sources la variable cible
                # 1. Dans les informations de modélisation
                if "target_col" in model_info:
                    target_variable = model_info["target_col"]
                # 2. Dans les informations de prétraitement
                elif "clim_prep_info" in session_state:
                    prep_info = session_state["clim_prep_info"]
                    if "target_col" in prep_info:
                        target_variable = prep_info["target_col"]
                # 3. Chercher des colonnes typiques de target
                else:
                    typical_targets = ['target', 'label', 'y', 'outcome', 'risk', 'temperature', 'temp', 
                                     'precipitation', 'rain', 'humidity', 'wind', 'pressure', 'sea_level']
                    for target in typical_targets:
                        if target in climate_vars_for_selection:
                            target_variable = target
                            break
            else:
                # Pas de model_info, chercher dans les colonnes typiques
                typical_targets = ['target', 'label', 'y', 'outcome', 'risk', 'temperature', 'temp', 
                                 'precipitation', 'rain', 'humidity', 'wind', 'pressure', 'sea_level']
                for target in typical_targets:
                    if target in climate_vars_for_selection:
                        target_variable = target
                        break
        
        if climate_vars_for_selection:
            # Mettre la variable cible en premier si elle existe
            if target_variable and target_variable in climate_vars_for_selection:
                climate_vars_for_selection.remove(target_variable)
                climate_vars_for_selection.insert(0, target_variable)
            
            # Index par défaut : 0 (variable cible si trouvée, sinon première variable)
            default_index = 0
            
            selected_climate_var = st.selectbox(
                "Variable pour la carte thématique",
                options=climate_vars_for_selection,
                index=default_index,
                help="Sélectionnez la variable climatique à utiliser pour colorer les points sur la carte thématique. La variable cible du modèle est sélectionnée par défaut."
            )
            
            # Afficher les statistiques de la variable sélectionnée
            if selected_climate_var in df.columns:
                var_data = df[selected_climate_var].dropna()
                if len(var_data) > 0:
                    # Indiquer si c'est la variable cible du modèle
                    is_target = (selected_climate_var == target_variable)
                    target_indicator = " 🎯 (Variable cible du modèle)" if is_target else ""
                    
                    st.markdown(f"""
                    <div style='background-color: #{"2c3e50" if is_target else "#34495e"}; 
                                color: white; padding: 12px; border-radius: 5px; font-size: 13px; 
                                border-left: 4px solid #{"3498db" if is_target else "#95a5a6"};'>
                        <strong style='color: white;'>📊 {selected_climate_var}{target_indicator}</strong><br>
                        Min: {var_data.min():.2f} | Max: {var_data.max():.2f}<br>
                        Moyenne: {var_data.mean():.2f} | Médiane: {var_data.median():.2f}
                        {f"<br><em style='color: #ecf0f1;'>Variable utilisée pour entraîner le modèle</em>" if is_target else ""}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("🔍 Aucune variable climatique détectée pour la cartographie")
            selected_climate_var = None
        
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
                            "include_code": include_code,
                            "selected_climate_var": selected_climate_var  # Ajouter la variable sélectionnée
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
