"""
Module de génération de rapports HTML professionnels pour l'analyse de risque climatique
Crée des rapports consolidés avec visualisations et métriques pour les données climatiques
"""

from __future__ import annotations

import os
import base64
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuration des chemins
OUT_DIR = os.path.join("outputs", "reports")
os.makedirs(OUT_DIR, exist_ok=True)

# Configuration matplotlib pour de meilleurs rendus
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def _img_to_base64(fig: plt.Figure, width: int = 800) -> str:
    """Convertit une figure matplotlib en base64 pour HTML.
    
    Args:
        fig: Figure matplotlib à convertir
        width: Largeur maximale de l'image en pixels
        
    Returns:
        Chaîne HTML contenant l'image encodée en base64
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='#0b1120', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<div class="figure-container"><img src="data:image/png;base64,{img_str}" style="max-width:{width}px; width:100%; height:auto;"></div>'


def _wrap_table(html: str) -> str:
    """Enveloppe un tableau HTML dans un conteneur scrollable.
    
    Args:
        html: Code HTML du tableau
        
    Returns:
        Code HTML du tableau enveloppé
    """
    return f'<div class="table-container">{html}</div>'


def _get_climate_report_css() -> str:
    """Retourne le CSS moderne pour le rapport climatique.
    
    Returns:
        Chaîne CSS formatée pour le rapport
    """
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
            color: #e5e7eb;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 20px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: #0b1120;
            padding: 32px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.45);
            color: #e5e7eb;
        }

        h1 {
            color: #facc15;
            font-size: 2.5em;
            margin-bottom: 20px;
            border-bottom: 3px solid #facc15;
            padding-bottom: 15px;
            text-align: center;
        }

        h2 {
            color: #facc15;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 5px solid #facc15;
            background: #1e293b;
            padding: 15px;
            border-radius: 5px;
        }

        h3 {
            color: #60a5fa;
            font-size: 1.4em;
            margin-top: 25px;
            margin-bottom: 15px;
        }

        h4 {
            color: #9ca3af;
            font-size: 1.1em;
            margin-top: 20px;
            margin-bottom: 10px;
            font-weight: 600;
        }

        p {
            margin: 10px 0;
            font-size: 1em;
            color: #e5e7eb;
        }

        .metric-box {
            display: inline-block;
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            padding: 15px 25px;
            margin: 10px 10px 10px 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
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

        .table-container {
            width: 100%;
            overflow-x: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 1px solid #1f2937;
            background: #020617;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: #020617;
            color: #e5e7eb;
            font-size: 0.85em;
            min-width: 600px;
        }

        thead {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        th, td {
            padding: 8px 12px;
            border-bottom: 1px solid #1f2937;
            text-align: left;
        }

        th {
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }

        tbody tr:nth-child(even) {
            background: #0f172a;
        }

        .info-box {
            background: #1e293b;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .warning-box {
            background: #451a03;
            border-left: 4px solid #f97316;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
        }

        .success-box {
            background: #064e3b;
            border-left: 4px solid #10b981;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
        }

        ul, ol {
            margin: 10px 0 10px 25px;
        }

        li {
            margin: 8px 0;
            line-height: 1.5;
        }

        code {
            background: #1e293b;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #f472b6;
        }

        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #1f2937;
            text-align: center;
            color: #9ca3af;
            font-size: 0.85em;
        }

        .figure-container {
            margin: 20px 0;
            text-align: center;
        }

        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }

        .metric-box.secondary {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        }

        .metric-box.warning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }

        .metric-box.danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }
    </style>
    """


def show_reporting_summary(session_state: Dict[str, Any]) -> None:
    """Affiche un résumé du projet climatique dans l'interface Streamlit.
    
    Args:
        session_state: État de la session Streamlit contenant les données du projet
    """
    st.subheader("📊 Synthèse du Projet Climatique")

    # Section Données
    with st.expander("📂 Données", expanded=True):
        cols = st.columns(2)
        
        if "clim_data" in session_state:
            df = session_state["clim_data"]
            cols[0].metric("Données initiales", f"{df.shape[0]:,} lignes × {df.shape[1]:,} colonnes")
            
            # Aperçu des données
            if st.checkbox("Afficher un aperçu des données"):
                st.dataframe(df.head())
        
        if "clim_data_prep" in session_state and session_state["clim_data_prep"] is not None:
            df_prep = session_state["clim_data_prep"]
            cols[1].metric("Données prétraitées", f"{df_prep.shape[0]:,} lignes × {df_prep.shape[1]:,} colonnes")
    
    # Section Prétraitement
    if "clim_prep_info" in session_state and session_state["clim_prep_info"]:
        prep_info = session_state["clim_prep_info"]
        with st.expander("🔧 Prétraitement", expanded=False):
            cols = st.columns(3)
            
            if "date_col" in prep_info:
                cols[0].metric("Colonne date", prep_info['date_col'])
            
            if "freq" in prep_info:
                cols[1].metric("Fréquence", prep_info['freq'])
                
            if "shape" in prep_info:
                rows, cols_count = prep_info['shape']
                cols[2].metric("Dimensions après prétraitement", f"{rows} × {cols_count}")
            
            # Détails supplémentaires
            details = []
            if prep_info.get("rolling"):
                details.append("✅ Features temporelles (rolling)")
            if prep_info.get("anomaly_summary"):
                details.append("✅ Détection d'anomalies")
            if details:
                st.markdown("### Détails du prétraitement")
                for detail in details:
                    st.markdown(f"- {detail}")
    
    # Section Modélisation
    if "clim_model_info" in session_state and session_state["clim_model_info"]:
        model_info = session_state["clim_model_info"]
        with st.expander("🤖 Modélisation", expanded=False):
            cols = st.columns(3)
            
            # Métriques principales
            if "model_name" in model_info:
                cols[0].metric("Modèle", model_info['model_name'])
            
            if "task_type" in model_info:
                cols[1].metric("Type de tâche", model_info['task_type'])
            
            # Affichage des métriques
            if "metric_name" in model_info and "metric_value" in model_info:
                metric_name = model_info["metric_name"]
                metric_value = model_info["metric_value"]
                if metric_value is not None:
                    cols[2].metric(metric_name.upper(), f"{metric_value:.4f}")
            
            # Détails supplémentaires
            details = []
            if model_info.get("handle_imbalance"):
                details.append("⚖️ Gestion du déséquilibre des classes")
            if "cv_scores" in model_info and model_info["cv_scores"] is not None:
                cv_mean = np.mean(model_info["cv_scores"])
                cv_std = np.std(model_info["cv_scores"])
                details.append(f"📊 Validation croisée: {cv_mean:.4f} ± {cv_std:.4f}")
            
            if details:
                st.markdown("### Détails du modèle")
                for detail in details:
                    st.markdown(f"- {detail}")
    
    # Bouton de génération du rapport
    st.markdown("---")
    if st.button("📊 Générer le rapport complet", use_container_width=True, type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            try:
                report_path = generate_html_report(session_state)
                if report_path:
                    st.success("✅ Rapport généré avec succès!")
                    
                    # Aperçu du rapport
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    
                    # Téléchargement du rapport
                    st.download_button(
                        label="💾 Télécharger le rapport complet",
                        data=report_content,
                        file_name=os.path.basename(report_path),
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    # Aperçu intégré
                    st.components.v1.html(report_content, height=800, scrolling=True)
                else:
                    st.error("❌ Erreur lors de la génération du rapport")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")


def _create_time_series_plot(df: pd.DataFrame, date_col: str, value_col: str, title: str) -> plt.Figure:
    """Crée un graphique de série temporelle.
    
    Args:
        df: DataFrame contenant les données
        date_col: Nom de la colonne de date
        value_col: Nom de la colonne de valeurs
        title: Titre du graphique
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0b1120')
    
    # Style du graphique
    ax.set_facecolor('#0b1120')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    
    # Tracé de la série temporelle
    ax.plot(df[date_col], df[value_col], color='#60a5fa', linewidth=1.5)
    
    # Mise en forme
    ax.set_title(title, color='white', pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', color='#9ca3af', fontsize=11)
    ax.set_ylabel(value_col, color='#9ca3af', fontsize=11)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, linestyle='--', alpha=0.3, color='#334155')
    
    # Rotation des étiquettes de l'axe des x
    plt.xticks(rotation=45, ha='right')
    
    # Ajustement des marges
    plt.tight_layout()
    
    return fig


def _create_feature_importance_plot(feature_names: List[str], importances: np.ndarray, top_n: int = 10) -> plt.Figure:
    """Crée un graphique d'importance des caractéristiques.
    
    Args:
        feature_names: Liste des noms des caractéristiques
        importances: Tableau des importances
        top_n: Nombre de caractéristiques à afficher
        
    Returns:
        Figure matplotlib
    """
    # Tri des caractéristiques par importance
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0b1120')
    
    # Style du graphique
    ax.set_facecolor('#0b1120')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    
    # Tracé des barres
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, align='center', color='#60a5fa')
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.02, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                va='center', ha='left', color='white', fontsize=9)
    
    # Mise en forme
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color='white')
    ax.tick_params(axis='x', colors='#9ca3af')
    ax.set_title('Top 10 des caractéristiques les plus importantes', 
                 color='white', pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', color='#9ca3af', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.2, color='#334155', axis='x')
    
    # Ajustement des marges
    plt.tight_layout()
    
    return fig


def generate_html_report(session_state: Dict[str, Any]) -> Optional[str]:
    """Génère un rapport HTML complet pour l'analyse de risque climatique.
    
    Args:
        session_state: État de la session contenant les données du projet
        
    Returns:
        Chemin vers le fichier HTML généré ou None en cas d'erreur
    """
    try:
        # Récupération des données de la session
        df = session_state.get("clim_data")
        df_prep = session_state.get("clim_data_prep")
        prep_info = session_state.get("clim_prep_info", {})
        model_info = session_state.get("clim_model_info", {})
        framing = session_state.get("project_framing", {})
        data_sources = session_state.get("data_sources", {})
        
        # Création du répertoire de sortie si nécessaire
        os.makedirs(OUT_DIR, exist_ok=True)
        
        # Nom du fichier de sortie avec horodatage
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"rapport_climat_{now}.html"
        out_path = os.path.join(OUT_DIR, filename)
        
        # Initialisation des parties du rapport
        parts: List[str] = []
        
        # En-tête du document
        parts.extend([
            "<!DOCTYPE html>",
            "<html lang='fr'>",
            "<head>",
            "    <meta charset='utf-8'>",
            "    <title>Rapport d'Analyse de Risque Climatique</title>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "    <link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap' rel='stylesheet'>",
            _get_climate_report_css(),
            "</head>",
            "<body>",
            "<div class='container'>"
        ])
        
        # En-tête du rapport
        parts.extend([
            "<header>",
            "    <h1>🌍 Rapport d'Analyse de Risque Climatique</h1>",
            "    <div class='info-box'>",
            f"       <p><strong>📅 Date de génération :</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>",
            "    </div>",
            "</header>"
        ])
        
        # 1. Résumé exécutif
        parts.append("<section id='executive-summary'>")
        parts.append("    <h2>📋 Résumé Exécutif</h2>")
        parts.append("    <div class='info-box'>")
        
        # Ajout des métriques clés
        parts.append("        <div class='metrics-container'>")
        
        # Métrique 1: Données initiales
        if isinstance(df, pd.DataFrame):
            parts.append(f"""
                <div class='metric-box'>
                    <span class='metric-label'>Données Initiales</span>
                    <span class='metric-value'>{df.shape[0]:,} × {df.shape[1]}</span>
                </div>
            """)
        
        # Métrique 2: Données prétraitées
        if isinstance(df_prep, pd.DataFrame):
            parts.append(f"""
                <div class='metric-box'>
                    <span class='metric-label'>Données Prétraitées</span>
                    <span class='metric-value'>{df_prep.shape[0]:,} × {df_prep.shape[1]}</span>
                </div>
            """)
        
        # Métrique 3: Performance du modèle
        if "metric_value" in model_info and model_info["metric_value"] is not None:
            metric_name = model_info.get("metric_name", "Métrique")
            metric_value = model_info["metric_value"]
            
            parts.append(f"""
                <div class='metric-box secondary'>
                    <span class='metric-label'>{metric_name.upper()}</span>
                    <span class='metric-value'>{metric_value:.4f}</span>
                </div>
            """)
        
        parts.append("        </div>")
        
        # Résumé textuel
        parts.append("        <h3>Contexte</h3>")
        if framing.get("objective_desc"):
            parts.append(f"<p>{framing['objective_desc']}</p>")
        else:
            parts.append("<p>Ce rapport présente les résultats de l'analyse de risque climatique réalisée avec l'outil d'analyse de données climatiques.</p>")
        
        parts.append("    </div>")  # Fin de la boîte d'info
        parts.append("</section>")
        
        # 2. Données et sources
        parts.append("<section id='data-sources'>")
        parts.append("    <h2>📂 Sources de Données</h2>")
        
        if data_sources:
            parts.append("    <div class='info-box'>")
            parts.append("        <ul>")
            for label, source_df in data_sources.items():
                if isinstance(source_df, pd.DataFrame):
                    parts.append(f"<li><strong>{label}</strong>: {source_df.shape[0]:,} lignes × {source_df.shape[1]:,} colonnes</li>")
            parts.append("        </ul>")
            parts.append("    </div>")
        else:
            parts.append("    <div class='warning-box'>")
            parts.append("        <p>Aucune source de données n'a été spécifiée.</p>")
            parts.append("    </div>")
        parts.append("</section>")
        
        # 3. Analyse exploratoire des données
        if isinstance(df, pd.DataFrame):
            parts.append("<section id='exploratory-analysis'>")
            parts.append("    <h2>🔍 Analyse Exploratoire des Données</h2>")
            
            # Aperçu des données
            parts.append("    <h3>📋 Aperçu des Données Initiales</h3>")
            parts.append(_wrap_table(df.head().to_html(classes='dataframe', index=False)))
            
            # Statistiques descriptives
            parts.append("    <h3>📊 Statistiques Descriptives</h3>")
            parts.append(_wrap_table(df.describe(include='all').round(2).to_html(classes='dataframe')))
            
            # Visualisation des séries temporelles (si une colonne de date est disponible)
            date_col = prep_info.get('date_col')
            if date_col and date_col in df.columns and df[date_col].dtype == 'datetime64[ns]':
                parts.append("    <h3>📈 Séries Temporelles</h3>")
                
                # Sélection des colonnes numériques pour le tracé
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Limite à 3 variables pour éviter la surcharge
                for i, col in enumerate(numeric_cols[:3]):
                    try:
                        fig = _create_time_series_plot(df, date_col, col, f"Évolution de {col}")
                        parts.append(_img_to_base64(fig))
                    except Exception as e:
                        parts.append(f"<p class='warning'>Erreur lors de la génération du graphique pour {col}: {str(e)}</p>")
            
            parts.append("</section>")
        
        # 4. Prétraitement des données
        if prep_info or isinstance(df_prep, pd.DataFrame):
            parts.append("<section id='data-preprocessing'>")
            parts.append("    <h2>🔧 Prétraitement des Données</h2>")
            
            # Détails du prétraitement
            if prep_info:
                parts.append("    <h3>Étapes de Prétraitement</h3>")
                parts.append("    <ul>")
                
                if "date_col" in prep_info:
                    parts.append(f"<li>Colonne temporelle : <code>{prep_info['date_col']}</code></li>")
                
                if "freq" in prep_info:
                    parts.append(f"<li>Fréquence d'agrégation : <code>{prep_info['freq']}</code></li>")
                
                if prep_info.get("rolling"):
                    parts.append("<li>Calcul des indicateurs mobiles (moyenne mobile, écart-type, etc.)</li>")
                
                if prep_info.get("anomaly_summary"):
                    parts.append("<li>Détection et gestion des anomalies (méthode z-score)</li>")
                
                if "shape" in prep_info and isinstance(prep_info["shape"], (list, tuple)) and len(prep_info["shape"]) == 2:
                    rows, cols = prep_info["shape"]
                    parts.append(f"<li>Dimensions après prétraitement : {rows} lignes × {cols} colonnes</li>")
                
                parts.append("    </ul>")
            
            # Aperçu des données prétraitées
            if isinstance(df_prep, pd.DataFrame):
                parts.append("    <h3>📋 Aperçu des Données Prétraitées</h3>")
                parts.append(_wrap_table(df_prep.head().to_html(classes='dataframe', index=False)))
                
                # Statistiques après prétraitement
                parts.append("    <h3>📊 Statistiques après Prétraitement</h3>")
                parts.append(_wrap_table(df_prep.describe(include='all').round(2).to_html(classes='dataframe')))
            
            parts.append("</section>")
        
        # 5. Modélisation
        if model_info:
            parts.append("<section id='modeling'>")
            parts.append("    <h2>🤖 Modélisation</h2>")
            
            # Informations générales sur le modèle
            parts.append("    <div class='info-box'>")
            
            if "model_name" in model_info:
                parts.append(f"<h3>Modèle: {model_info['model_name']}</h3>")
            
            if "task_type" in model_info:
                parts.append(f"<p><strong>Type de tâche :</strong> {model_info['task_type']}</p>")
            
            # Métriques de performance
            parts.append("    <h4>Performance du Modèle</h4>")
            parts.append("    <div class='metrics-container'>")
            
            # Métrique principale
            if "metric_name" in model_info and model_info["metric_value"] is not None:
                metric_name = model_info["metric_name"]
                metric_value = model_info["metric_value"]
                parts.append(f"""
                    <div class='metric-box'>
                        <span class='metric-label'>{metric_name.upper()}</span>
                        <span class='metric-value'>{metric_value:.4f}</span>
                    </div>
                """)
            
            # Score F1 pour la classification
            if "f1_score" in model_info and model_info["f1_score"] is not None:
                f1 = model_info["f1_score"]
                parts.append(f"""
                    <div class='metric-box secondary'>
                        <span class='metric-label'>F1-SCORE</span>
                        <span class='metric-value'>{f1:.4f}</span>
                    </div>
                """)
            
            # Validation croisée
            if "cv_scores" in model_info and model_info["cv_scores"] is not None:
                cv_scores = model_info["cv_scores"]
                if len(cv_scores) > 0:
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                    parts.append(f"""
                        <div class='metric-box warning'>
                            <span class='metric-label'>VALIDATION CROISÉE</span>
                            <span class='metric-value'>{cv_mean:.4f} ± {cv_std:.4f}</span>
                        </div>
                    """)
            
            parts.append("    </div>")  # Fin du conteneur de métriques
            
            # Détails supplémentaires sur le modèle
            details = []
            
            if model_info.get("handle_imbalance"):
                details.append("Gestion du déséquilibre des classes activée")
            
            if model_info.get("used_stratify") is False and model_info.get("task_type") == "classification":
                details.append("Stratification désactivée en raison du déséquilibre des classes")
            
            if details:
                parts.append("    <h4>Détails du Modèle</h4>")
                parts.append("    <ul>")
                for detail in details:
                    parts.append(f"<li>{detail}</li>")
                parts.append("    </ul>")
            
            # Importance des caractéristiques
            if "feature_importance" in model_info and "feature_names" in model_info:
                feat_imp = model_info["feature_importance"]
                feat_names = model_info["feature_names"]
                
                if len(feat_imp) > 0 and len(feat_names) == len(feat_imp):
                    try:
                        fig = _create_feature_importance_plot(feat_names, feat_imp)
                        parts.append("    <h4>Importance des Caractéristiques</h4>")
                        parts.append(_img_to_base64(fig))
                    except Exception as e:
                        parts.append(f"<p class='warning'>Erreur lors de la génération du graphique d'importance des caractéristiques: {str(e)}</p>")
            
            parts.append("    </div>")  # Fin de la boîte d'info
            parts.append("</section>")
        
        # 6. Résultats et Interprétation
        parts.append("<section id='results'>")
        parts.append("    <h2>📊 Résultats et Interprétation</h2>")
        
        if model_info:
            parts.append("    <div class='info-box'>")
            
            # Interprétation des résultats
            parts.append("        <h3>Interprétation des Résultats</h3>")
            
            if model_info.get("task_type") == "classification":
                parts.append("""
                    <p>Le modèle de classification a été entraîné pour prédire les risques climatiques en fonction des caractéristiques d'entrée.</p>
                    <p>Les métriques de performance indiquent la capacité du modèle à distinguer entre les différentes classes de risque.</p>
                """)
            elif model_info.get("task_type") == "regression":
                parts.append("""
                    <p>Le modèle de régression a été entraîné pour prédire des valeurs continues liées aux risques climatiques.</p>
                    <p>Les métriques de performance indiquent la précision des prédictions du modèle par rapport aux valeurs réelles.</p>
                """)
            
            # Recommandations basées sur les résultats
            parts.append("        <h3>Recommandations</h3>")
            parts.append("        <ul>")
            parts.append("            <li>Valider les résultats avec des experts du domaine pour une interprétation contextuelle</li>")
            parts.append("            <li>Considérer les incertitudes liées aux données et aux hypothèses du modèle</li>")
            parts.append("            <li>Mettre en place un suivi continu des performances du modèle en production</li>")
            parts.append("        </ul>")
            
            parts.append("    </div>")  # Fin de la boîte d'info
        else:
            parts.append("    <div class='warning-box'>")
            parts.append("        <p>Aucun modèle n'a été entraîné ou les résultats ne sont pas disponibles.</p>")
            parts.append("    </div>")
        
        parts.append("</section>")
        
        # 7. Limites et Perspectives
        parts.append("<section id='limitations'>")
        parts.append("    <h2>⚠️ Limites et Perspectives</h2>")
        parts.append("    <div class='warning-box'>")
        parts.append("        <h3>Limites de l'Analyse</h3>")
        parts.append("        <ul>")
        parts.append("            <li>Ce modèle a été développé dans un contexte de prototype et nécessite une validation supplémentaire</li>")
        parts.append("            <li>Les performances peuvent varier en fonction de la qualité et de la représentativité des données</li>")
        parts.append("            <li>Les résultats doivent être interprétés avec prudence et dans leur contexte</li>")
        parts.append("        </ul>")
        
        parts.append("        <h3>Perspectives d'Amélioration</h3>")
        parts.append("        <ul>")
        parts.append("            <li>Enrichir le jeu de données avec des sources complémentaires</li>")
        parts.append("            <li>Expérimenter avec d'autres algorithmes et techniques de prétraitement</li>")
        parts.append("            <li>Mettre en place une validation croisée temporelle pour évaluer la robustesse du modèle</li>")
        parts.append("            <li>Intégrer des mécanismes de suivi de la dérive des données en production</li>")
        parts.append("        </ul>")
        parts.append("    </div>")  # Fin de la boîte d'avertissement
        parts.append("</section>")
        
        # Pied de page
        parts.extend([
            "<footer class='footer'>",
            "    <p>Rapport généré automatiquement par l'outil d'Analyse de Risque Climatique</p>",
            f"    <p>© {datetime.now().year} - Tous droits réservés</p>",
            "</footer>"
        ])
        
        # Fermeture des balises
        parts.extend([
            "</div>",  # Fin du conteneur
            "</body>",
            "</html>"
        ])
        
        # Écriture du fichier HTML
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(parts))
        
        return out_path
        
    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exemple d'utilisation
    import pandas as pd
    import numpy as np
    
    # Création d'un exemple de session_state pour les tests
    class SessionState:
        def __init__(self):
            self._state = {
                "clim_data": pd.DataFrame({
                    'date': pd.date_range(start='2020-01-01', periods=100, freq='M'),
                    'temperature': np.random.normal(25, 5, 100),
                    'precipitation': np.random.gamma(2, 2, 100),
                    'humidity': np.random.uniform(30, 90, 100)
                }),
                "clim_data_prep": pd.DataFrame({
                    'date': pd.date_range(start='2020-01-01', periods=100, freq='M'),
                    'temp_avg': np.random.normal(25, 2, 100),
                    'precip_total': np.random.gamma(2, 2, 100),
                    'humidity_avg': np.random.uniform(40, 80, 100)
                }),
                "clim_prep_info": {
                    "date_col": "date",
                    "freq": "M",
                    "rolling": True,
                    "anomaly_summary": True,
                    "shape": (100, 4)
                },
                "clim_model_info": {
                    "model_name": "RandomForestClassifier",
                    "task_type": "classification",
                    "metric_name": "accuracy",
                    "metric_value": 0.85,
                    "f1_score": 0.82,
                    "handle_imbalance": True,
                    "used_stratify": True,
                    "cv_scores": np.array([0.82, 0.84, 0.83, 0.85, 0.81]),
                    "feature_importance": np.array([0.4, 0.3, 0.2, 0.1]),
                    "feature_names": ["temp_avg", "precip_total", "humidity_avg", "wind_speed"]
                },
                "project_framing": {
                    "objective_type": "Prédiction des risques climatiques",
                    "objective_desc": "Évaluation des risques liés aux conditions climatiques extrêmes",
                    "unit_of_analysis": "Mensuelle",
                    "target_desc": "Catégorie de risque (Faible, Moyen, Élevé)",
                    "context": "Analyse des tendances climatiques pour la planification des risques"
                },
                "data_sources": {
                    "Données météorologiques": pd.DataFrame({
                        'date': pd.date_range(start='2020-01-01', periods=100, freq='M'),
                        'temperature': np.random.normal(25, 5, 100),
                        'precipitation': np.random.gamma(2, 2, 100)
                    }),
                    "Données d'humidité": pd.DataFrame({
                        'date': pd.date_range(start='2020-01-01', periods=100, freq='M'),
                        'humidity': np.random.uniform(30, 90, 100)
                    })
                }
            }
            
        def __getitem__(self, key):
            return self._state.get(key)
            
        def get(self, key, default=None):
            return self._state.get(key, default)
    
    # Test de la fonction de génération de rapport
    session = SessionState()
    report_path = generate_html_report(session)
    print(f"Rapport généré : {report_path}")
    
    # Test de l'affichage dans Streamlit (si disponible)
    try:
        st.set_page_config(page_title="Rapport Climat", layout="wide")
        st.title("Test du Module de Rapport Climatique")
        show_reporting_summary(session)
    except:
        print("Streamlit n'est pas disponible pour l'affichage")
