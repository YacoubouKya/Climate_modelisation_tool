"""
Module de génération de rapports HTML professionnels pour l'analyse de risque climatique
Crée des rapports consolidés avec visualisations et métriques pour les données climatiques
"""

from __future__ import annotations

import os
import base64
from datetime import datetime
from io import BytesIO
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
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white', edgecolor='none')
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
            color: #333;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: white;
            padding: 32px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 20px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            text-align: center;
        }

        h2 {
            color: #34495e;
            font-size: 1.8em;
            margin-top: 30px;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 5px solid #facc15;
            background: #f8f9fa;
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
            background: #e8f4f8;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #f97316;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
        }

        .success-box {
            background: #d4edda;
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
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e91e63;
            border: 1px solid #e9ecef;
        }

        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
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


def _img_to_base64(fig, width=800):
    """Convertit une figure matplotlib en base64 pour HTML"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<div class="figure-container"><img src="data:image/png;base64,{img_str}" style="max-width:{width}px; width:100%; height:auto;"></div>'

def _wrap_table(table_html: str) -> str:
    """Enveloppe un tableau HTML dans un container scrollable"""
    return f'<div class="table-container">{table_html}</div>'

def _get_modern_css():
    """Retourne le CSS moderne pour le rapport"""
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
        
        .table-container {
            content: "← Faites défiler horizontalement →";
            display: block;
            text-align: center;
            font-size: 0.75em;
            color: #95a5a6;
            padding: 5px;
            font-style: italic;
        }
        
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
            .tabs-header, .tab-button, .modal {
                display: none !important;
            }
            .tab-content {
                display: block !important;
                opacity: 1 !important;
                position: relative !important;
            }
        }
        
        /* Styles pour les onglets */
        .tabs-container {
            margin: 25px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .tabs-header {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        
        .tab-button {
            background: none;
            border: none;
            padding: 12px 20px;
            font-size: 0.9em;
            font-weight: 600;
            color: #6c757d;
            cursor: pointer;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .tab-button:hover {
            background: #e9ecef;
            color: #495057;
        }
        
        .tab-button.active {
            color: #3498db;
            border-bottom: 3px solid #3498db;
            background: white;
        }
        
        .tab-content {
            display: none;
            padding: 20px;
            background: white;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Grille d'informations sur les données */
        .data-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .data-info-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e9ecef;
        }
        
        .data-info-card h4 {
            margin-top: 0;
            color: #495057;
            font-size: 1em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Cartes de sources */
        .sources-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .source-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid #e9ecef;
        }
        
        .source-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        
        .source-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .source-header i {
            font-size: 1.5em;
            color: #3498db;
        }
        
        .source-header h4 {
            margin: 0;
            color: #2c3e50;
            font-size: 1.1em;
        }
        
        .source-body {
            padding: 15px;
        }
        
        .source-body p {
            margin: 8px 0;
            font-size: 0.9em;
            color: #6c757d;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .source-body i {
            width: 20px;
            color: #6c757d;
        }
        
        .source-footer {
            padding: 10px 15px;
            border-top: 1px solid #e9ecef;
            text-align: right;
        }
        
        /* Score de qualité */
        .quality-score {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .quality-score.success {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
        }
        
        .quality-score.warning {
            background: #fff8e1;
            border-left: 4px solid #ffc107;
        }
        
        .quality-score.danger {
            background: #ffebee;
            border-left: 4px solid #f44336;
        }
        
        .score-circle {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-weight: bold;
        }
        
        .score-circle .score {
            font-size: 1.8em;
            line-height: 1;
        }
        
        .score-circle .score-label {
            font-size: 0.8em;
            opacity: 0.7;
        }
        
        .score-details h4 {
            margin: 0 0 5px 0;
            color: #2c3e50;
            font-size: 1.2em;
        }
        
        .score-details p {
            margin: 0;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .quality-issues {
            margin: 20px 0;
            padding: 15px;
            background: #fff8e1;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
        
        .quality-issues h4 {
            margin-top: 0;
            color: #e65100;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .quality-issues ul {
            margin: 10px 0 0 0;
            padding-left: 20px;
        }
        
        .quality-issues li {
            margin-bottom: 5px;
            color: #6c757d;
        }
        
        /* Modales */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
            overflow: auto;
            padding: 20px;
            box-sizing: border-box;
        }
        
        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 25px;
            border-radius: 8px;
            max-width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
            box-shadow: 0 5px 30px rgba(0,0,0,0.3);
            animation: modalFadeIn 0.3s;
        }
        
        @keyframes modalFadeIn {
            from { opacity: 0; transform: translateY(-50px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .close {
            position: absolute;
            right: 20px;
            top: 15px;
            font-size: 28px;
            font-weight: bold;
            color: #6c757d;
            cursor: pointer;
            transition: color 0.3s;
        }
        
        .close:hover {
            color: #343a40;
        }
        
        .modal-footer {
            margin-top: 20px;
            text-align: right;
        }
        
        /* Alertes */
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .alert.success {
            background: #e8f5e9;
            color: #2e7d32;
            border-left: 4px solid #4caf50;
        }
        
        .alert.warning {
            background: #fff8e1;
            color: #e65100;
            border-left: 4px solid #ffc107;
        }
        
        .alert.info {
            background: #e3f2fd;
            color: #1565c0;
            border-left: 4px solid #2196f3;
        }
    </style>
    """

def show_reporting_summary(session_state: dict) -> None:
    """Affiche l'interface de génération de rapport dans Streamlit"""
    st.subheader("📝 Générer un rapport consolidé")
    
    # Vérification des données disponibles
    has_data = "clim_data" in session_state and session_state["clim_data"] is not None
    has_prep = "clim_data_prep" in session_state and session_state["clim_data_prep"] is not None
    has_model = "clim_model_info" in session_state and session_state["clim_model_info"]
    
    if not has_data:
        st.warning("⚠️ Aucune donnée n'a été chargée. Veuillez d'abord charger des données.")
        return
    
    # Options de personnalisation
    col1, col2 = st.columns(2)
    with col1:
        title_default = f"Rapport_Climat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        title = st.text_input("Titre du rapport", value=title_default)
    
    with col2:
        # Options de sections basées sur les données disponibles
        section_options = [
            ("Données", "data", True),
            ("Prétraitement", "preprocessing", has_prep),
            ("Analyse exploratoire", "exploration", True),
            ("Modélisation", "modeling", has_model),
            ("Visualisations", "visualizations", has_prep or has_model),
            ("Recommandations", "recommendations", True)
        ]
        
        # Sélection des sections
        selected_sections = st.multiselect(
            "Sections à inclure",
            [opt[0] for opt in section_options if opt[2]],
            default=[opt[0] for opt in section_options if opt[2] and opt[0] != "Visualisations"]
        )
    
    include_plots = st.checkbox("Inclure les visualisations détaillées", value=True)
    
    if st.button(" Créer rapport HTML", type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            try:
                # Créer le contexte du rapport
                report_context = {
                    **session_state,
                    "report_options": {
                        "title": title,
                        "sections": selected_sections,
                        "include_plots": include_plots
                    }
                }
                
                # Générer le rapport
                report_path = generate_html_report(report_context)
                
                if report_path:
                    st.success("✅ Rapport généré avec succès !")
                    
                    # Aperçu du rapport
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_content = f.read()
                    
                    # Téléchargement du rapport
                    st.download_button(
                        label=" Télécharger le rapport HTML",
                        data=report_content,
                        file_name=os.path.basename(report_path),
                        mime="text/html",
                        type="primary"
                    )
                    
                    # Aperçu intégré
                    st.markdown("---")
                    st.markdown("### 👁️ Aperçu du rapport")
                    st.components.v1.html(report_content, height=800, scrolling=True)
                else:
                    st.error("❌ Erreur lors de la génération du rapport")
                    
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
                st.exception(e)


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
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    
    # Style du graphique
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e9ecef')
    
    # Tracé de la série temporelle
    ax.plot(df[date_col], df[value_col], color='#3498db', linewidth=1.5)
    
    # Mise en forme
    ax.set_title(title, color='#2c3e50', pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', color='#6c757d', fontsize=11)
    ax.set_ylabel(value_col, color='#6c757d', fontsize=11)
    ax.tick_params(colors='#6c757d')
    ax.grid(True, linestyle='--', alpha=0.3, color='#dee2e6')
    
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
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Style du graphique
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e9ecef')
    
    # Tracé des barres
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, align='center', color='#3498db')
    
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
    ax.set_title('Top 10 des caractéristiques les plus importantes', 
                 color='#2c3e50', pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', color='#6c757d', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.2, color='#dee2e6', axis='x')
    
    # Ajustement des marges
    plt.tight_layout()
    
    return fig


def generate_html_report(session_state: Dict[str, Any]) -> Optional[str]:
    """
    Génère un rapport HTML complet pour l'analyse de risque climatique.
    
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
        
        # Récupération des options du rapport
        report_options = session_state.get("report_options", {
            "title": f"Rapport_Climat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "sections": ["Données", "Analyse exploratoire", "Recommandations"],
            "include_plots": True
        })
        
        title = report_options.get("title", "Rapport d'Analyse Climatique")
        selected_sections = report_options.get("sections", [])
        include_plots = report_options.get("include_plots", True)
        
        # Création du répertoire de sortie si nécessaire
        os.makedirs(OUT_DIR, exist_ok=True)
        
        # Nom du fichier de sortie avec horodatage
        safe_title = title.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(OUT_DIR, f"{safe_title}.html")
        
        # Initialisation des parties du rapport
        parts = []
        
        # En-tête du document
        parts.extend([
            "<!DOCTYPE html>",
            "<html lang='fr'>",
            "<head>",
            "    <meta charset='utf-8'>",
            f"    <title>{title}</title>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes'>",
            "    <link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap' rel='stylesheet'>",
            "    <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'>",
            _get_modern_css(),
            "</head>",
            "<body>",
            "<div class='container'>"
        ])
        
        # Barre de navigation latérale
        parts.extend([
            "<div class='sidebar'>",
            "    <div class='sidebar-header'>",
            "        <h3><i class='fas fa-bars'></i> Navigation</h3>",
            "    </div>",
            "    <ul class='nav-links'>",
            "        <li><a href='#executive-summary'><i class='fas fa-home'></i> Résumé Exécutif</a></li>"
        ])
        
        # Ajout des liens de navigation dynamiques
        if "Données" in selected_sections:
            parts.append("        <li><a href='#data-sources'><i class='fas fa-database'></i> Données et Sources</a></li>")
        
        if "Analyse exploratoire" in selected_sections and isinstance(df, pd.DataFrame):
            parts.append("        <li><a href='#exploratory-analysis'><i class='fas fa-chart-line'></i> Analyse Exploratoire</a></li>")
        
        if "Prétraitement" in selected_sections and (prep_info or isinstance(df_prep, pd.DataFrame)):
            parts.append("        <li><a href='#data-preprocessing'><i class='fas fa-tools'></i> Prétraitement</a></li>")
        
        if "Modélisation" in selected_sections and model_info:
            parts.append("        <li><a href='#modeling'><i class='fas fa-robot'></i> Modélisation</a></li>")
            parts.append("        <li><a href='#results'><i class='fas fa-chart-bar'></i> Résultats</a></li>")
        
        if "Recommandations" in selected_sections:
            parts.append("        <li><a href='#recommendations'><i class='fas fa-lightbulb'></i> Recommandations</a></li>")
            
        parts.extend([
            "        <li><a href='#limitations'><i class='fas fa-exclamation-triangle'></i> Limites</a></li>",
            "    </ul>",
            "    <div class='sidebar-footer'>",
            f"        <p><i class='far fa-calendar-alt'></i> {datetime.now().strftime('%d/%m/%Y')}</p>",
            "    </div>",
            "</div>"
        ])
        
        # Contenu principal
        parts.append("<div class='main-content'>")
        
        # En-tête du rapport
        parts.extend([
            "<header class='report-header'>",
            f"    <h1><i class='fas fa-chart-pie'></i> {title}</h1>",
            "    <div class='report-meta'>",
            f"        <p><i class='far fa-calendar-alt'></i> Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>",
            "    </div>",
            "</header>"
        ])
        
        # Section 1: Résumé exécutif
        parts.append("<section id='executive-summary' class='section-card'>")
        parts.append("    <div class='section-header'>")
        parts.append("        <h2><i class='fas fa-home'></i> Résumé Exécutif</h2>")
        parts.append("        <div class='section-actions'>")
        parts.append("            <button class='btn btn-print' onclick='window.print()'><i class='fas fa-print'></i> Imprimer</button>")
        parts.append("            <button class='btn btn-pdf'><i class='far fa-file-pdf'></i> Exporter en PDF</button>")
        parts.append("        </div>")
        parts.append("    </div>")
        
        parts.append("    <div class='info-box'>")
        
        # Métriques clés
        parts.append("    <div class='metrics-grid'>")
        
        # Métrique 1: Données initiales
        if "Données" in selected_sections and isinstance(df, pd.DataFrame):
            parts.append(f"""
                <div class='metric-card'>
                    <div class='metric-icon'><i class='fas fa-database'></i></div>
                    <div class='metric-content'>
                        <span class='metric-label'>Données Initiales</span>
                        <span class='metric-value'>{df.shape[0]:,} × {df.shape[1]}</span>
                        <span class='metric-desc'>Lignes × Colonnes</span>
                    </div>
                </div>
            """)
        
        # Métrique 2: Données prétraitées
        if isinstance(df_prep, pd.DataFrame):
            parts.append(f"""
                <div class='metric-card'>
                    <div class='metric-icon'><i class='fas fa-tools'></i></div>
                    <div class='metric-content'>
                        <span class='metric-label'>Données Prétraitées</span>
                        <span class='metric-value'>{df_prep.shape[0]:,} × {df_prep.shape[1]}</span>
                        <span class='metric-desc'>Lignes × Colonnes</span>
                    </div>
                </div>
            """)
        
        # Métrique 3: Performance du modèle
        if "metric_value" in model_info and model_info["metric_value"] is not None:
            metric_name = model_info.get("metric_name", "Métrique")
            metric_value = model_info["metric_value"]
            
            # Déterminer la classe de couleur en fonction de la valeur
            if metric_value >= 0.9:
                metric_class = "success"
            elif metric_value >= 0.7:
                metric_class = "warning"
            else:
                metric_class = "danger"
            
            parts.append(f"""
                <div class='metric-card {metric_class}'>
                    <div class='metric-icon'><i class='fas fa-chart-line'></i></div>
                    <div class='metric-content'>
                        <span class='metric-label'>{metric_name.upper()}</span>
                        <span class='metric-value'>{metric_value:.4f}</span>
                        <span class='metric-desc'>Performance du modèle</span>
                    </div>
                </div>
            """)
        
        # Métrique 4: Taux de valeurs manquantes
        if isinstance(df, pd.DataFrame):
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            missing_class = "success" if missing_pct < 5 else "danger" if missing_pct > 20 else "warning"
            
            parts.append(f"""
                <div class='metric-card {missing_class}'>
                    <div class='metric-icon'><i class='fas fa-exclamation-triangle'></i></div>
                    <div class='metric-content'>
                        <span class='metric-label'>Valeurs Manquantes</span>
                        <span class='metric-value'>{missing_pct:.1f}%</span>
                        <span class='metric-desc'>Pourcentage global</span>
                    </div>
                </div>
            """)
        
        parts.append("    </div>")  # Fin de la grille de métriques
        
        # Résumé textuel
        parts.append("    <div class='summary-content'>")
        parts.append("        <h3><i class='fas fa-info-circle'></i> Contexte de l'Analyse</h3>")
        
        if framing.get("objective_desc"):
            parts.append(f"<p>{framing['objective_desc']}</p>")
        else:
            parts.append("""
                <p>Ce rapport présente les résultats de l'analyse de risque climatique réalisée avec l'outil d'analyse de données climatiques. 
                L'objectif est d'identifier et d'évaluer les risques liés aux changements climatiques pour une meilleure prise de décision.</p>
            """)
        
        # Points clés
        parts.append("        <div class='key-points'>")
        parts.append("            <h4><i class='fas fa-key'></i> Points Clés</h4>")
        parts.append("            <ul>")
        parts.append("                <li><i class='far fa-check-circle'></i> Analyse complète des données climatiques disponibles</li>")
        
        if model_info:
            model_name = model_info.get("model_name", "modèle")
            parts.append(f"                <li><i class='far fa-check-circle'></i> Développement d'un {model_name} pour la prédiction des risques</li>")
        
        parts.append("                <li><i class='far fa-check-circle'></i> Recommandations basées sur les résultats de l'analyse</li>")
        parts.append("            </ul>")
        parts.append("        </div>")
        
        parts.append("    </div>")  # Fin du contenu du résumé
        parts.append("    </div>")  # Fin de la boîte d'info
        parts.append("</section>")
        
        # Section 2: Données et sources
        if "Données" in selected_sections and (isinstance(df, pd.DataFrame) or data_sources):
            parts.append("<section id='data-sources' class='section-card'>")
            parts.append("    <div class='section-header'>")
            parts.append("        <h2><i class='fas fa-database'></i> Données et Sources</h2>")
            parts.append("    </div>")
            
            # Conteneur d'onglets
            parts.append("""
                <div class='tabs-container'>
                    <div class='tabs-header'>
                        <button class='tab-button active' onclick="openTab(event, 'data-overview')"><i class='fas fa-table'></i> Aperçu</button>
                        <button class='tab-button' onclick="openTab(event, 'data-sources-tab')"><i class='fas fa-folder-open'></i> Sources</button>
                        <button class='tab-button' onclick="openTab(event, 'data-quality')"><i class='fas fa-check-circle'></i> Qualité</button>
                    </div>
                    
                    <!-- Onglet Aperçu des données -->
                    <div id='data-overview' class='tab-content' style='display: block;'>""")
            
            # Aperçu des données
            if isinstance(df, pd.DataFrame):
                # Statistiques descriptives
                parts.append("<h3><i class='fas fa-chart-bar'></i> Statistiques Descriptives</h3>")
                
                # Graphique de distribution pour les colonnes numériques
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    try:
                        # Création d'un graphique de distribution
                        plt.figure(figsize=(10, 6))
                        for i, col in enumerate(num_cols[:5]):  # Limiter à 5 colonnes pour éviter la surcharge
                            sns.kdeplot(df[col].dropna(), label=col, linewidth=2)
                        
                        plt.title('Distribution des Variables Numériques')
                        plt.xlabel('Valeurs')
                        plt.ylabel('Densité')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # Convertir le graphique en base64
                        dist_plot = _img_to_base64(plt.gcf())
                        parts.append(dist_plot)
                    except Exception as e:
                        parts.append(f"<p class='warning'>Erreur lors de la génération du graphique de distribution: {str(e)}</p>")
                
                # Aperçu du DataFrame
                parts.append("<h3><i class='fas fa-table'></i> Aperçu des Données</h3>")
                parts.append("<div class='table-responsive'>")
                parts.append(df.head(10).to_html(classes='data-table', index=False))
                parts.append("</div>")
                
                # Types de données et valeurs manquantes
                parts.append("<div class='data-info-grid'>")
                
                # Types de données
                parts.append("<div class='data-info-card'>")
                parts.append("<h4><i class='fas fa-tags'></i> Types de Données</h4>")
                type_counts = df.dtypes.value_counts().reset_index()
                type_counts.columns = ['Type', 'Nombre de colonnes']
                parts.append(type_counts.to_html(classes='data-table', index=False))
                parts.append("</div>")
                
                # Valeurs manquantes
                missing = df.isnull().sum().reset_index()
                missing.columns = ['Colonne', 'Valeurs manquantes']
                missing['Pourcentage'] = (missing['Valeurs manquantes'] / len(df) * 100).round(2)
                
                parts.append("<div class='data-info-card'>")
                parts.append("<h4><i class='fas fa-exclamation-triangle'></i> Valeurs Manquantes</h4>")
                parts.append("<p>Total des valeurs manquantes: {:,} ({:.2f}%)</p>".format(
                    missing['Valeurs manquantes'].sum(), 
                    (missing['Valeurs manquantes'].sum() / (len(df) * len(df.columns)) * 100)
                ))
                
                # Afficher uniquement les colonnes avec des valeurs manquantes
                missing = missing[missing['Valeurs manquantes'] > 0]
                if not missing.empty:
                    parts.append("<p>Colonnes avec valeurs manquantes :</p>")
                    parts.append(missing.to_html(classes='data-table', index=False))
                else:
                    parts.append("<p class='success'><i class='fas fa-check-circle'></i> Aucune valeur manquante détectée</p>")
                parts.append("</div>")
                
                parts.append("</div>")  # Fin de data-info-grid
            
            # Fin de l'onglet Aperçu
            parts.append("</div>")  # Fin de data-overview
            
            # Onglet Sources de données
            parts.append("""
                <div id='data-sources-tab' class='tab-content'>
                    <h3><i class='fas fa-folder-open'></i> Sources de Données</h3>""")
            
            if data_sources:
                parts.append("<div class='sources-grid'>")
                for name, data in data_sources.items():
                    if isinstance(data, pd.DataFrame):
                        parts.append(f"""
                            <div class='source-card'>
                                <div class='source-header'>
                                    <i class='fas fa-database'></i>
                                    <h4>{name}</h4>
                                </div>
                                <div class='source-body'>
                                    <p><i class='fas fa-table'></i> {data.shape[0]:,} lignes × {data.shape[1]} colonnes</p>
                                    <p><i class='far fa-calendar-alt'></i> Généré le {datetime.now().strftime('%d/%m/%Y')}</p>
                                </div>
                                <div class='source-footer'>
                                    <button class='btn btn-sm btn-outline' onclick='showSourcePreview(\'{name}\')'><i class='fas fa-eye'></i> Aperçu</button>
                                </div>
                            </div>
                        """)
                parts.append("</div>")  # Fin de sources-grid
                
                # Ajouter des modaux pour l'aperçu des sources
                for name, data in data_sources.items():
                    if isinstance(data, pd.DataFrame):
                        parts.append(f"""
                            <div id='source-preview-{name}' class='modal'>
                                <div class='modal-content'>
                                    <span class='close' onclick="document.getElementById('source-preview-{name}').style.display='none'">&times;</span>
                                    <h3>{name}</h3>
                                    <div class='table-responsive'>
                                        {data.head(10).to_html(classes='data-table', index=False)}
                                    </div>
                                    <div class='modal-footer'>
                                        <button class='btn' onclick="document.getElementById('source-preview-{name}').style.display='none'">Fermer</button>
                                    </div>
                                </div>
                            </div>
                        """)
            else:
                parts.append("<p class='info'>Aucune source de données supplémentaire n'a été fournie.</p>")
            
            parts.append("</div>")  # Fin de data-sources-tab
            
            # Onglet Qualité des données
            parts.append("""
                <div id='data-quality' class='tab-content'>
                    <h3><i class='fas fa-check-circle'></i> Qualité des Données</h3>""")
            
            if isinstance(df, pd.DataFrame):
                # Score global de qualité
                quality_score = 100
                issues = []
                
                # Vérifier les valeurs manquantes
                missing_pct = (df.isnull().sum().sum() / (df.size)) * 100
                if missing_pct > 5:
                    quality_score -= 20
                    issues.append(f"Valeurs manquantes élevées ({missing_pct:.1f}%)")
                
                # Vérifier les doublons
                dup_rows = df.duplicated().sum()
                if dup_rows > 0:
                    dup_pct = (dup_rows / len(df)) * 100
                    quality_score -= 15
                    issues.append(f"{dup_rows} doublons détectés ({dup_pct:.1f}%)")
                
                # Vérifier les valeurs aberrantes (pour les colonnes numériques)
                if not df.select_dtypes(include=['number']).empty:
                    try:
                        Q1 = df.select_dtypes(include=['number']).quantile(0.25)
                        Q3 = df.select_dtypes(include=['number']).quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df.select_dtypes(include=['number']) < (Q1 - 1.5 * IQR)) | 
                                  (df.select_dtypes(include=['number']) > (Q3 + 1.5 * IQR))).sum().sum()
                        
                        if outliers > 0:
                            quality_score -= 10
                            issues.append(f"{outliers} valeurs aberrantes potentielles")
                    except:
                        pass
                
                # Afficher le score de qualité
                quality_class = "success" if quality_score >= 80 else "warning" if quality_score >= 60 else "danger"
                
                parts.append(f"""
                    <div class='quality-score {quality_class}'>
                        <div class='score-circle'>
                            <span class='score'>{quality_score}</span>
                            <span class='score-label'>/100</span>
                        </div>
                        <div class='score-details'>
                            <h4>Score de Qualité des Données</h4>
                            <p>Évaluation globale de la qualité des données</p>
                        </div>
                    </div>
                """)
                
                # Afficher les problèmes détectés
                if issues:
                    parts.append("<div class='quality-issues'>")
                    parts.append("<h4><i class='fas fa-exclamation-triangle'></i> Problèmes Détectés</h4>")
                    parts.append("<ul>")
                    for issue in issues:
                        parts.append(f"<li>{issue}</li>")
                    parts.append("</ul>")
                    parts.append("</div>")
                else:
                    parts.append("<div class='alert success'><i class='fas fa-check-circle'></i> Aucun problème majeur détecté dans les données.</div>")
                
                # Matrice de corrélation
                if len(df.select_dtypes(include=['number']).columns) > 1:
                    try:
                        plt.figure(figsize=(10, 8))
                        corr = df.select_dtypes(include=['number']).corr()
                        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                                  linewidths=0.5, linecolor='white', cbar=True)
                        plt.title('Matrice de Corrélation')
                        plt.tight_layout()
                        
                        corr_plot = _img_to_base64(plt.gcf())
                        parts.append("<h4>Matrice de Corrélation</h4>")
                        parts.append(corr_plot)
                    except Exception as e:
                        parts.append(f"<p class='warning'>Erreur lors de la génération de la matrice de corrélation: {str(e)}</p>")
            
            parts.append("</div>")  # Fin de data-quality
            
            # Script JavaScript pour les onglets
            parts.append("""
                <script>
                function openTab(evt, tabName) {
                    var i, tabcontent, tabbuttons;
                    
                    // Masquer tous les contenus d'onglets
                    tabcontent = document.getElementsByClassName('tab-content');
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = 'none';
                    }
                    
                    // Désactiver tous les boutons d'onglets
                    tabbuttons = document.getElementsByClassName('tab-button');
                    for (i = 0; i < tabbuttons.length; i++) {
                        tabbuttons[i].className = tabbuttons[i].className.replace(' active', '');
                    }
                    
                    // Afficher l'onglet actuel et marquer le bouton comme actif
                    document.getElementById(tabName).style.display = 'block';
                    evt.currentTarget.className += ' active';
                }
                
                function showSourcePreview(sourceName) {
                    var modal = document.getElementById('source-preview-' + sourceName);
                    modal.style.display = 'block';
                    
                    // Fermer la modale en cliquant en dehors
                    window.onclick = function(event) {
                        if (event.target == modal) {
                            modal.style.display = 'none';
                        }
                    }
                }
                </script>
            """)
            
            parts.append("</div>")  # Fin de tabs-container
            parts.append("</section>")  # Fin de la section Données et sources
        if "Données" in selected_sections:
            parts.append("<section id='data-sources' class='section-card'>")
            parts.append("    <div class='section-header'>")
            parts.append("        <h2><i class='fas fa-database'></i> Données et Sources</h2>")
            parts.append("    </div>")
            
            if data_sources:
                parts.append("    <div class='info-box'>")
                parts.append("        <h3><i class='fas fa-table'></i> Sources de Données Utilisées</h3>")
                
                # Tableau des sources de données
                parts.append("        <div class='table-responsive'>")
                parts.append("            <table class='data-table'>")
                parts.append("                <thead>")
                parts.append("                    <tr>")
                parts.append("                        <th>Source</th>")
                parts.append("                        <th>Lignes</th>")
                parts.append("                        <th>Colonnes</th>")
                parts.append("                        <th>Période</th>")
                parts.append("                        <th>Statut</th>")
                parts.append("                    </tr>")
                parts.append("                </thead>")
                parts.append("                <tbody>")
                
                for label, source_df in data_sources.items():
                    if isinstance(source_df, pd.DataFrame):
                        # Détecter les colonnes de date pour déterminer la période
                        date_cols = source_df.select_dtypes(include=['datetime64']).columns
                        period = "N/A"
                        
                        if len(date_cols) > 0:
                            min_date = source_df[date_cols[0]].min()
                            max_date = source_df[date_cols[0]].max()
                            period = f"{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}"
                        
                        parts.append(f"""
                            <tr>
                                <td><strong>{label}</strong></td>
                                <td>{source_df.shape[0]:,}</td>
                                <td>{source_df.shape[1]}</td>
                                <td>{period}</td>
                                <td><span class='status-badge success'><i class='fas fa-check-circle'></i> Chargé</span></td>
                            </tr>
                        """)
                
                parts.append("                </tbody>")
                parts.append("            </table>")
                parts.append("        </div>")
                
                # Aperçu des données brutes
                if isinstance(df, pd.DataFrame):
                    parts.append("        <h3><i class='fas fa-table'></i> Aperçu des Données Brutes</h3>")
                    parts.append("        <div class='tabs'>")
                    parts.append("            <div class='tab-header'>")
                    parts.append("                <div class='active' data-tab='head'>Aperçu</div>")
                    parts.append("                <div data-tab='info'>Résumé</div>")
                    parts.append("                <div data-tab='types'>Types</div>")
                    parts.append("                <div data-tab='missing'>Valeurs manquantes</div>")
                    parts.append("            </div>")
                    
                    # Onglet Aperçu
                    parts.append("            <div class='tab-content active' id='head'>")
                    parts.append(_wrap_table(df.head(10).style.set_table_attributes('class="dataframe"').render()))
                    parts.append("            </div>")
                    
                    # Onglet Résumé statistique
                    parts.append("            <div class='tab-content' id='info'>")
                    parts.append(_wrap_table(df.describe(include='all').round(2).style.set_table_attributes('class="dataframe"').render()))
                    parts.append("            </div>")
                    
                    # Onglet Types de données
                    parts.append("            <div class='tab-content' id='types'>")
                    dtype_df = pd.DataFrame({
                        'Colonne': df.dtypes.index,
                        'Type': df.dtypes.values,
                        'Valeurs uniques': df.nunique().values,
                        'Valeurs nulles': df.isnull().sum().values
                    })
                    parts.append(_wrap_table(dtype_df.style.set_table_attributes('class="dataframe"').render()))
                    parts.append("            </div>")
                    
                    # Onglet Valeurs manquantes
                    parts.append("            <div class='tab-content' id='missing'>")
                    missing_df = pd.DataFrame({
                        'Colonne': df.columns,
                        'Valeurs manquantes': df.isnull().sum().values,
                        '% manquant': (df.isnull().mean() * 100).round(2).astype(str) + '%'
                    }).sort_values('Valeurs manquantes', ascending=False)
                    
                    # Filtrer pour ne montrer que les colonnes avec des valeurs manquantes
                    missing_df = missing_df[missing_df['Valeurs manquantes'] > 0]
                    
                    if len(missing_df) > 0:
                        parts.append(_wrap_table(missing_df.style.set_table_attributes('class="dataframe"').render()))
                        
                        # Graphique des valeurs manquantes
                        try:
                            plt.figure(figsize=(10, 6))
                            sns.barplot(x=missing_df['% manquant'].str.rstrip('%').astype(float), 
                                       y=missing_df['Colonne'], 
                                       palette='viridis')
                            plt.title('Pourcentage de valeurs manquantes par colonne')
                            plt.xlabel('Pourcentage de valeurs manquantes (%)')
                            plt.ylabel('Colonne')
                            plt.tight_layout()
                            
                            parts.append(_img_to_base64(plt.gcf()))
                        except Exception as e:
                            parts.append(f"<p class='warning'>Erreur lors de la génération du graphique des valeurs manquantes: {str(e)}</p>")
                    else:
                        parts.append("<div class='success-box'><i class='fas fa-check-circle'></i> Aucune valeur manquante détectée dans les données.</div>")
                    
                    parts.append("            </div>")
                    parts.append("        </div>")  # Fin des onglets
                    
                    # Script pour la gestion des onglets
                    parts.append("""
                        <script>
                            document.addEventListener('DOMContentLoaded', function() {
                                const tabHeaders = document.querySelectorAll('.tab-header div');
                                const tabContents = document.querySelectorAll('.tab-content');
                                
                                tabHeaders.forEach(header => {
                                    header.addEventListener('click', () => {
                                        // Désactiver tous les onglets
                                        tabHeaders.forEach(h => h.classList.remove('active'));
                                        tabContents.forEach(c => c.classList.remove('active'));
                                        
                                        // Activer l'onglet sélectionné
                                        header.classList.add('active');
                                        const tabId = header.getAttribute('data-tab');
                                        document.getElementById(tabId).classList.add('active');
                                    });
                                });
                            });
                        </script>
                    """)
                    
                    # Visualisation des distributions
                    if include_plots:
                        parts.append("        <h3><i class='fas fa-chart-bar'></i> Distribution des Variables Numériques</h3>")
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if len(numeric_cols) > 0:
                            # Limiter à 6 variables pour éviter la surcharge
                            for i in range(0, min(len(numeric_cols), 6), 2):
                                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                                
                                # Graphique 1: Distribution avec histogramme et KDE
                                col1 = numeric_cols[i]
                                sns.histplot(df[col1].dropna(), kde=True, ax=axes[0], color='#4e79a7')
                                axes[0].set_title(f'Distribution de {col1}')
                                axes[0].set_xlabel('')
                                axes[0].grid(True, alpha=0.3)
                                
                                # Graphique 2: Boîte à moustaches
                                sns.boxplot(x=df[col1].dropna(), ax=axes[1], color='#f28e2b')
                                axes[1].set_title(f'Boîte à moustaches de {col1}')
                                axes[1].set_xlabel('')
                                
                                plt.tight_layout()
                                parts.append(_img_to_base64(fig))
                                
                                # Ajouter une analyse des valeurs aberrantes
                                q1 = df[col1].quantile(0.25)
                                q3 = df[col1].quantile(0.75)
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                outliers = df[(df[col1] < lower_bound) | (df[col1] > upper_bound)]
                                
                                if len(outliers) > 0:
                                    parts.append(f"""
                                        <div class='warning-box'>
                                            <h4><i class='fas fa-exclamation-triangle'></i> Valeurs aberrantes détectées pour {col1}</h4>
                                            <p>Plage normale attendue : [{lower_bound:.2f}, {upper_bound:.2f}]</p>
                                            <p>Nombre de valeurs aberrantes : {len(outliers):,} ({len(outliers)/len(df)*100:.1f}% des données)</p>
                                        </div>
                                    """)
                        
                        # Matrice de corrélation pour les variables numériques
                        if len(numeric_cols) > 1:
                            parts.append("        <h3><i class='fas fa-project-diagram'></i> Matrice de Corrélation</h3>")
                            
                            try:
                                # Calculer la matrice de corrélation
                                corr = df[numeric_cols].corr()
                                
                                # Créer un masque pour le triangle supérieur
                                mask = np.triu(np.ones_like(corr, dtype=bool))
                                
                                # Créer la figure
                                plt.figure(figsize=(12, 10))
                                
                                # Créer une palette de couleurs divergente
                                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                                
                                # Tracer la heatmap
                                sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                                           square=True, linewidths=.5, cbar_kws={"shrink": .8}, 
                                           annot=True, fmt=".2f")
                                
                                plt.title('Matrice de corrélation entre les variables numériques', pad=20)
                                plt.tight_layout()
                                
                                parts.append(_img_to_base64(plt.gcf()))
                                
                                # Identifier les paires de variables fortement corrélées
                                corr_pairs = []
                                for i in range(len(corr.columns)):
                                    for j in range(i):
                                        if abs(corr.iloc[i, j]) > 0.7:  # Seuil de corrélation élevée
                                            corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                                
                                if corr_pairs:
                                    parts.append("        <div class='info-box'>")
                                    parts.append("            <h4><i class='fas fa-link'></i> Corrélations fortes détectées</h4>")
                                    parts.append("            <ul>")
                                    
                                    for var1, var2, corr_value in corr_pairs:
                                        direction = "positive" if corr_value > 0 else "négative"
                                        strength = "très forte" if abs(corr_value) > 0.9 else "forte"
                                        parts.append(f"                <li><strong>{var1}</strong> et <strong>{var2}</strong>: corrélation {direction} {strength} ({corr_value:.2f})</li>")
                                    
                                    parts.append("            </ul>")
                                    parts.append("        </div>")
                                
                            except Exception as e:
                                parts.append(f"<p class='warning'>Erreur lors de la génération de la matrice de corrélation: {str(e)}</p>")
                
                parts.append("    </div>")  # Fin de la boîte d'info
            else:
                parts.append("    <div class='warning-box'>")
                parts.append("        <p><i class='fas fa-exclamation-triangle'></i> Aucune source de données n'a été spécifiée.</p>")
                parts.append("    </div>")
            
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
