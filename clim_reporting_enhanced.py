"""
Module de reporting avancé pour l'application Climate Risk Tool.
Génère des rapports HTML professionnels avec visualisations et analyses des risques climatiques.
"""

import os
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple

# Configuration des dossiers de sortie
OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# Fonctions utilitaires
# ============================================

def _get_plotly_figure_html(fig: go.Figure, width: int = 800, height: int = 500) -> str:
    """Convertit une figure Plotly en HTML."""
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True})

def _get_css_styles() -> str:
    """Retourne le CSS personnalisé pour le rapport."""
    return """
    <style>
        :root {
            --primary-color: #3b82f6;
            --secondary-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
            --light-bg: #f8fafc;
            --dark-bg: #0f172a;
            --text-color: #1e293b;
            --text-light: #64748b;
            --border-color: #e2e8f0;
            --success-color: #10b981;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: #f1f5f9;
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 0.5rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .header h1 {
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            color: var(--text-light);
            font-size: 1.1rem;
        }
        
        .section {
            margin: 2rem 0;
            padding: 1.5rem;
            background: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .section-title {
            color: var(--primary-color);
            margin-top: 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .kpi-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.2s;
            border-left: 4px solid var(--primary-color);
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: 0.5rem 0;
        }
        
        .kpi-label {
            color: var(--text-light);
            font-size: 0.9rem;
        }
        
        .warning {
            background-color: #fffbeb;
            border-left: 4px solid var(--warning-color);
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .warning svg {
            color: var(--warning-color);
            flex-shrink: 0;
        }
        
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
            margin: 2rem 0;
        }
        
        @media (max-width: 1200px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }
        
        .plot-container {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .plot-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0,0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .plot-container h3 {
            margin-top: 0;
            color: var(--text-color);
            font-size: 1.1rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        /* Styles pour les tableaux */
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border-radius: 0.5rem;
            overflow: hidden;
        }
        
        .dataframe th, .dataframe td {
            padding: 0.75rem 1rem;
            text-align: left;
            border: 1px solid var(--border-color);
        }
        
        .dataframe th {
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }
        
        .dataframe tr:nth-child(even) {
            background-color: #f8fafc;
        }
        
        .dataframe tr:hover {
            background-color: #f1f5f9;
        }
        
        /* Styles pour les sections spéciales */
        .model-section {
            background-color: #f0f9ff;
            border-left: 4px solid var(--info-color);
        }
        
        .metrics-section {
            background-color: #f0fdf4;
            border-left: 4px solid var(--success-color);
        }
        
        .trends-section {
            background-color: #fffbeb;
            border-left: 4px solid var(--warning-color);
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
            color: var(--text-light);
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .kpi-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """

def _create_kpi_card(value: Any, label: str, icon: str = "📊", color: str = "var(--primary-color)") -> str:
    """Crée une carte KPI pour le rapport."""
    return f"""
    <div class="kpi-card" style="border-left-color: {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """

# ============================================
# Fonctions d'analyse des données
# ============================================

def _analyze_climate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse les données climatiques et retourne des métriques clés."""
    analysis = {}
    
    # Vérifier les colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Détection des colonnes de température, précipitations, etc.
    temp_cols = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'tmax', 'tmin', 'tavg'])]
    precip_cols = [col for col in df.columns if any(term in col.lower() for term in ['precip', 'rain', 'pluie'])]
    
    # Calcul des métriques de base
    analysis['num_rows'] = len(df)
    analysis['num_cols'] = len(df.columns)
    analysis['missing_values'] = df.isna().sum().sum()
    analysis['missing_percent'] = (analysis['missing_values'] / (len(df) * len(df.columns)) * 100).round(2)
    
    # Statistiques sur les températures
    if temp_cols:
        temp_df = df[temp_cols].select_dtypes(include=['number'])
        if not temp_df.empty:
            analysis['avg_temp'] = temp_df.mean().mean().round(1)
            analysis['temp_range'] = (temp_df.max().max() - temp_df.min().min()).round(1)
    
    # Statistiques sur les précipitations
    if precip_cols:
        precip_df = df[precip_cols].select_dtypes(include=['number'])
        if not precip_df.empty:
            analysis['avg_precip'] = precip_df.mean().mean().round(1)
            analysis['max_precip'] = precip_df.max().max().round(1)
    
    # Détection des valeurs aberrantes
    if numeric_cols:
        numeric_df = df[numeric_cols]
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric_df < (q1 - 1.5 * iqr)) | (numeric_df > (q3 + 1.5 * iqr))).sum().sum()
        analysis['outliers'] = outliers
    
    return analysis

def _create_temperature_plot(df: pd.DataFrame, temp_cols: List[str]) -> Optional[go.Figure]:
    """Crée un graphique d'évolution des températures."""
    if not temp_cols:
        return None
        
    # Sélectionner uniquement les colonnes de température numériques
    temp_df = df[temp_cols].select_dtypes(include=['number'])
    if temp_df.empty:
        return None
    
    # Préparer les données pour le tracé
    x_values = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
    
    # Créer un graphique d'évolution
    fig = go.Figure()
    
    for col in temp_df.columns:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=temp_df[col].values,
            name=col,
            mode='lines+markers',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Évolution des températures",
        xaxis_title="Date" if isinstance(df.index, pd.DatetimeIndex) else "Index",
        yaxis_title="Température (°C)",
        legend_title="Légende",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def _create_precipitation_plot(df: pd.DataFrame, precip_cols: List[str]) -> Optional[go.Figure]:
    """Crée un graphique des précipitations."""
    if not precip_cols:
        return None
        
    # Sélectionner uniquement les colonnes de précipitations numériques
    precip_df = df[precip_cols].select_dtypes(include=['number'])
    if precip_df.empty:
        return None
        
    # Créer un graphique à barres empilées
    fig = go.Figure()
    
    for col in precip_df.columns:
        x_values = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
        fig.add_trace(go.Bar(
            x=x_values,
            y=precip_df[col],
            name=col
        ))
    
    fig.update_layout(
        title="Précipitations",
        xaxis_title="Date" if isinstance(df.index, pd.DatetimeIndex) else "Index",
        yaxis_title="Précipitations (mm)",
        barmode='stack',
        legend_title="Légende",
        template="plotly_white"
    )
    
    return fig

# ============================================
# Fonction principale de génération de rapport
# ============================================

def generate_climate_report(session_state: Dict[str, Any], report_type: str = "complet") -> str:
    """
    Génère un rapport HTML complet sur les données climatiques.
    
    Args:
        session_state: État de la session Streamlit
        report_type: Type de rapport ('complet', 'executif', 'technique')
        
    Returns:
        str: Contenu HTML du rapport
    """
    # Vérifier si des données sont disponibles
    if 'df' not in session_state:
        return "<div class='error'>Aucune donnée disponible pour générer le rapport.</div>"
    
    df = session_state['df']
    
    # Analyser les données
    analysis = _analyze_climate_data(df)
    
    # Détecter les colonnes de température et précipitations
    temp_cols = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'tmax', 'tmin', 'tavg'])]
    precip_cols = [col for col in df.columns if any(term in col.lower() for term in ['precip', 'rain', 'pluie'])]
    
    # Détecter les colonnes de localisation
    loc_cols = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'lon', 'long', 'latitude', 'longitude'])]
    
    # Détecter les colonnes de date
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    
    # Créer le contenu HTML
    html_parts = []
    
    # En-tête du document
    report_date = datetime.now().strftime("%d/%m/%Y à %H:%M")
    html_parts.append(f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'Analyse Climatique</title>
        {_get_css_styles()}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌍 Rapport d'Analyse Climatique</h1>
                <div class="subtitle">Généré le {report_date}</div>
            </div>
    """)
    
    # Section de résumé exécutif
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">📊 Résumé Exécutif</h2>
        <p>Ce rapport présente une analyse complète des données climatiques chargées dans l'application.</p>
        
        <div class="kpi-container">
    """)
    
    # Ajouter les KPIs
    html_parts.append(_create_kpi_card(
        value=f"{analysis.get('num_rows', 0):,}",
        label="Observations",
        icon="📈"
    ))
    
    html_parts.append(_create_kpi_card(
        value=analysis.get('num_cols', 0),
        label="Variables",
        icon="📋"
    ))
    
    if 'avg_temp' in analysis:
        html_parts.append(_create_kpi_card(
            value=f"{analysis['avg_temp']}°C",
            label="Température moyenne",
            icon="🌡️",
            color="var(--danger-color)"
        ))
    
    if 'avg_precip' in analysis:
        html_parts.append(_create_kpi_card(
            value=f"{analysis['avg_precip']} mm",
            label="Précipitations moyennes",
            icon="🌧️",
            color="var(--primary-color)"
        ))
    
    html_parts.append("</div>")
    
    # Avertissements
    if analysis.get('missing_values', 0) > 0:
        html_parts.append(f"""
        <div class="warning">
            <span>⚠️</span>
            <div>
                <strong>Attention :</strong> {analysis['missing_values']} valeurs manquantes détectées 
                ({analysis['missing_percent']}% des données).
            </div>
        </div>
        """)
    
    if analysis.get('outliers', 0) > 0:
        html_parts.append(f"""
        <div class="warning">
            <span>⚠️</span>
            <div>
                <strong>Attention :</strong> {analysis['outliers']} valeurs aberrantes détectées 
                (en dehors de l'intervalle interquartile).
            </div>
        </div>
        """)
    
    html_parts.append("</div>")  # Fin de la section Résumé Exécutif
    
    # Section d'analyse détaillée
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">🔍 Analyse Détailée</h2>
        <div class="grid-container">
    """)
    
    # Graphique des températures
    temp_fig = _create_temperature_plot(df, temp_cols)
    if temp_fig:
        temp_html = _get_plotly_figure_html(temp_fig)
        html_parts.append(f"""
        <div class="plot-container">
            <h3>📈 Évolution des Températures</h3>
            {temp_html}
            <p class="text-muted">Évolution temporelle des températures enregistrées. Utilisez les contrôles pour zoomer et explorer les données.</p>
        </div>
        """)
    
    # Graphique des précipitations
    precip_fig = _create_precipitation_plot(df, precip_cols)
    if precip_fig:
        precip_html = _get_plotly_figure_html(precip_fig)
        html_parts.append(f"""
        <div class="plot-container">
            <h3>🌧️ Précipitations</h3>
            {precip_html}
            <p class="text-muted">Distribution et évolution des précipitations. Les barres empilées montrent les différents types de précipitations.</p>
        </div>
        """)
    
    html_parts.append("</div></div>")  # Fin de la grille et de la section Analyse Détailée
    
    # Section des statistiques descriptives
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">📊 Statistiques Descriptives</h2>
        <div class="grid-2">
    """)
    
    # Aperçu des données
    html_parts.append("""
    <div>
        <h3>Aperçu des Données</h3>
        <div class="table-container">
    """)
    html_parts.append(df.head().to_html(classes='dataframe', index=False))
    html_parts.append("</div></div>")
    
    # Statistiques descriptives
    if not df.select_dtypes(include=['number']).empty:
        html_parts.append("""
        <div>
            <h3>Statistiques Numériques</h3>
            <div class="table-container">
        """)
        html_parts.append(df.describe().round(2).to_html(classes='dataframe'))
        html_parts.append("</div></div>")
    
    html_parts.append("</div>")  # Fin de la grille
    
    # Section d'analyse des tendances
    html_parts.append("""
    <div class="section trends-section">
        <h2 class="section-title">📈 Analyse des Tendances</h2>
        <p>Cette section présente les tendances temporelles et les modèles identifiés dans les données climatiques.</p>
        <div class="grid-container">
    """)
    
    # Ici, vous pouvez ajouter des graphiques de tendance ou d'autres analyses
    if date_cols and temp_cols:
        # Exemple de graphique de tendance des températures
        try:
            temp_trend_fig = px.scatter(
                df, 
                x=date_cols[0], 
                y=temp_cols[0],
                trendline="lowess",
                title=f"Tendance des {temp_cols[0]}"
            )
            temp_trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title=temp_cols[0],
                template="plotly_white"
            )
            html_parts.append(f"""
            <div class="plot-container">
                <h3>Tendance des Températures</h3>
                {_get_plotly_figure_html(temp_trend_fig)}
                <p class="text-muted">Courbe de tendance lissée avec la méthode LOWESS</p>
            </div>
            """)
        except Exception as e:
            st.warning(f"Impossible de générer le graphique de tendance : {str(e)}")
    
    if date_cols and precip_cols:
        # Exemple de graphique de tendance des précipitations
        try:
            precip_trend_fig = px.bar(
                df, 
                x=date_cols[0], 
                y=precip_cols[0],
                title=f"Tendance des {precip_cols[0]}"
            )
            precip_trend_fig.update_layout(
                xaxis_title="Date",
                yaxis_title=precip_cols[0],
                template="plotly_white"
            )
            html_parts.append(f"""
            <div class="plot-container">
                <h3>Tendance des Précipitations</h3>
                {_get_plotly_figure_html(precip_trend_fig)}
                <p class="text-muted">Évolution temporelle des précipitations</p>
            </div>
            """)
        except Exception as e:
            st.warning(f"Impossible de générer le graphique de tendance : {str(e)}")
    
    html_parts.append("</div></div>")  # Fin de la section des tendances
    
    # Section de modélisation
    html_parts.append("""
    <div class="section model-section">
        <h2 class="section-title">🤖 Modélisation</h2>
        <p>Cette section présente les résultats des modèles appliqués aux données climatiques.</p>
        <div class="grid-2">
    """)
    
    # Ici, vous pouvez ajouter des visualisations de modèles
    html_parts.append("""
    <div class="plot-container">
        <h3>Modèle de Prédiction</h3>
        <p>Les fonctionnalités de modélisation avancée seront disponibles dans une prochaine version.</p>
    </div>
    
    <div class="plot-container">
        <h3>Importance des Variables</h3>
        <p>L'analyse d'importance des variables sera disponible après l'application d'un modèle prédictif.</p>
    </div>
    """)
    
    html_parts.append("</div></div>")  # Fin de la section de modélisation
    
    # Section des métriques avancées
    html_parts.append("""
    <div class="section metrics-section">
        <h2 class="section-title">📐 Métriques Avancées</h2>
        <p>Cette section présente des indicateurs avancés calculés à partir des données climatiques.</p>
        <div class="kpi-container">
    """)
    
    # Ajout de métriques avancées
    if 'avg_temp' in analysis and 'min_temp' in analysis and 'max_temp' in analysis:
        temp_range = analysis['max_temp'] - analysis['min_temp']
        html_parts.append(_create_kpi_card(
            value=f"{temp_range:.1f}°C",
            label="Amplitude thermique",
            icon="🌡️",
            color="var(--danger-color)"
        ))
    
    if 'avg_precip' in analysis and 'max_precip' in analysis:
        html_parts.append(_create_kpi_card(
            value=f"{analysis['max_precip']} mm",
            label="Précipitations max. journalières",
            icon="💧",
            color="var(--primary-color)"
        ))
    
    # Ajoutez d'autres métriques selon les données disponibles
    html_parts.append(_create_kpi_card(
        value=f"{analysis.get('num_rows', 0):,}",
        label="Jours d'observation",
        icon="📅",
        color="var(--secondary-color)"
    ))
    
    html_parts.append("</div></div>")  # Fin de la section des métriques
    
    # Section d'informations sur les données
    html_parts.append("""
    <div class="section">
        <h2 class="section-title">ℹ️ Informations sur les Données</h2>
        <h3>Types de Données</h3>
    """)
    
    # Informations sur les types de données
    type_info = pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes.astype(str),
        'Valeurs uniques': df.nunique(),
        'Valeurs manquantes': df.isna().sum(),
        '% Manquantes': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%'
    })
    html_parts.append("""
    <div class="table-container">
        <style>
            .dataframe .highlight {
                font-weight: bold;
                color: var(--danger-color);
            }
        </style>
    """)
    
    # Appliquer un style pour les valeurs manquantes
    def highlight_missing(val):
        if '%' in str(val):
            pct = float(str(val).replace('%', ''))
            if pct > 10:  # Mettre en évidence les colonnes avec plus de 10% de valeurs manquantes
                return 'highlight'
        return ''
    
    # Convertir le DataFrame en HTML avec mise en forme
    type_info_html = type_info.style.applymap(highlight_missing).to_html(classes='dataframe', index=False)
    html_parts.append(type_info_html)
    html_parts.append("</div>")
    
    html_parts.append("</div>")  # Fin de la section Statistiques
    
    # Pied de page
    html_parts.append(f"""
    <div class="footer">
        <p>Rapport généré par Climate Risk Tool • {report_date}</p>
    </div>
    </div> <!-- Fin du container -->
    </body>
    </html>
    """)
    
    # Combiner toutes les parties du HTML
    return "\n".join(html_parts)

def show_reporting_ui():
    """Affiche l'interface utilisateur pour la génération de rapports."""
    st.title("📊 Reporting Climat")
    
    # Vérifier si des données sont disponibles
    if 'df' not in st.session_state:
        st.warning("Veuvez d'abord charger des données dans l'onglet 'Chargement'.")
        return
    
    # Options du rapport
    st.sidebar.header("Options du Rapport")
    report_type = st.sidebar.selectbox(
        "Type de rapport",
        ["Complet", "Exécutif", "Technique"],
        index=0
    )
    
    include_plots = st.sidebar.checkbox("Inclure les graphiques", value=True)
    
    # Bouton de génération
    if st.sidebar.button("🔄 Générer le Rapport", type="primary"):
        with st.spinner("Génération du rapport en cours..."):
            try:
                # Générer le rapport HTML
                html_content = generate_climate_report(
                    st.session_state, 
                    report_type=report_type.lower()
                )
                
                # Afficher un aperçu du rapport
                st.subheader("Aperçu du Rapport")
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger le Rapport HTML",
                    data=html_content,
                    file_name=f"rapport_climat_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"Erreur lors de la génération du rapport : {str(e)}")
                st.exception(e)
    else:
        # Afficher uniquement les informations de base sur les données
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nombre de lignes", f"{len(st.session_state['df']):,}")
        with col2:
            st.metric("Nombre de colonnes", len(st.session_state['df'].columns))
        
        # Conseils pour l'utilisateur
        st.info("ℹ️ Utilisez le panneau latéral pour générer un rapport personnalisé.")

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Exemple d'utilisation
    import pandas as pd
    import numpy as np
    
    # Créer des données de démonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    data = {
        'date': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 2, 365),
        'precipitation': np.random.gamma(shape=2, scale=5, size=365).clip(0, 50),
        'humidite': 60 + 20 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 5, 365),
        'vent_vitesse': np.random.weibull(2, 365) * 10,
        'ville': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Bordeaux'], 365)
    }
    
    df = pd.DataFrame(data)
    st.session_state['df'] = df
    
    # Afficher l'interface
    show_reporting_ui()
