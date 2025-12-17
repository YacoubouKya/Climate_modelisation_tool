"""Reporting pour Data Tool Climatique.

Résumé dans l’interface Streamlit + génération d’un rapport HTML inspiré
du reporting principal (sections structurées et CSS moderne).
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st


def _get_climate_report_css() -> str:
    """CSS moderne inspiré du reporting principal."""

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
            font-size: 2.2em;
            margin-bottom: 10px;
            border-bottom: 3px solid #facc15;
            padding-bottom: 12px;
            text-align: center;
        }

        h2 {
            color: #facc15;
            font-size: 1.6em;
            margin-top: 32px;
            margin-bottom: 14px;
        }

        h3 {
            color: #e5e7eb;
            font-size: 1.2em;
            margin-top: 18px;
            margin-bottom: 10px;
        }

        p {
            margin: 8px 0;
            font-size: 0.95em;
        }

        .metric-box {
            display: inline-block;
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            padding: 10px 18px;
            margin: 8px 10px 8px 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            font-weight: bold;
            font-size: 0.9em;
        }

        .metric-label {
            font-size: 0.8em;
            opacity: 0.9;
            display: block;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 1.4em;
            display: block;
        }

        .table-container {
            width: 100%;
            overflow-x: auto;
            margin: 14px 0;
            border-radius: 8px;
            border: 1px solid #1f2937;
            background: #020617;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: #020617;
            color: #e5e7eb;
            font-size: 0.85em;
            min-width: 500px;
        }

        thead {
            background: #111827;
        }

        th, td {
            padding: 6px 8px;
            border-bottom: 1px solid #1f2937;
            white-space: nowrap;
        }

        th {
            text-align: left;
            font-weight: 600;
            color: #facc15;
        }

        tbody tr:nth-child(even) {
            background: #020617;
        }

        .info-box {
            background: #0f172a;
            border-left: 4px solid #38bdf8;
            padding: 12px;
            margin: 14px 0;
            border-radius: 6px;
        }

        .warning-box {
            background: #451a03;
            border-left: 4px solid #f97316;
            padding: 12px;
            margin: 14px 0;
            border-radius: 6px;
        }

        ul {
            list-style: none;
            padding-left: 0;
        }

        ul li {
            padding: 6px 0;
            padding-left: 20px;
            position: relative;
        }

        ul li:before {
            content: "▸";
            position: absolute;
            left: 0;
            color: #38bdf8;
        }

        .footer {
            margin-top: 32px;
            padding-top: 12px;
            border-top: 1px solid #1f2937;
            text-align: center;
            color: #9ca3af;
            font-size: 0.85em;
        }
    </style>
    """


def _wrap_table(html: str) -> str:
    return f'<div class="table-container">{html}</div>'


def show_reporting_summary(session_state) -> None:
    """Affiche un résumé textuel du projet climat en cours.

    Cette fonction est volontairement simple :
    - rappelle les sources de données chargées,
    - affiche la cible et le type de tâche,
    - rappelle la métrique principale obtenue.
    """

    st.subheader("📝 Synthèse du projet climat")

    if "clim_data" in session_state:
        df = session_state["clim_data"]
        st.markdown(f"- **Données initiales** : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    if "clim_prep_info" in session_state:
        info = session_state["clim_prep_info"]
        freq = info.get("freq", "Aucune")
        shape = info.get("shape")
        st.markdown("- **Prétraitement** :")
        st.markdown(f"  - fréquence d’agrégation : `{freq}`")
        if shape:
            st.markdown(f"  - shape après prétraitement : {shape[0]} lignes × {shape[1]} colonnes")

    if "clim_model_info" in session_state:
        minfo = session_state["clim_model_info"]
        metric_name = minfo.get("metric_name")
        metric_value = minfo.get("metric_value")
        task_type = minfo.get("task_type")
        st.markdown("- **Modélisation** :")
        st.markdown(f"  - type de tâche : `{task_type}`")
        if metric_name and metric_value is not None:
            st.markdown(f"  - {metric_name} : **{metric_value:.4f}**")

    # Génération d’un mini-rapport HTML basique
    if st.button("Générer un rapport HTML climat"):
        path = generate_html_report(session_state)
        if path:
            st.success(f"Rapport HTML généré : {os.path.basename(path)}")
            with open(path, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le rapport HTML climat",
                    data=f,
                    file_name=os.path.basename(path),
                    mime="text/html",
                )


def generate_html_report(session_state) -> str | None:
    """Génère un rapport HTML climat structuré (cadrage, données, prétraitement, modèle, limites)."""

    df = session_state.get("clim_data")
    df_prep = session_state.get("clim_data_prep")
    prep_info = session_state.get("clim_prep_info", {})
    model_info = session_state.get("clim_model_info", {})
    framing = session_state.get("project_framing", {})
    data_sources = session_state.get("data_sources", {})

    out_dir = os.path.join("outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"rapport_climat_{now}.html"
    out_path = os.path.join(out_dir, filename)

    parts: list[str] = []
    parts.append("<html><head><meta charset='utf-8'><title>Rapport Climat - Analyse de Risque</title>")
    parts.append(_get_climate_report_css())
    parts.append("</head><body>")
    parts.append("<div class='container'>")
    parts.append("<h1>🌍 Rapport Climat – Hackathon</h1>")

    # Bloc d'infos générales
    parts.append("<div class='info-box'>")
    parts.append(
        f"<p><strong>📅 Date de génération :</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>"
    )
    if isinstance(df, pd.DataFrame):
        parts.append(
            f"<p><strong>📁 Données initiales :</strong> {df.shape[0]} lignes × {df.shape[1]} colonnes</p>"
        )
    if isinstance(df_prep, pd.DataFrame):
        parts.append(
            f"<p><strong>🛠️ Données prétraitées :</strong> {df_prep.shape[0]} lignes × {df_prep.shape[1]} colonnes</p>"
        )
    parts.append("</div>")

    # 0. Cadrage du projet (si disponible)
    if framing:
        parts.append("<h2>🎯 Cadrage du Projet</h2>")
        parts.append("<div class='info-box'>")
        if framing.get("objective_type"):
            parts.append(f"<p><strong>Type d'objectif :</strong> {framing['objective_type']}</p>")
        if framing.get("objective_desc"):
            parts.append(f"<p><strong>Description :</strong> {framing['objective_desc']}</p>")
        if framing.get("unit_of_analysis"):
            parts.append(f"<p><strong>Unité d'analyse :</strong> {framing['unit_of_analysis']}</p>")
        if framing.get("target_desc"):
            parts.append(f"<p><strong>Cible attendue :</strong> {framing['target_desc']}</p>")
        if framing.get("context"):
            parts.append(f"<p><strong>Contexte :</strong> {framing['context']}</p>")
        parts.append("</div>")

    # 0.5 Sources de données
    if data_sources:
        parts.append("<h2>📂 Sources de Données</h2>")
        parts.append("<ul>")
        for label, source_df in data_sources.items():
            parts.append(f"<li><strong>{label}</strong> : {source_df.shape[0]} lignes × {source_df.shape[1]} colonnes</li>")
        parts.append("</ul>")

    # 1. Données
    if isinstance(df, pd.DataFrame):
        parts.append("<h2>1. Données initiales</h2>")
        parts.append("<h3>📋 Aperçu (5 premières lignes)</h3>")
        parts.append(_wrap_table(df.head().to_html(index=False, classes='dataframe')))

    # 2. Prétraitement
    if isinstance(df_prep, pd.DataFrame) or prep_info:
        parts.append("<h2>2. Prétraitement climatique</h2>")
        if prep_info:
            parts.append("<ul>")
            if "date_col" in prep_info:
                parts.append(
                    f"<li>Colonne date utilisée : <code>{prep_info['date_col']}</code></li>"
                )
            if "freq" in prep_info:
                parts.append(
                    f"<li>Fréquence d’agrégation : <code>{prep_info['freq']}</code></li>"
                )
            if prep_info.get("rolling"):
                parts.append("<li>Features temporelles (rolling) activées</li>")
            if prep_info.get("anomaly_summary"):
                parts.append("<li>Résumé d’anomalies (z-score) calculé</li>")
            parts.append("</ul>")

        if isinstance(df_prep, pd.DataFrame):
            parts.append("<h3>📋 Aperçu des données prétraitées</h3>")
            parts.append(_wrap_table(df_prep.head().to_html(index=False, classes='dataframe')))

    # 3. Modèle de risque
    if model_info:
        parts.append("<h2>3. Modèle de risque climatique</h2>")
        ttype = model_info.get("task_type")
        mname = model_info.get("model_name")
        metric_name = model_info.get("metric_name")
        metric_value = model_info.get("metric_value")
        used_stratify = model_info.get("used_stratify")
        f1_score = model_info.get("f1_score")
        cv_scores = model_info.get("cv_scores")
        handle_imbalance = model_info.get("handle_imbalance")

        parts.append("<div class='info-box'>")
        parts.append(f"<p><strong>🎯 Type de tâche :</strong> {ttype}</p>")
        parts.append(f"<p><strong>🤖 Modèle :</strong> {mname}</p>")
        if handle_imbalance:
            parts.append("<p><strong>⚖️ Gestion du déséquilibre :</strong> Activée (class_weight='balanced')</p>")
        if metric_name and metric_value is not None:
            parts.append(
                "<div class='metric-box'>"
                f"<span class='metric-label'>{metric_name.upper()}</span>"
                f"<span class='metric-value'>{metric_value:.4f}</span>"
                "</div>"
            )
        if f1_score is not None:
            parts.append(
                "<div class='metric-box'>"
                f"<span class='metric-label'>F1-SCORE</span>"
                f"<span class='metric-value'>{f1_score:.4f}</span>"
                "</div>"
            )
        if cv_scores is not None:
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            parts.append(f"<p><strong>📊 Validation temporelle :</strong> {cv_mean:.4f} ± {cv_std:.4f}</p>")
        if used_stratify is False and ttype == "classification":
            parts.append(
                "<div class='warning-box'><p>Stratification désactivée au split train/test "
                "(classes trop déséquilibrées pour un split stratifié).</p></div>"
            )
        parts.append("</div>")
        
        # Feature importance
        feat_imp = model_info.get("feature_importance")
        feat_names = model_info.get("feature_names")
        if feat_imp is not None and feat_names is not None:
            parts.append("<h3>📊 Feature Importance (Top 10)</h3>")
            imp_df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": feat_imp
            }).sort_values("Importance", ascending=False).head(10)
            parts.append(_wrap_table(imp_df.to_html(index=False, classes='dataframe')))
    
    # 4. Limites et recommandations
    parts.append("<h2>⚠️ Limites et Recommandations</h2>")
    parts.append("<div class='warning-box'>")
    parts.append("<h3>Limites du modèle</h3>")
    parts.append("<ul>")
    parts.append("<li>Ce modèle a été développé dans un contexte de hackathon/prototype rapide</li>")
    parts.append("<li>La validation croisée temporelle est recommandée pour les données climatiques</li>")
    parts.append("<li>Les données manquantes et les outliers doivent être analysés en détail</li>")
    parts.append("<li>La période de référence climatologique doit être adaptée au contexte</li>")
    parts.append("</ul>")
    parts.append("<h3>Recommandations</h3>")
    parts.append("<ul>")
    parts.append("<li>Valider le modèle sur des données externes ou une période future</li>")
    parts.append("<li>Enrichir avec des données d'exposition et de vulnérabilité</li>")
    parts.append("<li>Intégrer des scénarios climatiques futurs (RCP, SSP)</li>")
    parts.append("<li>Documenter les hypothèses et les sources de données</li>")
    parts.append("<li>Consulter des experts métier pour valider les résultats</li>")
    parts.append("</ul>")
    parts.append("</div>")

    # Footer
    parts.append("<div class='footer'>")
    parts.append("<p>Rapport généré automatiquement par Data Tool Climatique.</p>")
    parts.append("</div>")

    parts.append("</div></body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    return out_path
