"""Application Streamlit principale pour Data Tool Climatique.

Flux fonctionnel :
- Chargement des données climatiques / exposition
- EDA rapide
- Prétraitement de base (dates, agrégation, rolling, résumé d'anomalies)
- Modélisation (plusieurs modèles au choix)
- Évaluation
- Cartographie du risque
- Reporting (synthèse + mini-rapport HTML)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

# Imports des modules avec gestion d'erreur
try:
    import clim_data_loader
    import clim_preprocessing
    import clim_modeling
    import clim_evaluation
    from clim_reporting_updated import show_reporting_summary, generate_html_report
    import clim_maps
    import clim_model_comparison
    from clim_data_utils import merge_dataframes
except ImportError as e:
    st.error(f"❌ Erreur d'import des modules : {e}")
    st.stop()


st.set_page_config(
    page_title="Data Tool Climatique",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optimisations de performance
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True


@st.cache_resource
def _inject_custom_css() -> None:
    """Applique le même thème que l'app principale Data Project Tool."""

    st.markdown(
        """
        <style>

        /********* HEADER *********/
        .custom-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 60px;
            background-color: #1E3A5F;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 40px;
            z-index: 9999;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.3);
        }
        .custom-header .logo { font-size: 22px; font-weight: bold; color: #FFD700; }
        .custom-header .menu { display: flex; gap: 20px; }
        .custom-header .menu a { color: white; text-decoration: none; font-weight: 500; font-family: 'Segoe UI', sans-serif; transition: color 0.3s; }
        .custom-header .menu a:hover { color: #FFD700; }

        .block-container { padding-top: 80px !important; }
        .stApp { background-color: #1E3A5F; }
        .block-container, .st-emotion-cache-18e3th9, .st-emotion-cache-1y4p8pa { background-color: transparent !important; }

        /********* TITRES *********/
        h1, h2, h3, h4 { color: #FFD700; font-family: 'Segoe UI', sans-serif; }

        /********* TEXTE GLOBAL *********/
        .block-container p,
        .block-container span,
        .block-container label,
        .block-container div:not([data-testid="stFileUploader"]):not(.stSelectbox):not([role="radiogroup"]) {
            color: #FFFFFF !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /********* SIDEBAR *********/
        [data-testid="stSidebar"] { background-color: #1569C7 !important; color: yellow !important; }
        [data-testid="stSidebar"] h1, h2, h3, label { color: yellow !important; }

        /********* BOUTONS *********/
        .stButton>button { background-color: #FFD700; color: #1E3A5F; border-radius: 10px; padding: 10px 20px; border: none; font-weight: bold; }
        .stButton>button:hover { background-color: #FFA500; color: white; }

        /********* FILE UPLOADER *********/
        [data-testid="stFileUploader"] {
            background-color: #FFD700 !important;
            border-radius: 10px;
            padding: 10px;
        }

        [data-testid="stFileUploader"] * {
            color: #FFFFFF !important;
            font-weight: 600;
        }

        [data-testid="stFileUploaderDropzone"] {
            background-color: #111827 !important;
            border: 2px dashed #FFD700 !important;
        }

        /********* RADIO + SELECTBOX *********/
        div[role="radiogroup"] label {
            background: #34495E !important;
            color: yellow !important;
            padding: 8px 15px;
            border-radius: 8px;
            margin: 3px 0;
            cursor: pointer;
        }

        div[role="radiogroup"] label:hover {
            background: #1ABC9C !important;
        }

        .stSelectbox * {
            background-color: #34495E !important;
            color: yellow !important;
        }

        /********* JSON & CODE *********/
        [data-testid="stJson"] {
            background-color: #000000 !important;
            border-radius: 8px;
            padding: 10px;
        }

        [data-testid="stJson"] *,
        [data-testid="stJson"] div,
        [data-testid="stJson"] span,
        [data-testid="stJson"] p {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            font-family: 'Courier New', monospace !important;
        }

        code, pre {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border-radius: 5px;
            padding: 10px !important;
        }

        /********* DATAFRAMES *********/
        [data-testid="stDataFrame"] {
            background-color: #000000 !important;
        }

        [data-testid="stDataFrame"] * {
            color: #FFFFFF !important;
        }

        .stDataFrame table {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        .stDataFrame th {
            background-color: #1E3A5F !important;
            color: #FFD700 !important;
            font-weight: bold;
        }

        .stDataFrame td {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        /********* EXPANDERS *********/
        [data-testid="stExpander"] {
            background-color: #1E3A5F !important;
            border: 1px solid #FFD700 !important;
        }

        [data-testid="stExpander"] * {
            color: #FFFFFF !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_custom_css()
    
    # Initialisation des clés de session_state si elles n'existent pas
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'clim_data': None,
            'clim_data_prep': None,
            'clim_prep_info': {},
            'clim_model_info': {},
            'project_framing': {},
            'data_sources': {}
        })
    
    # Header HTML fixé en haut, comme pour l'app principale
    st.markdown(
        """
        <div class="custom-header">
            <div class="logo">🌍 Data Tool Climatique</div>
            <div class="menu">
                <a href="#">About</a>
                <a href="#">Documentation</a>
                <a href="#">Hackathon</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title(" Data Tool Climatique")
    st.markdown("Bienvenue dans ton outil de risque climatique interactif ")

    st.sidebar.title(" Navigation")
    section = st.sidebar.radio(
        "Aller à :",
        [
            " Cadrage du Projet",
            " Chargement",
            " EDA Climatique",
            " Prétraitement Climat",
            " Modélisation du Risque",
            " Évaluation & Scénarios",
            " Cartographie du Risque",
            " Reporting",
        ],
    )

    if section == " Cadrage du Projet":
        page_framing()
    elif section == " Chargement":
        page_loading()
    elif section == " EDA Climatique":
        page_eda()
    elif section == " Prétraitement Climat":
        page_preprocessing()
    elif section == " Modélisation du Risque":
        page_modeling()
    elif section == " Évaluation & Scénarios":
        page_evaluation()
    elif section == " Cartographie du Risque":
        page_maps()
    else:
        page_reporting()


def page_framing() -> None:
    """Page de cadrage du projet climat : objectif, unité d'analyse, cible."""

    st.header(" Cadrage du Projet Climat")
    st.markdown(
        """Définissez clairement l'objectif métier et le périmètre de votre analyse 
        avant de charger et traiter les données. Ces informations seront reprises dans le rapport final."""
    )

    st.subheader("1. Objectif métier")
    objective_type = st.selectbox(
        "Type d'objectif",
        [
            "Classification (risque élevé/moyen/faible, événement oui/non)",
            "Régression (score continu, perte attendue, variable climatique)",
            "Prévision (série temporelle future)",
        ],
        index=0,
    )
    objective_desc = st.text_area(
        "Description de l'objectif",
        placeholder="Ex: Prédire la probabilité d'inondation d'une zone dans les 12 prochains mois",
        height=80,
    )

    st.subheader("2. Unité d'analyse")
    unit_of_analysis = st.text_input(
        "Unité d'analyse",
        placeholder="Ex: zone géographique (maille), actif, quartier, station, jour/mois",
    )

    st.subheader("3. Cible attendue")
    target_desc = st.text_area(
        "Description de la variable cible",
        placeholder="Ex: Colonne 'risque_inondation' (0/1), ou 'perte_financiere' (continue)",
        height=80,
    )

    st.subheader("4. Contexte (optionnel)")
    context = st.text_input(
        "Contexte du projet",
        placeholder="Ex: Hackathon 48h, mission client, étude académique",
    )

    if st.button(" Enregistrer le cadrage"):
        st.session_state["project_framing"] = {
            "objective_type": objective_type,
            "objective_desc": objective_desc,
            "unit_of_analysis": unit_of_analysis,
            "target_desc": target_desc,
            "context": context,
        }
        st.success("✅ Cadrage enregistré ! Vous pouvez maintenant charger vos données.")

    # Afficher le cadrage actuel si déjà enregistré
    if "project_framing" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Cadrage actuel")
        framing = st.session_state["project_framing"]
        st.markdown(f"**Type d'objectif :** {framing['objective_type']}")
        if framing["objective_desc"]:
            st.markdown(f"**Description :** {framing['objective_desc']}")
        if framing["unit_of_analysis"]:
            st.markdown(f"**Unité d'analyse :** {framing['unit_of_analysis']}")
        if framing["target_desc"]:
            st.markdown(f"**Cible attendue :** {framing['target_desc']}")
        if framing["context"]:
            st.markdown(f"**Contexte :** {framing['context']}")


def _select_data_source() -> pd.DataFrame:
    """Helper pour sélectionner la source de données à utiliser."""
    try:
        data_sources = st.session_state.get("data_sources", {})
        df_prep = st.session_state.get("clim_data_prep")
        
        # Options disponibles - Ordre : Climat d'abord, Fusion en dernier
        options = []
        
        # 1. Ajouter les sources individuelles (Climat en priorité)
        if data_sources:
            # Priorité à "Climat" s'il existe
            if "Climat" in data_sources:
                options.append("Source : Climat")
            # Puis les autres sources dans l'ordre
            for label in data_sources.keys():
                if label != "Climat":
                    options.append(f"Source : {label}")
        
        # 2. Ajouter l'option de fusion (avant les données prétraitées)
        if data_sources and len(data_sources) > 1:
            options.append("Fusionner toutes les sources")
        
        # 3. Ajouter les données prétraitées en dernier
        if isinstance(df_prep, pd.DataFrame) and not df_prep.empty:
            options.append("Données prétraitées (fusionnées)")
        
        if not options:
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la sélection de la source : {e}")
        return None
    
    # Si une seule option, la retourner directement sans UI
    if len(options) == 1:
        if "prétraitées" in options[0]:
            return df_prep
        elif "Fusionner" in options[0]:
            # Cache la fusion
            if "merged_data" not in st.session_state:
                try:
                    st.session_state["merged_data"] = merge_dataframes(list(data_sources.values()))
                except (ValueError, TypeError) as e:
                    st.error(f"❌ Erreur lors de la fusion des données : {e}")
                    return None
            return st.session_state["merged_data"]
        else:
            source_label = options[0].replace("Source : ", "")
            return data_sources[source_label]
    
    # Sinon, proposer un selectbox
    choice = st.selectbox("📂 Choisir la source de données", options, key="data_source_selector")
    
    if "prétraitées" in choice:
        return df_prep
    elif "Fusionner" in choice:
        # Cache la fusion
        if "merged_data" not in st.session_state:
            try:
                st.session_state["merged_data"] = merge_dataframes(list(data_sources.values()))
            except (ValueError, TypeError) as e:
                st.error(f"❌ Erreur lors de la fusion des données : {e}")
                return None
        return st.session_state["merged_data"]
    else:
        source_label = choice.replace("Source : ", "")
        return data_sources[source_label]


def page_loading() -> None:
    st.header("📥 Chargement des données (multi-sources)")
    st.markdown(
        """Chargez plusieurs fichiers (climat, géographie, exposition, événements) 
        et labelisez-les pour les fusionner plus tard dans le prétraitement."""
    )

    # Initialiser data_sources si nécessaire
    if "data_sources" not in st.session_state:
        st.session_state["data_sources"] = {}

    st.subheader("📂 Ajouter une source de données")

    col1, col2 = st.columns([1, 2])
    with col1:
        source_label = st.selectbox(
            "Type de source",
            ["Climat", "Géographie", "Exposition", "Événements", "Autre"],
            key="source_label_select",
        )
        if source_label == "Autre":
            source_label = st.text_input("Nom personnalisé", key="custom_label")

    with col2:
        uploaded = st.file_uploader(
            "Charger un fichier (CSV ou Excel)",
            type=["csv", "xlsx", "xls"],
            key="multi_file_uploader",
        )

    sep = ","
    sheet = None
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            sep = st.selectbox("Séparateur CSV", [",", ";", "\t"], index=0)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Feuille Excel", xls.sheet_names)

        if st.button("➕ Ajouter cette source"):
            df = clim_data_loader.load_tabular_file(uploaded, sep=sep, sheet_name=sheet)
            if df is not None:
                st.session_state["data_sources"][source_label] = df
                st.success(f"✅ Source '{source_label}' ajoutée avec succès ({df.shape[0]} lignes × {df.shape[1]} colonnes).")
                # Compatibilité : si première source ou source "Climat", la mettre aussi dans clim_data
                if len(st.session_state["data_sources"]) == 1 or source_label == "Climat":
                    st.session_state["clim_data"] = df

    # Afficher les sources chargées
    if st.session_state["data_sources"]:
        st.markdown("---")
        st.subheader(" Sources chargées")
        for idx, (label, df) in enumerate(st.session_state["data_sources"].items()):
            st.markdown(f"** {label}** : {df.shape[0]} lignes × {df.shape[1]} colonnes")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                if st.button(f"🗑️ Supprimer", key=f"del_source_{idx}"):
                    del st.session_state["data_sources"][label]
                    if label == "Climat" and "clim_data" in st.session_state:
                        del st.session_state["clim_data"]
                    st.rerun()
            st.markdown("---")


def page_eda() -> None:
    st.header("🔎 EDA Climatique")
    
    # Sélection de la source
    df = _select_data_source()
    if df is None:
        st.warning("Veuillez d'abord charger des données dans l'onglet 📥 Chargement.")
        return

    st.subheader("Aperçu général")
    st.write(f"Shape : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    # Limiter à 10 lignes pour meilleures performances
    st.dataframe(df.head(10), use_container_width=True, height=300)

    st.subheader("Série temporelle simple")
    date_col = st.selectbox("Colonne date", options=["(aucune)"] + df.columns.tolist())
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    value_col = st.selectbox("Variable à tracer", options=num_cols) if num_cols else None

    if date_col != "(aucune)" and value_col:
        try:
            tmp = df[[date_col, value_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            tmp = tmp.sort_values(date_col)

            # Si énormément de points, on échantillonne pour ne pas saturer le navigateur
            max_points = 2000  # Réduit de 5000 à 2000 pour meilleures performances
            if len(tmp) > max_points:
                tmp = tmp.iloc[:: int(len(tmp) / max_points) + 1, :]

            chart = (
                alt.Chart(tmp)
                .mark_line(color="#FFD700", strokeWidth=2)
                .encode(
                    x=alt.X(
                        date_col, 
                        type="temporal", 
                        title=date_col,
                        axis=alt.Axis(
                            labelAngle=-45,
                            labelOverlap=False,
                            labelLimit=100,
                            format="%Y-%m-%d"
                        )
                    ),
                    y=alt.Y(value_col, type="quantitative", title=value_col),
                )
                .properties(height=300)
                .configure_view(strokeWidth=0)  # Optimisation
                .configure_axis(grid=True, gridOpacity=0.3)  # Grille légère pour meilleur rendu
            )

            st.altair_chart(chart, use_container_width=True)
        except Exception as exc:  # pragma: no cover - affichage utilisateur
            st.error(f"Impossible de tracer la série temporelle : {exc}")


def page_preprocessing() -> None:
    st.header(" Prétraitement Climat")
    
    # Sélection de la source (comme dans EDA)
    df = _select_data_source()
    if df is None:
        st.warning("Veuillez d'abord charger des données dans l'onglet 📥 Chargement.")
        return

    st.subheader("Paramètres de prétraitement")
    date_col = st.selectbox("Colonne date", options=["(aucune)"] + df.columns.tolist())
    freq = st.selectbox("Fréquence d’agrégation", options=["Aucune", "Jour", "Mois"], index=0)

    id_cols: list[str] = []
    st.markdown("**Colonnes d’identifiant (optionnel)**")
    id_cols = st.multiselect(
        "Colonnes d’identifiant (station, zone, etc.)",
        options=df.columns.tolist(),
    )

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    st.markdown("**Features temporelles avancées (optionnel)**")
    use_rolling = st.checkbox("Ajouter des moyennes glissantes (rolling)", value=False)
    rolling_cols = (
        st.multiselect("Colonnes numériques à étendre", options=num_cols)
        if use_rolling
        else []
    )

    st.markdown("**Détection simple d'anomalies (optionnel)**")
    use_anomaly = st.checkbox("Calculer un résumé d'outliers (z-score)", value=False)
    anomaly_cols = (
        st.multiselect("Colonnes numériques à analyser", options=num_cols, key="anomaly_cols_select")
        if use_anomaly
        else []
    )

    st.markdown("---")
    st.subheader("🌡️ Feature Engineering Climat Avancé")

    st.markdown("**Cumuls glissants (précipitations, degrés-jours, etc.)**")
    use_cumul = st.checkbox("Ajouter des cumuls sur N jours", value=False)
    cumul_cols = []
    cumul_windows = [7, 30]
    if use_cumul:
        cumul_cols = st.multiselect("Colonnes à cumuler", options=num_cols, key="cumul_cols_select")
        cumul_windows_str = st.text_input("Fenêtres (jours, séparées par virgule)", value="7,30")
        cumul_windows = [int(x.strip()) for x in cumul_windows_str.split(",") if x.strip().isdigit()]

    st.markdown("**Comptage de jours au-dessus d'un seuil**")
    use_threshold = st.checkbox("Compter les jours > seuil", value=False)
    threshold_cols = []
    thresholds_dict = {}
    threshold_windows = [7, 30]
    if use_threshold:
        threshold_cols = st.multiselect("Colonnes à analyser", options=num_cols, key="threshold_cols_select")
        if threshold_cols:
            st.markdown("Définir les seuils pour chaque colonne :")
            for col in threshold_cols:
                thresholds_dict[col] = st.number_input(f"Seuil pour {col}", value=30.0, key=f"threshold_{col}")
            threshold_windows_str = st.text_input("Fenêtres (jours, séparées par virgule)", value="7,30", key="threshold_windows")
            threshold_windows = [int(x.strip()) for x in threshold_windows_str.split(",") if x.strip().isdigit()]

    st.markdown("**Anomalies vs période de référence climatologique**")
    use_ref_anomaly = st.checkbox("Calculer anomalies vs référence", value=False)
    ref_anomaly_cols = []
    ref_start = "1990-01-01"
    ref_end = "2020-12-31"
    if use_ref_anomaly:
        ref_anomaly_cols = st.multiselect("Colonnes climatiques", options=num_cols, key="ref_anomaly_cols_select")
        col1, col2 = st.columns(2)
        with col1:
            ref_start = st.text_input("Début référence (YYYY-MM-DD)", value="1990-01-01")
        with col2:
            ref_end = st.text_input("Fin référence (YYYY-MM-DD)", value="2020-12-31")

    st.markdown("**Extremes glissants (min/max sur fenêtre)**")
    use_extremes = st.checkbox("Ajouter min/max glissants", value=False)
    extreme_cols = []
    extreme_windows = [7, 30]
    if use_extremes:
        extreme_cols = st.multiselect("Colonnes à analyser", options=num_cols, key="extreme_cols_select")
        extreme_windows_str = st.text_input("Fenêtres (jours, séparées par virgule)", value="7,30", key="extreme_windows")
        extreme_windows = [int(x.strip()) for x in extreme_windows_str.split(",") if x.strip().isdigit()]

    if st.button("Appliquer le prétraitement"):
        with st.spinner("Prétraitement en cours..."):
            dcol = None if date_col == "(aucune)" else date_col
            df_prep, info = clim_preprocessing.basic_climate_preprocessing(
                df,
                date_col=dcol,
                freq=freq,
                id_cols=id_cols,
                add_rolling=use_rolling,
                rolling_cols=rolling_cols,
                detect_anomalies=use_anomaly,
                anomaly_cols=anomaly_cols,
            )
            
            # Appliquer les features avancées si demandées
            if dcol:
                if use_cumul and cumul_cols:
                    df_prep = clim_preprocessing.add_cumulative_features(
                        df_prep, date_col=dcol, value_cols=cumul_cols, windows=cumul_windows
                    )
                    info["cumul_features"] = True
                
                if use_threshold and threshold_cols and thresholds_dict:
                    df_prep = clim_preprocessing.add_threshold_exceedance_features(
                        df_prep, date_col=dcol, value_cols=threshold_cols, 
                        thresholds=thresholds_dict, windows=threshold_windows
                    )
                    info["threshold_features"] = True
                
                if use_ref_anomaly and ref_anomaly_cols:
                    try:
                        df_prep = clim_preprocessing.add_reference_anomaly_features(
                            df_prep, date_col=dcol, value_cols=ref_anomaly_cols,
                            reference_start=ref_start, reference_end=ref_end
                        )
                        info["ref_anomaly_features"] = True
                    except Exception as e:
                        st.warning(f"Impossible de calculer les anomalies de référence : {e}")
                
                if use_extremes and extreme_cols:
                    df_prep = clim_preprocessing.add_extreme_features(
                        df_prep, date_col=dcol, value_cols=extreme_cols, windows=extreme_windows
                    )
                    info["extreme_features"] = True
            
            st.session_state["clim_data_prep"] = df_prep
            st.session_state["clim_prep_info"] = info

        st.success("Prétraitement terminé.")
        st.subheader("Aperçu après prétraitement")
        st.write(f"Shape : {df_prep.shape[0]} lignes × {df_prep.shape[1]} colonnes")
        st.dataframe(df_prep.head(), use_container_width=True)

        if info.get("anomaly_summary"):
            st.subheader("Résumé des anomalies (z-score > 3)")
            summary = info["anomaly_summary"]
            rows = [
                {
                    "colonne": col,
                    "nb_outliers": vals["nb_outliers"],
                    "pct_outliers": vals["pct_outliers"],
                }
                for col, vals in summary.items()
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_modeling() -> None:
    st.header(" Modélisation du Risque Climatique")
    
    # Sélection de la source de données
    df = _select_data_source()
    if df is None or df.empty:
        st.warning("Veuillez d'abord charger des données dans l'onglet 📥 Chargement.")
        return

    # Configuration de base
    st.subheader(" Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox("Colonne cible (risque)", options=df.columns.tolist())
        test_size = st.slider("Taille du jeu de test", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    with col2:
        # Détecter automatiquement le type de tâche
        y = df[target_col]
        detected_task = clim_model_comparison.detect_task_type(y)
        task_type = st.selectbox(
            "Type de tâche",
            options=["auto", "classification", "regression"],
            index=0,
            help=f"Détection automatique : {detected_task}"
        )
        if task_type == "auto":
            st.info(f"✓ Tâche détectée : **{detected_task}**")

    # Choix du mode
    st.markdown("---")
    st.subheader("🔧 Mode de modélisation")
    modeling_mode = st.radio(
        "Choisissez votre approche",
        ["Modèle unique", "Comparaison de modèles", "Affiner le meilleur modèle"],
        horizontal=True
    )

    # Options communes
    st.markdown("---")
    st.subheader(" Options avancées")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        handle_imbalance = st.checkbox("Gérer le déséquilibre", value=False)
    with col2:
        use_cv = st.checkbox("Validation croisée (5-fold)", value=False)
    with col3:
        fast_mode = st.checkbox("Mode rapide", value=False, help="Hyperparamètres optimisés pour la vitesse")

    # MODE 1 : Modèle unique
    if modeling_mode == "Modèle unique":
        st.markdown("---")
        st.subheader(" Modèle unique")
        
        model_name = st.selectbox(
            "Type de modèle",
            options=["Random Forest", "Gradient Boosting", "Logistic Regression", "Linear Regression", "Decision Tree"],
            index=0,
        )

        if st.button(" Entraîner le modèle"):
            with st.spinner("Entraînement en cours..."):
                results, final_task = clim_model_comparison.compare_models(
                    df,
                    target_col=target_col,
                    task=task_type,
                    test_size=test_size,
                    selected_models=[model_name],
                    fast_mode=fast_mode,
                    use_cv=use_cv,
                    handle_imbalance=handle_imbalance,
                )
                
                if results and results[0]["success"]:
                    result = results[0]
                    st.session_state["clim_model"] = result["pipeline"]
                    st.session_state["clim_model_info"] = {
                        "task_type": final_task,
                        "model_name": result["model_name"],
                        "metric_name": result["metric_name"],
                        "metric_value": result["test_score"],
                        "f1_score": result["f1_score"],
                        "cv_scores": result["cv_scores"],
                        "y_test": result.get("y_test"),
                        "y_pred": result.get("y_pred"),
                        "y_proba": result.get("y_proba"),
                        "X_test": result.get("X_test"),
                    }
                    
                    st.success(f"✅ Modèle entraîné : {result['model_name']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score Test", f"{result['test_score']:.4f}")
                    with col2:
                        st.metric("Score Train", f"{result['train_score']:.4f}")
                    with col3:
                        st.metric("Temps", f"{result['training_time']:.2f}s")

    # MODE 2 : Comparaison de modèles
    elif modeling_mode == "Comparaison de modèles":
        st.markdown("---")
        st.subheader(" Comparaison de modèles")
        
        # Obtenir les modèles disponibles
        available_models = list(clim_model_comparison.get_available_models(
            detected_task if task_type == "auto" else task_type, 
            fast_mode
        ).keys())
        
        selected_models = st.multiselect(
            "Modèles à comparer",
            options=available_models,
            default=available_models[:5] if len(available_models) >= 5 else available_models
        )

        if st.button(" Comparer les modèles"):
            if not selected_models:
                st.warning("Veuillez sélectionner au moins un modèle.")
            else:
                with st.spinner(f"Comparaison de {len(selected_models)} modèles..."):
                    results, final_task = clim_model_comparison.compare_models(
                        df,
                        target_col=target_col,
                        task=task_type,
                        test_size=test_size,
                        selected_models=selected_models,
                        fast_mode=fast_mode,
                        use_cv=use_cv,
                        handle_imbalance=handle_imbalance,
                    )
                    
                    # Afficher les résultats
                    best_result = clim_model_comparison.display_comparison_results(results, final_task)
                    
                    # Stocker le meilleur modèle
                    if best_result:
                        st.session_state["clim_model"] = best_result["pipeline"]
                        st.session_state["clim_model_info"] = {
                            "task_type": final_task,
                            "model_name": best_result["model_name"],
                            "metric_name": best_result["metric_name"],
                            "metric_value": best_result["test_score"],
                            "f1_score": best_result["f1_score"],
                            "cv_scores": best_result["cv_scores"],
                            "y_test": best_result.get("y_test"),
                            "y_pred": best_result.get("y_pred"),
                            "y_proba": best_result.get("y_proba"),
                            "X_test": best_result.get("X_test"),
                        }
                        st.session_state["clim_comparison_results"] = results

    # MODE 3 : Affiner le meilleur modèle
    elif modeling_mode == "Affiner le meilleur modèle":
        st.markdown("---")
        st.subheader("🔬 Affinage du meilleur modèle")
        
        if "clim_comparison_results" not in st.session_state:
            st.warning("⚠️ Veuillez d'abord comparer des modèles pour identifier le meilleur.")
        else:
            results = st.session_state["clim_comparison_results"]
            successful_results = [r for r in results if r["success"]]
            best_result = successful_results[max(range(len(successful_results)), key=lambda i: successful_results[i]["test_score"])]
            
            # Afficher le score de base avec contexte
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info(f" Modèle sélectionné : **{best_result['model_name']}**")
            with col2:
                st.metric("Score de base", f"{best_result['test_score']:.4f}")
            with col3:
                st.metric("Temps", f"{best_result['training_time']:.2f}s")
            
            st.success(f" Meilleur modèle de la comparaison (Score: {best_result['test_score']:.4f})")
            st.info(" Les hyperparamètres du meilleur modèle sont pré-remplis. Vous pouvez les modifier pour optimiser davantage.")
            
            st.markdown("---")
            st.markdown("**Hyperparamètres à affiner**")
            
            # Affinage selon le type de modèle
            if "Random Forest" in best_result["model_name"]:
                # Extraire les hyperparamètres actuels du pipeline
                current_model = best_result["pipeline"].named_steps.get("model")
                
                # Valeurs par défaut (pré-remplies depuis le meilleur modèle)
                default_n_estimators = getattr(current_model, 'n_estimators', 100)
                default_max_depth = getattr(current_model, 'max_depth', None) or 10
                default_min_samples_split = getattr(current_model, 'min_samples_split', 2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("n_estimators", 50, 500, default_n_estimators, 50, key="tune_n_est")
                with col2:
                    max_depth = st.slider("max_depth", 3, 30, default_max_depth, 1, key="tune_max_depth")
                with col3:
                    min_samples_split = st.slider("min_samples_split", 2, 20, default_min_samples_split, 1, key="tune_min_split")
                
                if st.button("🚀 Affiner le modèle"):
                    with st.spinner("Réentraînement avec les nouveaux hyperparamètres..."):
                        # Récupérer le type de tâche
                        task_type = st.session_state.get("clim_model_info", {}).get("task_type", detected_task)
                        
                        # Créer le nouveau modèle avec les hyperparamètres affinés
                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                        
                        if task_type == "classification":
                            tuned_model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42,
                                n_jobs=-1,
                                class_weight="balanced" if handle_imbalance else None
                            )
                        else:
                            tuned_model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42,
                                n_jobs=-1
                            )
                        
                        # Réentraîner
                        tuned_results, _ = clim_model_comparison.compare_models(
                            df,
                            target_col=target_col,
                            task=task_type,
                            test_size=test_size,
                            selected_models=["Random Forest (Tuned)"],
                            fast_mode=False,
                            use_cv=use_cv,
                            handle_imbalance=handle_imbalance,
                        )
                        
                        # Remplacer le modèle par le modèle affiné
                        if tuned_results and tuned_results[0]["success"]:
                            tuned_result = tuned_results[0]
                            
                            # Comparer avec le modèle de base
                            st.markdown("---")
                            st.subheader("📊 Résultats de l'affinage")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Score Test",
                                    f"{tuned_result['test_score']:.4f}",
                                    delta=f"{tuned_result['test_score'] - best_result['test_score']:.4f}"
                                )
                            with col2:
                                st.metric(
                                    "Score Train",
                                    f"{tuned_result['train_score']:.4f}",
                                    delta=f"{tuned_result['train_score'] - best_result['train_score']:.4f}"
                                )
                            with col3:
                                st.metric("Temps", f"{tuned_result['training_time']:.2f}s")
                            
                            # Afficher un message selon l'amélioration
                            improvement = tuned_result['test_score'] - best_result['test_score']
                            if improvement > 0.01:
                                st.success(f" Amélioration significative : +{improvement:.4f}")
                            elif improvement > 0:
                                st.info(f"✓ Légère amélioration : +{improvement:.4f}")
                            else:
                                st.warning(f"⚠️ Pas d'amélioration : {improvement:.4f}")
                            
                            # Sauvegarder le modèle affiné
                            st.session_state["clim_model"] = tuned_result["pipeline"]
                            st.session_state["clim_model_info"] = {
                                "task_type": task_type,
                                "model_name": f"{tuned_result['model_name']} (Affiné)",
                                "metric_name": tuned_result["metric_name"],
                                "metric_value": tuned_result["test_score"],
                                "f1_score": tuned_result["f1_score"],
                                "cv_scores": tuned_result["cv_scores"],
                            }
                            
                            st.success("✅ Modèle affiné sauvegardé !")
            
            elif "Gradient Boosting" in best_result["model_name"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("n_estimators", 50, 500, 100, 50, key="tune_gb_n_est")
                with col2:
                    learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key="tune_gb_lr")
                with col3:
                    max_depth = st.slider("max_depth", 3, 10, 3, 1, key="tune_gb_depth")
                
                if st.button("🚀 Affiner le modèle"):
                    st.info("Affinage Gradient Boosting en cours...")
                    # Logique similaire pour GB
                    st.warning("Implémentation complète à venir pour Gradient Boosting")
            
            else:
                st.info(" L'affinage détaillé pour ce type de modèle sera ajouté prochainement.")
                st.markdown("""
                **Modèles supportés pour l'affinage :**
                - ✅ Random Forest
                - 🔄 Gradient Boosting (bientôt)
                - 🔄 Autres modèles (bientôt)
                """)


def page_evaluation() -> None:
    st.header(" Évaluation & Scénarios")
    info = st.session_state.get("clim_model_info")
    if info is None:
        st.warning("Aucun modèle climat n’a encore été entraîné.")
        return

    clim_evaluation.show_evaluation(info)


def page_maps() -> None:
    # Fusionner toutes les sources de données disponibles
    df_prep = st.session_state.get("clim_data_prep")
    data_sources = st.session_state.get("data_sources", {})
    
    # Priorité : données prétraitées, sinon fusion des sources
    if isinstance(df_prep, pd.DataFrame) and not df_prep.empty:
        df = df_prep
    elif data_sources:
        # Fusionner toutes les sources sur une colonne commune si possible
        dfs = list(data_sources.values())
        if len(dfs) == 1:
            df = dfs[0]
        else:
            # Essayer de fusionner sur colonnes communes (lat/lon ou date)
            df = dfs[0].copy()
            for i, other_df in enumerate(dfs[1:], 1):
                # Détection de colonnes communes
                common_cols = list(set(df.columns) & set(other_df.columns))
                if common_cols:
                    # Fusionner sur colonnes communes
                    df = pd.merge(df, other_df, on=common_cols, how="outer", suffixes=("", f"_dup{i}"))
                    # Supprimer les colonnes dupliquées
                    dup_cols = [c for c in df.columns if f"_dup{i}" in c]
                    if dup_cols:
                        df = df.drop(columns=dup_cols)
                else:
                    # Concaténer si pas de colonnes communes
                    st.info("Fusion par concaténation (pas de colonnes communes détectées)")
                    other_df_renamed = other_df.copy()
                    for col in other_df.columns:
                        if col in df.columns:
                            other_df_renamed = other_df_renamed.rename(columns={col: f"{col}_src{i+1}"})
                    df = pd.concat([df, other_df_renamed], axis=1)
    else:
        df = st.session_state.get("clim_data")

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("Veuillez d'abord charger des données.")
        return

    # Récupérer les sources de données et les données prétraitées
    data_sources = st.session_state.get("data_sources", {})
    df_prep = st.session_state.get("clim_data_prep")
    
    # Ne pas afficher de titre ici car run_maps_page le fait déjà
    clim_maps.run_maps_page(df, title="", data_sources=data_sources, df_prep=df_prep)


def page_reporting() -> None:
    """Page de reporting avec interface améliorée"""
    st.header("📊 Reporting Climat")
    
    # Vérification des données disponibles
    has_data = "clim_data" in st.session_state and st.session_state["clim_data"] is not None
    has_prep = "clim_data_prep" in st.session_state and st.session_state["clim_data_prep"] is not None
    has_model = "clim_model_info" in st.session_state and st.session_state["clim_model_info"]
    
    if not has_data:
        st.warning("⚠️ Aucune donnée n'a été chargée. Veuillez d'abord charger des données depuis l'onglet 'Chargement des Données'.")
        return
    
    # Affichage du résumé
    st.markdown("---")
    st.subheader("📊 Synthèse du Projet Climatique")
    show_reporting_summary(st.session_state)
    
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
        
        **Sections sélectionnées :** {len(selected_sections)}/6
        """
        st.info(info_text)
        
    # Bouton de génération
    st.markdown("---")
    st.subheader("📤 Exporter le Rapport")
    
    # Centrer le bouton de génération
    col_generate, col_empty = st.columns([1, 1])
    with col_generate:
        if st.button("💾 Générer le rapport HTML", type="primary", use_container_width=True):
            with st.spinner("Génération du rapport en cours..."):
                try:
                    # Créer une copie du contexte avec les sections sélectionnées
                    report_context = {
                        **st.session_state,
                        "report_options": {
                            "sections": selected_sections,
                            "title": report_title,
                            "include_code": include_code
                        }
                    }
                    
                    report_path = generate_html_report(report_context)
                    if report_path:
                        st.success("✅ Rapport généré avec succès !")
                        
                        # Affichage du bouton de téléchargement
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Télécharger le rapport",
                                data=f,
                                file_name=os.path.basename(report_path),
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
        """)


if __name__ == "__main__":  # pragma: no cover
    main()
