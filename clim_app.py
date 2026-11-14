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

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from climatool import data_loader, evaluation, maps, model_comparison, preprocessing, pricing, reporting, tuning
from climatool.data_utils import merge_dataframes

APP_VERSION = "1.3.0"
CSS_FILE = Path(__file__).parent / "climatool" / "assets" / "style.css"


st.set_page_config(
    page_title="Data Tool Climatique",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _inject_custom_css() -> None:
    """Applique le thème visuel commun avec Data Project Tool."""
    st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    _inject_custom_css()

    # Initialisation des clés de session_state si elles n'existent pas
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'clim_data': None,
            'clim_data_prep': None,
            'clim_prep_info': {},
            'clim_model_info': None,
            'project_framing': {},
            'data_sources': {}
        })

    st.caption(f"Développé par Yacoubou KOUMAI - v{APP_VERSION}")
    st.title("Data Tool Climatique")
    st.markdown("Analyse et modélisation du risque climatique, du cadrage au rapport final.")

    pages = {
        "Cadrage du Projet": page_framing,
        "Chargement": page_loading,
        "EDA Climatique": page_eda,
        "Prétraitement Climat": page_preprocessing,
        "Modélisation du Risque": page_modeling,
        "Évaluation & Scénarios": page_evaluation,
        "Tarification": page_pricing,
        "Cartographie du Risque": page_maps,
        "Reporting": page_reporting,
    }
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Aller à :", list(pages))
    pages[section]()


def page_framing() -> None:
    """Page de cadrage du projet climat : objectif, unité d'analyse, cible."""

    st.header("Cadrage du Projet Climat")
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

    if st.button("Enregistrer le cadrage"):
        st.session_state["project_framing"] = {
            "objective_type": objective_type,
            "objective_desc": objective_desc,
            "unit_of_analysis": unit_of_analysis,
            "target_desc": target_desc,
            "context": context,
        }
        st.success("Cadrage enregistré ! Vous pouvez maintenant charger vos données.")

    # Afficher le cadrage actuel si déjà enregistré
    if st.session_state.get("project_framing"):
        st.markdown("---")
        st.subheader("Cadrage actuel")
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


def _merged_sources() -> pd.DataFrame | None:
    """Fusion des sources chargées, mise en cache jusqu'au prochain ajout ou retrait de source."""
    if "merged_data" not in st.session_state:
        try:
            st.session_state["merged_data"] = merge_dataframes(list(st.session_state.get("data_sources", {}).values()))
        except (ValueError, TypeError) as e:
            st.error(f"Erreur lors de la fusion des données : {e}")
            return None
    return st.session_state["merged_data"]


def _resolve_source(option: str) -> pd.DataFrame | None:
    if "prétraitées" in option:
        return st.session_state.get("clim_data_prep")
    if "Fusionner" in option:
        return _merged_sources()
    return st.session_state.get("data_sources", {}).get(option.replace("Source : ", ""))


def _select_data_source(page: str, prefer_prepared: bool = True,
                        required_cols: list[str] | None = None) -> pd.DataFrame | None:
    """Choix de la source de données d'une page.

    Avec ``required_cols``, seules les sources qui contiennent toutes ces colonnes sont
    proposées (par exemple celles sur lesquelles le modèle a été entraîné).
    """
    data_sources = st.session_state.get("data_sources", {}) or {}
    df_prep = st.session_state.get("clim_data_prep")

    # Ordre : Climat, autres sources, fusion, données prétraitées
    options = []
    if "Climat" in data_sources:
        options.append("Source : Climat")
    options += [f"Source : {label}" for label in data_sources if label != "Climat"]
    if len(data_sources) > 1:
        options.append("Fusionner toutes les sources")
    if isinstance(df_prep, pd.DataFrame) and not df_prep.empty:
        options.append("Données prétraitées (fusionnées)")

    if required_cols:
        def has_cols(option: str) -> bool:
            if "Fusionner" in option:
                # Évite de calculer la fusion juste pour vérifier ses colonnes
                cols = set().union(*(set(d.columns) for d in data_sources.values()))
            else:
                d = _resolve_source(option)
                cols = set(d.columns) if isinstance(d, pd.DataFrame) else set()
            return set(required_cols) <= cols
        options = [o for o in options if has_cols(o)]

    if not options:
        return None
    if len(options) == 1:
        return _resolve_source(options[0])

    # Par défaut les données prétraitées si elles existent (sauf pour le prétraitement
    # lui-même, qui part des données brutes fusionnées).
    if prefer_prepared and any("prétraitées" in o for o in options):
        default = next(i for i, o in enumerate(options) if "prétraitées" in o)
    elif any("Fusionner" in o for o in options):
        default = next(i for i, o in enumerate(options) if "Fusionner" in o)
    else:
        default = 0
    key = f"data_source_{page}"
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    choice = st.selectbox("Choisir la source de données", options, index=default, key=key)
    return _resolve_source(choice)


def page_loading() -> None:
    st.header("Chargement des données (multi-sources)")
    st.markdown(
        """Chargez plusieurs fichiers (climat, géographie, exposition, événements)
        et labelisez-les pour les fusionner plus tard dans le prétraitement."""
    )

    # Initialiser data_sources si nécessaire
    if "data_sources" not in st.session_state:
        st.session_state["data_sources"] = {}

    st.subheader("Ajouter une source de données")

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

        if st.button("Ajouter cette source"):
            df = data_loader.load_tabular_file(uploaded, sep=sep, sheet_name=sheet)
            if df is not None:
                st.session_state["data_sources"][source_label] = df
                st.session_state.pop("merged_data", None)  # la fusion doit être recalculée
                st.success(f"Source '{source_label}' ajoutée avec succès ({df.shape[0]} lignes × {df.shape[1]} colonnes).")
                # Compatibilité : si première source ou source "Climat", la mettre aussi dans clim_data
                if len(st.session_state["data_sources"]) == 1 or source_label == "Climat":
                    st.session_state["clim_data"] = df

    # Afficher les sources chargées
    if st.session_state["data_sources"]:
        st.markdown("---")
        st.subheader("Sources chargées")
        for idx, (label, df) in enumerate(st.session_state["data_sources"].items()):
            st.markdown(f"**{label}** : {df.shape[0]} lignes × {df.shape[1]} colonnes")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(df.head(), width="stretch")
            with col2:
                if st.button("Supprimer", key=f"del_source_{idx}"):
                    del st.session_state["data_sources"][label]
                    st.session_state.pop("merged_data", None)
                    if label == "Climat" and "clim_data" in st.session_state:
                        del st.session_state["clim_data"]
                    st.rerun()
            st.markdown("---")


def page_eda() -> None:
    st.header("EDA Climatique")

    # Sélection de la source
    df = _select_data_source("eda")
    if df is None:
        st.warning("Veuillez d'abord charger des données dans l'onglet Chargement.")
        return

    st.subheader("Aperçu général")
    st.write(f"Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    # Limiter à 10 lignes pour meilleures performances
    st.dataframe(df.head(10), width="stretch", height=300)

    st.subheader("Série temporelle simple")
    date_candidates = _date_columns(df)
    if not date_candidates:
        st.info("Aucune colonne date dans cette source : choisissez une autre source pour tracer une série temporelle.")
        return
    date_col = st.selectbox("Colonne date", options=date_candidates)
    id_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != date_col
               and 1 < df[c].nunique() <= 50]
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ("latitude", "longitude")]
    value_col = st.selectbox("Variable à tracer", options=num_cols) if num_cols else None

    group_col = None
    if id_cols:
        group_col = id_cols[0]
        groups = st.multiselect(f"{group_col} à afficher", options=sorted(df[group_col].dropna().unique().tolist()),
                                default=sorted(df[group_col].dropna().unique().tolist())[:3])
    if value_col:
        try:
            cols = [date_col, value_col] + ([group_col] if group_col else [])
            tmp = df[cols].copy()
            if group_col:
                tmp = tmp[tmp[group_col].isin(groups)]
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col])
            tmp = tmp.sort_values(date_col)

            # Si énormément de points, on échantillonne pour ne pas saturer le navigateur
            max_points = 2000  # Réduit de 5000 à 2000 pour meilleures performances
            if len(tmp) > max_points and not group_col:
                tmp = tmp.iloc[:: int(len(tmp) / max_points) + 1, :]
            elif group_col and len(tmp) > max_points:
                # Moyenne hebdomadaire par groupe plutôt qu'un sous-échantillon qui hacherait les courbes
                tmp = (tmp.groupby([group_col, pd.Grouper(key=date_col, freq="W")])[value_col]
                       .mean().reset_index())

            chart = (
                alt.Chart(tmp)
                .mark_line(strokeWidth=1.5)
                .encode(
                    color=(alt.Color(f"{group_col}:N", title=group_col) if group_col
                           else alt.value("#FFD700")),
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

            st.altair_chart(chart, width="stretch")
        except Exception as exc:  # pragma: no cover - affichage utilisateur
            st.error(f"Impossible de tracer la série temporelle : {exc}")


def page_preprocessing() -> None:
    """Page de prétraitement épurée et simplifiée"""
    st.header("Prétraitement Climat")

    # Sélection de la source
    df = _select_data_source("preprocessing", prefer_prepared=False)
    if df is None:
        st.warning("Veuillez d'abord charger des données dans l'onglet Chargement.")
        return

    st.caption(
        "Les cumuls et moyennes glissantes sont calculés séparément pour chaque identifiant "
        "(zone, station) : les valeurs de deux zones ne sont jamais mélangées."
    )
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text_cols = [c for c in df.columns if c not in num_cols]

    st.subheader("1. Temps et identifiants")
    col1, col2 = st.columns(2)
    with col1:
        date_options = ["(aucune)"] + _date_columns(df)
        date_col = st.selectbox("Colonne date", options=date_options, index=1 if len(date_options) > 1 else 0)
    has_date = date_col != "(aucune)"
    with col2:
        freq = st.selectbox(
            "Agrégation temporelle",
            options=["Aucune", "Jour", "Mois"] if has_date else ["Aucune"],
            index=0,
            help="Regroupe les observations par période et par identifiant.",
        )
    if not has_date:
        st.info("Cette source n'a pas de colonne date : l'agrégation temporelle et les variables "
                "glissantes ne s'appliquent pas. Seul le contrôle des valeurs extrêmes est disponible.")
    id_guess = [c for c in text_cols if any(k in c.lower() for k in ("zone", "station", "site", "commune", "region", "id"))]
    id_cols = st.multiselect(
        "Colonnes d'identifiant (zone, station...)",
        options=df.columns.tolist(),
        default=id_guess,
        help="Chaque identifiant forme une série temporelle distincte.",
    )

    sum_cols: list[str] = []
    if freq != "Aucune":
        sum_guess = [c for c in num_cols if any(k in c.lower() for k in ("precip", "pluie", "cout", "coût", "sinistre", "montant", "perte"))]
        sum_cols = st.multiselect(
            "Colonnes à additionner sur la période",
            options=num_cols,
            default=sum_guess,
            help="Pluie, coûts de sinistres... Les colonnes 0/1 (événement) prennent le maximum, "
                 "les autres la moyenne.",
        )

    windows: list[int] = [3, 7]
    use_cumul = use_rolling = False
    cumul_cols: list[str] = []
    rolling_cols: list[str] = []
    if has_date:
        st.subheader("2. Variables construites")
        unit = {"Mois": "mois"}.get(freq, "jours")
        windows = st.multiselect(
            f"Fenêtres glissantes (en {unit})",
            options=[2, 3, 5, 7, 14, 30],
            default=[3, 7],
        )
        use_cumul = st.checkbox(
            "Cumuls glissants (ex. pluie cumulée sur 3 et 7 jours)",
            value=False,
            help="Souvent la variable la plus explicative du risque d'inondation.",
        )
        cumul_guess = [c for c in num_cols if any(k in c.lower() for k in ("precip", "pluie"))]
        cumul_cols = st.multiselect("Colonnes à cumuler", options=num_cols, default=cumul_guess) if use_cumul else []
        use_rolling = st.checkbox("Moyennes glissantes (ex. température moyenne sur 7 jours)", value=False)
        rolling_cols = st.multiselect("Colonnes à lisser", options=num_cols) if use_rolling else []
        if (use_cumul or use_rolling) and not windows:
            st.warning("Choisissez au moins une fenêtre glissante.")

    st.subheader("3. Contrôle qualité" if has_date else "2. Contrôle qualité")
    use_anomaly = st.checkbox("Signaler les valeurs extrêmes (écart à la moyenne de plus de 3 écarts-types)", value=False)
    anomaly_cols = st.multiselect("Colonnes à contrôler", options=num_cols, key="anomaly_cols") if use_anomaly else []

    st.markdown("---")
    if st.button("Appliquer le prétraitement", type="primary", width="stretch"):
        with st.spinner("Prétraitement en cours..."):
            try:
                dcol = None if date_col == "(aucune)" else date_col
                if (use_cumul or use_rolling or freq != "Aucune") and dcol is None:
                    st.error("Choisissez une colonne date pour l'agrégation et les variables glissantes.")
                    return
                df_prep, info = preprocessing.basic_climate_preprocessing(
                    df,
                    date_col=dcol,
                    freq=freq,
                    id_cols=id_cols,
                    add_rolling=use_rolling,
                    rolling_cols=rolling_cols,
                    detect_anomalies=use_anomaly,
                    anomaly_cols=anomaly_cols,
                    sum_cols=sum_cols,
                    add_cumul=use_cumul,
                    cumul_cols=cumul_cols,
                    windows=sorted(windows) or [3],
                )

                # Sauvegarde dans session_state
                st.session_state["clim_data_prep"] = df_prep
                st.session_state["clim_prep_info"] = info

                # Affichage des résultats
                st.success("Prétraitement terminé avec succès!")
                st.subheader("Résultats du prétraitement")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lignes", f"{df_prep.shape[0]:,}")
                with col2:
                    st.metric("Colonnes", df_prep.shape[1])
                with col3:
                    st.metric("Nouvelles variables", df_prep.shape[1] - df.shape[1])

                st.subheader("Aperçu des données prétraitées")
                st.dataframe(df_prep.head(), width="stretch")

                # Afficher le résumé des anomalies si disponible
                if info.get("anomaly_summary"):
                    st.subheader("Valeurs extrêmes détectées")
                    summary = info["anomaly_summary"]
                    rows = [
                        {
                            "Colonne": col,
                            "Valeurs extrêmes": vals["nb_outliers"],
                            "Part": f"{vals['pct_outliers']:.2f} %",
                        }
                        for col, vals in summary.items()
                    ]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch")

            except Exception as e:
                st.error(f"Erreur lors du prétraitement : {str(e)}")
                st.exception(e)


_ID_WORDS = ("zone", "station", "site", "commune", "region", "région", "ville", "pays", "id")
_COORD_WORDS = ("latitude", "longitude", "lat", "lon", "lng")


def _date_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes de type date, ou textes dont presque toutes les valeurs se lisent comme des dates."""
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
        elif not pd.api.types.is_numeric_dtype(s) and s.notna().any():
            sample = s.dropna().astype(str).head(200)
            if pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() >= 0.9:
                out.append(c)
    return out


def _target_columns(df: pd.DataFrame) -> list[str]:
    """Cibles possibles : variables numériques non constantes ou catégories (2 à 10 modalités),
    hors dates, coordonnées et identifiants."""
    dates = set(_date_columns(df))
    out = []
    for c in df.columns:
        name = c.lower()
        if c in dates or name in _COORD_WORDS or any(w == name or name.startswith(w + "_") for w in _ID_WORDS):
            continue
        n = df[c].nunique()
        if pd.api.types.is_numeric_dtype(df[c]):
            if n > 1:
                out.append(c)
        elif 2 <= n <= 10:
            out.append(c)
    return out


def _is_binary(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and set(series.dropna().unique()) <= {0, 1} and series.nunique() == 2


def _leakage_suspects(df: pd.DataFrame, target_col: str) -> list[str]:
    """Colonnes qui reproduisent (presque) la cible : à exclure des variables explicatives.

    Exemple : un coût de sinistre n'est positif que les jours d'événement.
    """
    y = df[target_col]
    suspects = []
    for col in df.select_dtypes(include="number").columns:
        if col == target_col:
            continue
        x = df[col]
        if _is_binary(y):
            if ((x.fillna(0) != 0).astype(int) == y.fillna(0)).mean() >= 0.99:
                suspects.append(col)
                continue
        if (pd.api.types.is_numeric_dtype(y) and x.std() > 0 and y.std() > 0
                and abs(x.corr(y)) >= 0.95):
            suspects.append(col)
    return suspects


def _select_features(df: pd.DataFrame, target_col: str) -> tuple[list[str], str | None]:
    """Choix des variables explicatives et du découpage train / test."""
    st.subheader("Variables explicatives")
    date_cols = [c for c in df.columns if c != target_col and (
        pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower())]
    id_like = [c for c in df.columns if c not in date_cols and c != target_col
               and not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 50]
    suspects = _leakage_suspects(df, target_col)
    candidates = [c for c in df.columns if c != target_col]
    default = [c for c in candidates if c not in date_cols + id_like + suspects]

    if suspects:
        st.warning(
            "Colonnes écartées par défaut car elles reproduisent la cible (fuite d'information) : "
            + ", ".join(f"`{c}`" for c in suspects)
            + ". Elles ne sont connues qu'après l'événement : les garder donnerait un modèle "
            "parfait en apparence mais inutilisable."
        )
    feature_cols = st.multiselect(
        "Variables utilisées par le modèle",
        options=candidates,
        default=default,
        help="Dates et identifiants à trop de modalités sont exclus par défaut.",
        # La clé dépend des colonnes disponibles : changer de source (données prétraitées...)
        # recalcule la sélection par défaut au lieu de garder l'ancienne.
        key=f"features_{target_col}_{abs(hash(tuple(df.columns))) % 10**8}",
    )

    time_col = None
    if date_cols:
        temporal = st.checkbox(
            f"Découpage temporel sur `{date_cols[0]}` : apprentissage sur le passé, test sur la période la plus récente",
            value=True,
            help="Plus réaliste pour un risque à anticiper : le modèle est jugé sur des dates qu'il n'a jamais vues.",
        )
        time_col = date_cols[0] if temporal else None
    return feature_cols, time_col


def page_modeling() -> None:
    st.header("Modélisation du Risque Climatique")

    # Sélection de la source de données
    df = _select_data_source("modeling")
    if df is None or df.empty:
        st.warning("Veuillez d'abord charger des données dans l'onglet Chargement.")
        return

    # Configuration de base
    st.subheader("Configuration")
    col1, col2 = st.columns(2)

    with col1:
        target_options = _target_columns(df)
        if not target_options:
            st.warning("Aucune colonne utilisable comme cible dans cette source.")
            return
        binary_cols = [c for c in target_options if _is_binary(df[c])]
        default_target = binary_cols[-1] if binary_cols else target_options[-1]
        target_col = st.selectbox(
            "Colonne cible (risque)",
            options=target_options,
            index=target_options.index(default_target),
            help="Par défaut, la dernière colonne 0/1 (événement observé). Dates, coordonnées et "
                 "identifiants (zone, station) sont exclus.",
        )
        test_size = st.slider("Taille du jeu de test", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    with col2:
        # Détecter automatiquement le type de tâche
        y = df[target_col]
        detected_task = model_comparison.detect_task_type(y)
        task_type = st.selectbox(
            "Type de tâche",
            options=["auto", "classification", "regression"],
            index=0,
            help=f"Détection automatique : {detected_task}"
        )
        if task_type == "auto":
            st.info(f"Tâche détectée : **{detected_task}**")

    effective_task = detected_task if task_type == "auto" else task_type
    if effective_task == "classification" and _is_binary(df[target_col]):
        rate = df[target_col].mean()
        st.caption(
            f"Fréquence de l'événement : {rate:.2%} des observations. La métrique principale est l'AUC ROC, "
            "qui reste pertinente quand l'événement est rare (contrairement à l'accuracy)."
        )

    feature_cols, time_col = _select_features(df, target_col)
    if not feature_cols:
        st.warning("Sélectionnez au moins une variable explicative.")
        return

    # Choix du mode
    st.markdown("---")
    st.subheader("Mode de modélisation")
    modeling_mode = st.radio(
        "Choisissez votre approche",
        ["Modèle unique", "Comparaison de modèles", "Affiner le meilleur modèle"],
        horizontal=True
    )

    # Options communes
    st.markdown("---")
    st.subheader("Options avancées")
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
        st.subheader("Modèle unique")

        model_name = st.selectbox(
            "Type de modèle",
            options=list(model_comparison.get_available_models(effective_task, fast_mode).keys()),
            index=0,
            key=f"single_model_{effective_task}_{fast_mode}",
        )

        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement en cours..."):
                results, final_task = model_comparison.compare_models(
                    df,
                    target_col=target_col,
                    task=task_type,
                    test_size=test_size,
                    selected_models=[model_name],
                    fast_mode=fast_mode,
                    use_cv=use_cv,
                    handle_imbalance=handle_imbalance,
                    feature_cols=feature_cols,
                    time_col=time_col,
                )

                if results and results[0]["success"]:
                    result = results[0]
                    st.session_state["clim_model"] = result["pipeline"]
                    st.session_state["clim_model_info"] = {
                        "task_type": final_task,
                        "model_name": result["model_name"],
                        "metric_name": result["metric_name"],
                        "metric_value": result["test_score"],
                        "target_col": target_col,  # Ajout de la variable cible
                        "feature_cols": feature_cols,
                        "time_col": time_col,
                        # Métriques de classification
                        "f1_score": result.get("f1_score"),
                        "precision": result.get("precision"),
                        "recall": result.get("recall"),
                        "accuracy": result.get("accuracy"),
                        "roc_auc": result.get("roc_auc"),
                        "pr_auc": result.get("pr_auc"),
                        # Métriques de régression
                        "mse": result.get("mse"),
                        "rmse": result.get("rmse"),
                        "mae": result.get("mae"),
                        "r2": result.get("r2"),
                        "cv_scores": result["cv_scores"],
                        "y_test": result.get("y_test"),
                        "y_pred": result.get("y_pred"),
                        "y_proba": result.get("y_proba"),
                        "X_test": result.get("X_test"),
                        # Informations manquantes ajoutées
                        "train_shape": result.get("train_shape"),
                        "test_shape": result.get("test_shape"),
                        "feature_importance": result.get("feature_importance"),
                        "feature_names": result.get("feature_names"),
                    }

                    st.success(f"Modèle entraîné : {result['model_name']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{result['metric_name']} (test)", f"{result['test_score']:.4f}")
                    with col2:
                        st.metric(f"{result['metric_name']} (apprentissage)", f"{result['train_score']:.4f}")
                    with col3:
                        st.metric("Temps", f"{result['training_time']:.2f}s")
                else:
                    error = results[0].get("error", "erreur inconnue") if results else "aucun résultat"
                    st.error(f"L'entraînement a échoué : {error}")

    # MODE 2 : Comparaison de modèles
    elif modeling_mode == "Comparaison de modèles":
        st.markdown("---")
        st.subheader("Comparaison de modèles")

        # Obtenir les modèles disponibles
        available_models = list(model_comparison.get_available_models(effective_task, fast_mode).keys())

        selected_models = st.multiselect(
            "Modèles à comparer",
            options=available_models,
            default=available_models[:5],
            key=f"compare_models_{effective_task}_{fast_mode}",
        )

        if st.button("Comparer les modèles"):
            if not selected_models:
                st.warning("Veuillez sélectionner au moins un modèle.")
            else:
                with st.spinner(f"Comparaison de {len(selected_models)} modèles..."):
                    results, final_task = model_comparison.compare_models(
                        df,
                        target_col=target_col,
                        task=task_type,
                        test_size=test_size,
                        selected_models=selected_models,
                        fast_mode=fast_mode,
                        use_cv=use_cv,
                        handle_imbalance=handle_imbalance,
                        feature_cols=feature_cols,
                        time_col=time_col,
                    )

                    # Afficher les résultats
                    best_result = model_comparison.display_comparison_results(results, final_task)

                    # Stocker le meilleur modèle
                    if best_result:
                        st.session_state["clim_model"] = best_result["pipeline"]
                        st.session_state["clim_model_info"] = {
                            "task_type": final_task,
                            "model_name": best_result["model_name"],
                            "metric_name": best_result["metric_name"],
                            "metric_value": best_result["test_score"],
                            "target_col": target_col,  # Ajout de la variable cible
                            "feature_cols": feature_cols,
                            "time_col": time_col,
                            # Métriques de classification
                            "f1_score": best_result.get("f1_score"),
                            "precision": best_result.get("precision"),
                            "recall": best_result.get("recall"),
                        "accuracy": best_result.get("accuracy"),
                        "roc_auc": best_result.get("roc_auc"),
                        "pr_auc": best_result.get("pr_auc"),
                            # Métriques de régression
                            "mse": best_result.get("mse"),
                            "rmse": best_result.get("rmse"),
                            "mae": best_result.get("mae"),
                            "r2": best_result.get("r2"),
                            "cv_scores": best_result["cv_scores"],
                            "y_test": best_result.get("y_test"),
                            "y_pred": best_result.get("y_pred"),
                            "y_proba": best_result.get("y_proba"),
                            "X_test": best_result.get("X_test"),
                            # Informations manquantes ajoutées
                            "train_shape": best_result.get("train_shape"),
                            "test_shape": best_result.get("test_shape"),
                            "feature_importance": best_result.get("feature_importance"),
                            "feature_names": best_result.get("feature_names"),
                        }
                        st.session_state["clim_comparison_results"] = results
                        st.session_state["clim_comparison_context"] = (target_col, tuple(feature_cols))

    # MODE 3 : Affiner le meilleur modèle
    elif modeling_mode == "Affiner le meilleur modèle":
        st.markdown("---")
        st.subheader("Affinage du meilleur modèle")

        results = st.session_state.get("clim_comparison_results") or []
        successful_results = [r for r in results if r.get("success")]
        context = st.session_state.get("clim_comparison_context")
        if not successful_results:
            st.info("Lancez d'abord une comparaison de modèles (mode « Comparaison de modèles ») : "
                    "l'affinage part du meilleur modèle trouvé.")
        elif context and context != (target_col, tuple(feature_cols)):
            st.info("La cible ou les variables ont changé depuis la dernière comparaison : "
                    "relancez la comparaison avant d'affiner.")
        else:
            best_result = successful_results[max(range(len(successful_results)), key=lambda i: successful_results[i]["test_score"])]

            # Afficher le score de base avec contexte
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info(f"Modèle sélectionné : **{best_result['model_name']}**")
            with col2:
                st.metric("Score de base", f"{best_result['test_score']:.4f}")
            with col3:
                st.metric("Temps", f"{best_result['training_time']:.2f}s")

            st.caption("Les hyperparamètres du meilleur modèle sont pré-remplis. Modifiez-les puis cliquez sur "
                       "« Affiner le modèle » : le modèle affiné n'est conservé que s'il fait mieux sur le test.")

            st.markdown("---")
            st.markdown("**Hyperparamètres à affiner**")

            model_name = best_result["model_name"]
            current_model = best_result["pipeline"].named_steps.get("model")
            if current_model is None or not tuning.is_tunable(model_name):
                st.info(f"{model_name} n'a pas d'hyperparamètre à affiner.")
            else:
                values = tuning.tuning_widgets(model_name, current_model)
                if st.button("Affiner le modèle"):
                    task_type = (st.session_state.get("clim_model_info") or {}).get("task_type", detected_task)
                    tuned_model = tuning.build_tuned_estimator(current_model, values)
                    _train_tuned_model(df, target_col, task_type, test_size, use_cv,
                                       handle_imbalance, best_result, model_name, tuned_model,
                                       feature_cols, time_col)


def _train_tuned_model(df, target_col, task_type, test_size, use_cv, handle_imbalance, best_result, label, estimator,
                       feature_cols=None, time_col=None) -> None:
    """Réentraîne le meilleur modèle avec les hyperparamètres choisis et compare au score de base."""
    with st.spinner("Réentraînement avec les nouveaux hyperparamètres..."):
        tuned_results, _ = model_comparison.compare_models(
            df,
            target_col=target_col,
            task=task_type,
            test_size=test_size,
            use_cv=use_cv,
            handle_imbalance=handle_imbalance,
            custom_models={label: estimator},
            feature_cols=feature_cols,
            time_col=time_col,
        )

    # Remplacer le modèle par le modèle affiné
    if not (tuned_results and tuned_results[0]["success"]):
        st.error("L'entraînement du modèle affiné a échoué.")
        return
    tuned_result = tuned_results[0]

    # Comparer avec le modèle de base
    st.markdown("---")
    st.subheader("Résultats de l'affinage")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"{tuned_result['metric_name']} (test)",
            f"{tuned_result['test_score']:.4f}",
            delta=f"{tuned_result['test_score'] - best_result['test_score']:.4f}"
        )
    with col2:
        st.metric(
            f"{tuned_result['metric_name']} (apprentissage)",
            f"{tuned_result['train_score']:.4f}",
            delta=f"{tuned_result['train_score'] - best_result['train_score']:.4f}"
        )
    with col3:
        st.metric("Temps", f"{tuned_result['training_time']:.2f}s")

    # Afficher un message selon l'amélioration
    improvement = tuned_result['test_score'] - best_result['test_score']
    if improvement > 0.01:
        st.success(f"Amélioration significative : +{improvement:.4f}")
    elif improvement > 0:
        st.info(f"Légère amélioration : +{improvement:.4f}")
    else:
        st.info(f"Pas d'amélioration sur le jeu de test (écart {improvement:+.4f}) : le modèle "
                f"d'origine, {best_result['model_name']}, est conservé. Essayez d'autres valeurs.")
        return

    # Sauvegarder le modèle affiné (seulement s'il fait au moins aussi bien)
    st.session_state["clim_model"] = tuned_result["pipeline"]
    st.session_state["clim_model_info"] = {
        "task_type": task_type,
        "model_name": f"{tuned_result['model_name']} (Affiné)",
        "metric_name": tuned_result["metric_name"],
        "metric_value": tuned_result["test_score"],
        "target_col": target_col,  # Ajout de la variable cible
        "feature_cols": feature_cols,
        "time_col": time_col,
        # Métriques de classification
        "f1_score": tuned_result.get("f1_score"),
        "precision": tuned_result.get("precision"),
        "recall": tuned_result.get("recall"),
                        "accuracy": tuned_result.get("accuracy"),
                        "roc_auc": tuned_result.get("roc_auc"),
                        "pr_auc": tuned_result.get("pr_auc"),
        # Métriques de régression
        "mse": tuned_result.get("mse"),
        "rmse": tuned_result.get("rmse"),
        "mae": tuned_result.get("mae"),
        "r2": tuned_result.get("r2"),
        "cv_scores": tuned_result["cv_scores"],
        # Informations manquantes ajoutées
        "train_shape": tuned_result.get("train_shape"),
        "test_shape": tuned_result.get("test_shape"),
        "feature_importance": tuned_result.get("feature_importance"),
        "feature_names": tuned_result.get("feature_names"),
    }

    st.success("Modèle affiné sauvegardé !")


def page_evaluation() -> None:
    st.header("Évaluation & Scénarios")
    info = st.session_state.get("clim_model_info")
    if info is None:
        st.warning("Aucun modèle climat n’a encore été entraîné.")
        return

    evaluation.show_evaluation(info)


def page_pricing() -> None:
    """Tarification : de la probabilité d'événement prédite à la prime par unité exposée."""
    st.header("Tarification du risque")
    st.markdown(
        "Prime pure = fréquence modélisée x coût moyen d'un sinistre, ramenée à l'année et à "
        "l'unité exposée (un bien assuré, par exemple). La prime commerciale ajoute les chargements."
    )

    info = st.session_state.get("clim_model_info")
    model = st.session_state.get("clim_model")
    if not info or model is None:
        st.warning("Entraînez d'abord un modèle de classification de l'événement (page Modélisation).")
        return
    event_col = info.get("target_col")
    feature_cols = info.get("feature_cols")
    if info.get("task_type") != "classification" or not hasattr(model, "predict_proba"):
        st.warning("La tarification utilise la probabilité d'événement : entraînez un modèle de classification (cible 0/1).")
        return

    required = list(dict.fromkeys((feature_cols or []) + [event_col]))
    df = _select_data_source("pricing", required_cols=required)
    if df is None or df.empty:
        st.warning("Aucune source chargée ne contient toutes les variables du modèle ("
                   + ", ".join(required) + "). Réentraînez le modèle ou rechargez les mêmes données.")
        return
    df = df.dropna(subset=[event_col])
    if not _is_binary(df[event_col]):
        st.warning(f"La cible `{event_col}` doit être un indicateur 0/1 d'événement.")
        return

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != event_col]
    seg_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and 1 < df[c].nunique() <= 50]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower()]

    def guess(options, keys):
        return next((c for c in options if any(k in c.lower() for k in keys)), None)

    st.subheader("1. Coût des sinistres")
    cost_options = ["(saisir un coût moyen)"] + num_cols
    cost_default = guess(num_cols, ("cout", "coût", "sinistre", "montant", "perte")) or "(saisir un coût moyen)"
    cost_choice = st.selectbox("Colonne du coût des sinistres", cost_options, index=cost_options.index(cost_default))
    cost_col = None if cost_choice.startswith("(") else cost_choice
    average_cost = None
    if cost_col is None:
        average_cost = st.number_input("Coût moyen d'un sinistre", min_value=0.0, value=10000.0, step=1000.0)

    st.subheader("2. Segmentation et exposition")
    col1, col2, col3 = st.columns(3)
    with col1:
        seg_options = ["(aucune)"] + seg_cols
        seg_default = guess(seg_cols, ("zone", "station", "region", "commune")) or "(aucune)"
        seg_choice = st.selectbox("Segment de tarification", seg_options, index=seg_options.index(seg_default))
        segment_col = None if seg_choice == "(aucune)" else seg_choice
    with col2:
        exp_options = ["(aucune)"] + num_cols
        exp_default = guess(num_cols, ("nb_biens", "biens", "exposition", "polices", "contrats")) or "(aucune)"
        exp_choice = st.selectbox("Exposition (unités assurées)", exp_options, index=exp_options.index(exp_default),
                                  help="Nombre de biens ou de contrats du segment : la prime est calculée par unité.")
        exposure_col = None if exp_choice == "(aucune)" else exp_choice
    with col3:
        date_options = ["(aucune)"] + date_cols
        date_choice = st.selectbox("Colonne date (durée d'observation)", date_options,
                                   index=1 if date_cols else 0,
                                   help="Sert à ramener la charge à l'année.")
        date_col = None if date_choice == "(aucune)" else date_choice

    severity_by_segment = False
    credibility_k = 0.0
    if cost_col and segment_col:
        severity_by_segment = st.checkbox("Coût moyen calculé par segment", value=True,
                                          help="Les segments diffèrent souvent fortement en coût (valeur des biens).")
        if severity_by_segment:
            credibility_k = float(st.slider(
                "Crédibilité : paramètre k (0 = coût du segment seul)", 0, 100, 10,
                help="Z = n / (n + k) : un segment avec peu de sinistres (n petit) est rapproché du coût moyen global.",
            ))

    st.subheader("3. Chargements")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        security = st.number_input("Sécurité (%)", 0.0, 100.0, 10.0, 1.0) / 100
    with col2:
        fees = st.number_input("Frais de gestion (%)", 0.0, 60.0, 15.0, 1.0) / 100
    with col3:
        margin = st.number_input("Marge (%)", 0.0, 30.0, 5.0, 1.0) / 100
    with col4:
        recalibrate = st.checkbox("Recaler la fréquence", value=True,
                                  help="Ramène la fréquence modélisée moyenne à la fréquence observée. "
                                       "Indispensable si le modèle a été entraîné avec la gestion du déséquilibre.")

    try:
        loadings = pricing.Loadings(security=security, fees=fees, margin=margin)
        proba = model.predict_proba(df[feature_cols])[:, 1]
        table = pricing.price_portfolio(
            df, proba, event_col, cost_col=cost_col, average_cost=average_cost,
            segment_col=segment_col, exposure_col=exposure_col, date_col=date_col,
            severity_by_segment=severity_by_segment, loadings=loadings,
            recalibrate=recalibrate, credibility_k=credibility_k,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    totals = pricing.portfolio_totals(table)
    st.markdown("---")
    st.subheader("Résultats")
    cols = st.columns(4)
    cols[0].metric("Charge annuelle attendue", f"{totals['charge_attendue']:,.0f}".replace(",", " "))
    if totals.get("charge_observee") is not None:
        cols[1].metric("Charge annuelle observée", f"{totals['charge_observee']:,.0f}".replace(",", " "))
    if totals.get("prime_pure_moyenne") is not None:
        cols[2].metric("Prime pure moyenne par unité", f"{totals['prime_pure_moyenne']:,.2f}".replace(",", " "))
        cols[3].metric("Prime commerciale moyenne", f"{totals['prime_commerciale_moyenne']:,.2f}".replace(",", " "))
    st.caption(f"Facteur de chargement commercial : x{loadings.commercial_factor():.3f} "
               f"= (1 + {security:.0%}) / (1 - {fees:.0%} - {margin:.0%}).")

    money = st.column_config.NumberColumn(format="%.0f")
    unit_money = st.column_config.NumberColumn(format="%.2f")
    pct = st.column_config.NumberColumn(format="percent")
    config = {
        "Fréquence observée": pct, "Fréquence modélisée": pct,
        "Coût moyen d'un sinistre": money, "Charge annuelle attendue": money,
        "Charge annuelle observée": money, "Exposition": st.column_config.NumberColumn(format="%.0f"),
        "Crédibilité Z": st.column_config.NumberColumn(format="%.2f"),
        "Ratio observé / attendu": st.column_config.NumberColumn(format="%.2f"),
        "Prime pure par unité": unit_money, "Prime commerciale par unité": unit_money,
    }
    st.dataframe(table, width="stretch", column_config=config)
    if "Ratio observé / attendu" in table:
        note = ("Ratio observé / attendu proche de 1 : la prime couvre la sinistralité historique du segment. "
                "Au-dessus de 1, le segment est sous-tarifé ; en dessous, sur-tarifé.")
        if credibility_k > 0:
            note += (" Avec la crédibilité, les segments ayant peu de sinistres sont rapprochés du coût moyen global : "
                     "leur ratio s'écarte de 1 par prudence, c'est voulu.")
        st.caption(note)
    chart_data = table.reset_index().rename(columns={table.index.name: "Segment"})
    chart = (
        alt.Chart(chart_data)
        .mark_bar(color="#FFD700")
        .encode(
            x=alt.X("Prime commerciale par unité:Q", title="Prime commerciale annuelle par unité"),
            y=alt.Y("Segment:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=250)),
            tooltip=["Segment", alt.Tooltip("Prime pure par unité:Q", format=".2f"),
                     alt.Tooltip("Prime commerciale par unité:Q", format=".2f")],
        )
        .properties(height=max(200, 32 * len(chart_data)), title="Prime commerciale par segment")
    )
    st.altair_chart(chart, width="stretch")

    st.download_button(
        "Télécharger la tarification (CSV)",
        table.round(4).to_csv().encode("utf-8"),
        file_name="tarification.csv",
        mime="text/csv",
        on_click="ignore",
    )
    st.session_state["clim_pricing"] = {
        "table": table,
        "totals": totals,
        "loadings": {"securite": security, "frais": fees, "marge": margin},
        "segment_col": segment_col,
        "exposure_col": exposure_col,
    }


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
    maps.run_maps_page(df, title="", data_sources=data_sources, df_prep=df_prep)


def page_reporting() -> None:
    """Page de reporting avec le nouveau design exact du rapport exemple"""
    st.header("Reporting Climat")

    # Utilisation du nouveau module de reporting
    reporting.show_climate_reporting_summary(st.session_state)


if __name__ == "__main__":  # pragma: no cover
    main()
