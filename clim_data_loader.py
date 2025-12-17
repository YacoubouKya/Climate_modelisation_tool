"""Chargement des données pour Data Tool Climatique.

Ce module fournit des fonctions simples pour charger des fichiers CSV/Excel
contenant des données climatiques, d’exposition ou d’événements.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def load_tabular_file(uploaded_file, sep: str = ",", sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Charge un fichier CSV ou Excel uploadé par l’utilisateur.

    Paramètres
    ----------
    uploaded_file : objet retourné par st.file_uploader
    sep : str
        Séparateur pour les fichiers CSV.
    sheet_name : str, optional
        Nom de la feuille pour les fichiers Excel.
    """

    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=sep)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            raise ValueError("Format de fichier non supporté (attendu: CSV, XLS, XLSX)")
    except Exception as exc:  # pragma: no cover - géré au niveau de l’app
        # On laisse l’app Streamlit gérer l’affichage de l’erreur.
        print(f"Erreur lors du chargement du fichier climat : {exc}")
        return None

    return df
