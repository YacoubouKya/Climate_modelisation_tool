"""Génère un jeu de données fictif pour tester le Data Tool Climatique.

Trois sources, à charger séparément dans l'application puis à fusionner :
- climat.csv       : relevés quotidiens par zone (2022-2024)
- exposition.csv   : portefeuille assuré par zone
- evenements.csv   : inondations observées et coût des sinistres

La probabilité d'inondation dépend de la pluie cumulée sur 3 jours, de
l'altitude et du caractère urbain de la zone ; le coût dépend de la valeur
assurée. Les données sont entièrement simulées.

Usage : python generer_donnees.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
OUT = Path(__file__).parent

ZONES = pd.DataFrame({
    "zone": ["Lome", "Aneho", "Tsevie", "Atakpame", "Sokode", "Kara", "Dapaong", "Kpalime"],
    "latitude": [6.137, 6.228, 6.426, 7.533, 8.983, 9.551, 10.862, 6.900],
    "longitude": [1.212, 1.592, 1.213, 1.126, 1.133, 1.186, 0.207, 0.633],
    "altitude": [10, 8, 60, 400, 400, 340, 290, 250],
    "type_zone": ["urbain", "urbain", "rural", "rural", "urbain", "urbain", "rural", "rural"],
    "nb_biens": [5200, 1300, 900, 1100, 1800, 2100, 700, 950],
    "valeur_assuree_moyenne": [42000, 30000, 22000, 24000, 26000, 28000, 18000, 21000],
})


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    rows = []
    for _, z in ZONES.iterrows():
        doy = dates.day_of_year.values
        # Deux saisons des pluies au sud, une au nord
        saison = np.sin((doy - 90) / 365 * 2 * np.pi) + (0.6 * np.sin((doy - 230) / 365 * 4 * np.pi) if z.latitude < 8 else 0)
        intensite = np.clip(saison, 0, None) * (1.4 if z.latitude < 8 else 1.0)
        pluie = rng.gamma(0.6 + 1.5 * intensite, 6.0) * (rng.random(len(dates)) < 0.25 + 0.45 * intensite)
        temperature = 27 + 2.5 * np.sin((doy - 60) / 365 * 2 * np.pi) - z.altitude / 200 + rng.normal(0, 1.2, len(dates))
        humidite = np.clip(55 + 30 * intensite + rng.normal(0, 6, len(dates)), 20, 100)
        rows.append(pd.DataFrame({
            "date": dates, "zone": z.zone, "latitude": z.latitude, "longitude": z.longitude,
            "temperature": temperature.round(1), "precipitation": pluie.round(1), "humidite": humidite.round(0),
        }))
    climat = pd.concat(rows, ignore_index=True)

    # Evenements : logit de la probabilite d'inondation
    df = climat.merge(ZONES, on=["zone", "latitude", "longitude"])
    pluie_3j = df.groupby("zone")["precipitation"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    logit = -5.0 + 0.07 * pluie_3j - 0.005 * df["altitude"] + 0.9 * (df["type_zone"] == "urbain")
    proba = 1 / (1 + np.exp(-logit))
    inondation = (rng.random(len(df)) < proba).astype(int)
    taux_dommage = rng.lognormal(mean=-3.2, sigma=0.8, size=len(df))
    biens_touches = rng.poisson(0.004 * df["nb_biens"]) + 1
    cout = np.where(inondation == 1, (biens_touches * df["valeur_assuree_moyenne"] * taux_dommage).round(0), 0)
    evenements = pd.DataFrame({"date": df["date"], "zone": df["zone"], "inondation": inondation, "cout_sinistre": cout})

    climat.to_csv(OUT / "climat.csv", index=False)
    ZONES.drop(columns=["latitude", "longitude"]).to_csv(OUT / "exposition.csv", index=False)
    evenements.to_csv(OUT / "evenements.csv", index=False)
    print(f"climat : {len(climat)} lignes | exposition : {len(ZONES)} zones | "
          f"inondations : {evenements.inondation.sum()} ({evenements.inondation.mean():.2%} des jours), "
          f"coût moyen par sinistre : {evenements.loc[evenements.inondation == 1, 'cout_sinistre'].mean():,.0f}")


if __name__ == "__main__":
    main()
