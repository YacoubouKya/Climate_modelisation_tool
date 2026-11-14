"""Génère un jeu de données fictif pour tester le Data Tool Climatique.

Trois sources, à charger séparément dans l'application puis à fusionner :
- climat.csv       : relevés quotidiens de huit communes françaises (2022-2024)
- exposition.csv   : portefeuille assuré par zone
- evenements.csv   : inondations observées et coût des sinistres

La probabilité d'inondation dépend de la pluie cumulée sur 3 jours, de
l'altitude et du caractère urbain de la zone ; le coût dépend de la valeur
assurée. Les montants sont en euros. Les données sont entièrement simulées
(les communes sont réelles, les relevés et le portefeuille sont inventés).

Usage : python generer_donnees.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
OUT = Path(__file__).parent

# Huit communes françaises aux régimes de pluie contrastés. Le portefeuille
# (nombre de biens, valeurs assurées) est inventé.
ZONES = pd.DataFrame({
    "zone": ["Paris", "Lyon", "Nimes", "Montpellier", "Nantes", "Strasbourg", "Quimperle", "Sommieres"],
    "latitude": [48.857, 45.764, 43.837, 43.611, 47.218, 48.573, 47.873, 43.784],
    "longitude": [2.352, 4.836, 4.360, 3.877, -1.554, 7.752, -3.548, 4.090],
    "altitude": [35, 170, 40, 27, 8, 142, 30, 35],
    "type_zone": ["urbain", "urbain", "urbain", "urbain", "urbain", "urbain", "rural", "rural"],
    "regime": ["oceanique", "continental", "mediterraneen", "mediterraneen", "oceanique", "continental",
               "oceanique", "mediterraneen"],
    "nb_biens": [6800, 3900, 2100, 2600, 2400, 2200, 650, 480],
    "valeur_assuree_moyenne": [310000, 240000, 210000, 230000, 220000, 215000, 170000, 180000],
})

# Par régime : probabilité de pluie de base, amplitude saisonnière, mois du pic,
# échelle des cumuls et probabilité d'un épisode intense (pluies méditerranéennes d'automne).
REGIMES = {
    "oceanique": dict(p=0.42, amp=0.12, pic=12, echelle=4.5, intense=0.004),
    "continental": dict(p=0.33, amp=0.06, pic=6, echelle=5.0, intense=0.006),
    "mediterraneen": dict(p=0.15, amp=0.07, pic=10, echelle=5.0, intense=0.011),
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    doy = dates.day_of_year.values
    rows = []
    for _, z in ZONES.iterrows():
        r = REGIMES[z.regime]
        saison = np.cos((doy - (r["pic"] - 0.5) * 30.4) / 365 * 2 * np.pi)
        p_pluie = np.clip(r["p"] + r["amp"] * saison, 0.05, 0.9)
        pluie = rng.gamma(0.8, r["echelle"], len(dates)) * (rng.random(len(dates)) < p_pluie)
        # Episodes intenses concentrés à l'automne pour le régime méditerranéen
        p_intense = r["intense"] * (1 + 2 * np.clip(saison, 0, None)) if z.regime == "mediterraneen" else r["intense"]
        pluie = pluie + (rng.random(len(dates)) < p_intense) * rng.gamma(2.5, 30.0, len(dates))
        temperature = (12.5 + (9 if z.regime == "continental" else 7.5) * np.sin((doy - 110) / 365 * 2 * np.pi)
                       + (3 if z.regime == "mediterraneen" else 0) - z.altitude / 150 + rng.normal(0, 2.0, len(dates)))
        humidite = np.clip(72 + 10 * (pluie > 0) + 6 * np.cos((doy - 15) / 365 * 2 * np.pi)
                           - (8 if z.regime == "mediterraneen" else 0) + rng.normal(0, 6, len(dates)), 25, 100)
        rows.append(pd.DataFrame({
            "date": dates, "zone": z.zone, "latitude": z.latitude, "longitude": z.longitude,
            "temperature": temperature.round(1), "precipitation": pluie.round(1), "humidite": humidite.round(0),
        }))
    climat = pd.concat(rows, ignore_index=True)

    # Evénements : logit de la probabilité d'inondation
    df = climat.merge(ZONES.drop(columns="regime"), on=["zone", "latitude", "longitude"])
    pluie_3j = df.groupby("zone")["precipitation"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    logit = -5.3 + 0.055 * pluie_3j - 0.004 * df["altitude"] + 0.5 * (df["type_zone"] == "urbain")
    proba = 1 / (1 + np.exp(-logit))
    inondation = (rng.random(len(df)) < proba).astype(int)
    taux_dommage = rng.lognormal(mean=-3.6, sigma=0.9, size=len(df))
    biens_touches = rng.poisson(0.004 * df["nb_biens"]) + 1
    cout = np.where(inondation == 1, (biens_touches * df["valeur_assuree_moyenne"] * taux_dommage).round(0), 0)
    evenements = pd.DataFrame({"date": df["date"], "zone": df["zone"], "inondation": inondation, "cout_sinistre": cout})

    climat.to_csv(OUT / "climat.csv", index=False)
    ZONES.drop(columns=["latitude", "longitude", "regime"]).to_csv(OUT / "exposition.csv", index=False)
    evenements.to_csv(OUT / "evenements.csv", index=False)
    print(f"climat : {len(climat)} lignes | exposition : {len(ZONES)} zones | "
          f"inondations : {evenements.inondation.sum()} ({evenements.inondation.mean():.2%} des jours), "
          f"coût moyen par sinistre : {evenements.loc[evenements.inondation == 1, 'cout_sinistre'].mean():,.0f} EUR")


if __name__ == "__main__":
    main()
