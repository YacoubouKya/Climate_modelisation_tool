"""Génère un jeu de données fictif pour tester le Data Tool Climatique.

Quatre sources, à charger séparément dans l'application (types Climat, Géographie,
Exposition et Événements) puis à fusionner :
- climat.csv       : relevés quotidiens de huit communes françaises (2022-2024)
- geographie.csv   : cinq quartiers (sites) par commune, avec coordonnées, altitude,
                     distance au cours d'eau et situation en zone inondable
- exposition.csv   : portefeuille assuré par site (nombre de biens, valeur moyenne)
- evenements.csv   : inondations observées par site et par jour, et leur coût

La fusion se fait sur les colonnes communes : zone (commune), site et date.
La probabilité d'inondation d'un site dépend de la pluie cumulée sur 3 jours dans
sa commune, de son altitude, de sa distance au cours d'eau, de sa situation en zone
inondable et du caractère urbain de la commune ; le coût dépend du nombre de biens
et de leur valeur. Les montants sont en euros. Les communes sont réelles ; les
quartiers, les relevés et le portefeuille sont inventés.

Usage : python generer_donnees.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
OUT = Path(__file__).parent
SITES_PAR_COMMUNE = 5

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
            "date": dates, "zone": z.zone,
            "temperature": temperature.round(1), "precipitation": pluie.round(1), "humidite": humidite.round(0),
        }))
    climat = pd.concat(rows, ignore_index=True)


    # Sites : quelques quartiers par commune, avec leurs caractéristiques géographiques
    sites = []
    for _, z in ZONES.iterrows():
        for k in range(1, SITES_PAR_COMMUNE + 1):
            distance = float(np.clip(rng.lognormal(np.log(700), 0.9), 30, 6000))
            inondable = int(distance < 300 or (distance < 900 and rng.random() < 0.35))
            sites.append({
                "site": f"{z.zone}-{k:02d}",
                "zone": z.zone,
                "latitude": round(z.latitude + rng.uniform(-0.035, 0.035), 4),
                "longitude": round(z.longitude + rng.uniform(-0.05, 0.05), 4),
                "altitude": int(max(1, z.altitude + rng.normal(0, 12) + distance / 120)),
                "distance_cours_eau_m": int(round(distance, -1)),
                "zone_inondable": inondable,
                "type_zone": z.type_zone,
            })
    geographie = pd.DataFrame(sites)

    # Portefeuille par site : biens de la commune répartis entre ses quartiers
    parts = []
    for _, z in ZONES.iterrows():
        poids = rng.dirichlet(np.full(SITES_PAR_COMMUNE, 4.0))
        for k, w in enumerate(poids, 1):
            parts.append({
                "site": f"{z.zone}-{k:02d}",
                "nb_biens": int(max(20, round(z.nb_biens * w))),
                "valeur_assuree_moyenne": int(round(z.valeur_assuree_moyenne * rng.lognormal(0, 0.15), -3)),
            })
    exposition = pd.DataFrame(parts)

    # Evénements par site et par jour : la pluie de la commune, modulée par la géographie du site
    df = climat.merge(geographie, on="zone").merge(exposition, on="site")
    df = df.sort_values(["site", "date"])
    pluie_3j = df.groupby("site")["precipitation"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    logit = (-6.4 + 0.055 * pluie_3j - 0.004 * df["altitude"] + 1.1 * df["zone_inondable"]
             - 0.45 * np.log(df["distance_cours_eau_m"] / 700) + 0.4 * (df["type_zone"] == "urbain"))
    proba = 1 / (1 + np.exp(-logit))
    inondation = (rng.random(len(df)) < proba).astype(int)
    taux_dommage = rng.lognormal(mean=-3.6, sigma=0.9, size=len(df))
    biens_touches = rng.poisson(0.01 * df["nb_biens"] * (1 + df["zone_inondable"])) + 1
    cout = np.where(inondation == 1, (biens_touches * df["valeur_assuree_moyenne"] * taux_dommage).round(0), 0)
    evenements = pd.DataFrame({"date": df["date"].values, "site": df["site"].values,
                               "inondation": inondation, "cout_sinistre": cout})

    climat.to_csv(OUT / "climat.csv", index=False)
    geographie.to_csv(OUT / "geographie.csv", index=False)
    exposition.to_csv(OUT / "exposition.csv", index=False)
    evenements.to_csv(OUT / "evenements.csv", index=False)
    print(f"climat : {len(climat)} lignes | sites : {len(geographie)} "
          f"({geographie.zone_inondable.sum()} en zone inondable) | "
          f"inondations : {evenements.inondation.sum()} ({evenements.inondation.mean():.2%} des jours-sites), "
          f"coût moyen par sinistre : {evenements.loc[evenements.inondation == 1, 'cout_sinistre'].mean():,.0f} EUR")


if __name__ == "__main__":
    main()
