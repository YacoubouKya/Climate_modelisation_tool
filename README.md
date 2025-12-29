# Data Tool Climatique

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Outil d'analyse du risque climatique pour le hackathon, permettant le traitement, l'analyse et la modélisation de données climatiques dans un contexte d'évaluation des risques.

##  Fonctionnalités

- **Chargement** de données climatiques et d'exposition (stations météo, séries temporelles, événements extrêmes)
- **Prétraitement** adapté aux données climatiques (agrégations temporelles, jointures spatiales, gestion des unités)
- **Analyse exploratoire** avec visualisations spécifiques (séries temporelles, cartes, heatmaps)
- **Modélisation** avec comparaison de modèles adaptés aux risques climatiques
- **Rapports** automatisés orientés décision

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/data-tool-climatique.git
   cd data-tool-climatique
   ```

2. Créez et activez un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: .\venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

Lancez l'application Streamlit :
```bash
streamlit run modules/clim_app.py
```

##  Structure du projet

```
data/                   # Données d'entrée
├── raw/               # Données brutes
└── processed/         # Données prétraitées

modules/               # Code source de l'application
├── clim_app.py        # Application principale Streamlit
├── clim_data_loader.py # Chargement des données
├── clim_preprocessing.py # Prétraitement
├── clim_modeling.py   # Modélisation
├── clim_evaluation.py # Évaluation des modèles
└── clim_reporting.py  # Génération de rapports

outputs/               # Sorties générées
├── models/            # Modèles entraînés
└── reports/           # Rapports générés

docs/                  # Documentation
```

## Fonctionnalités techniques

- **Exploration des données** : Visualisations interactives des séries temporelles et cartes
- **Prétraitement** : Gestion des valeurs manquantes, agrégations temporelles, création de features
- **Modélisation** : Comparaison de plusieurs algorithmes de machine learning
- **Évaluation** : Métriques adaptées au risque climatique
- **Rapports** : Génération automatisée de rapports HTML

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -am 'Ajout d\'une nouvelle fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créez une Pull Request

##  Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

##  Contact

Pour toute question, veuillez ouvrir une issue sur le dépôt.

