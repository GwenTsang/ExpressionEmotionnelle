# Expression Émotionnelle : Analyse des Marqueurs et Spécificité

> Corpus annotés et pipeline d'analyse pour l'étude linguistique et statistique des expressions émotionnelles.

Ce répertoire contient l'ensemble des scripts et des résultats permettant l'extraction, le traitement et l'évaluation statistique de marqueurs d'expressions émotionnelles. Il supporte deux formats d'annotations : le format XML issu de l'outil **Glozz** et le format tabulaire **XLSX**.

## Structure du Dépôt

L'architecture du projet est conçue de manière modulaire, séparant clairement les données, les résultats statistiques et le code source.

```
ExpressionEmotionnelle/
│
├── data/                  # Données du projet
│   ├── raw/               # Corpus bruts initiaux
│   │   ├── glozz/         # Annotations au format XML (Glozz)
│   │   └── xlsx/          # Corpus tabulaires (Excel)
│   └── processed/         # Données nettoyées ou transformées
│
├── results/               # Résultats des analyses
│   ├── glozz/             # Résultats spécifiques aux corpus Glozz
│   └── xlsx/              # Résultats spécifiques aux corpus XLSX
│
└── scripts/               # Scripts Python et orchestrateurs
    ├── run_analysis.py          # Orchestrateur pour le pipeline Glozz
    ├── run_analysis_xlsx.py     # Orchestrateur pour le pipeline XLSX
    ├── extract_markers_*.py     # Extraction des marqueurs depuis Glozz/XLSX
    ├── glozz_parser.py          # Parseur XML pour le format Glozz
    ├── marker_specificity.py    # Calcul statistique (entropie conditionnelle)
    ├── nlp_utils.py             # Fonctions partagées de NLP (Lemmatisation, tokenisation)
    └── top_markers.py           # Analyse des marqueurs les plus fréquents/spécifiques
```

## Pipelines d'Analyse

Le projet est divisé en deux flux de travail principaux selon la source des données :

### 1. Pipeline Glozz (`run_analysis.py`)
Ce script orchestre l'analyse complète des corpus Glozz selon les étapes suivantes :
1. **Parsing :** Extraction des annotations depuis les fichiers Glozz via `glozz_parser.py`.
2. **Extraction des marqueurs :** Identification et lemmatisation des marqueurs linguistiques via `extract_markers_glozz.py`.
3. **Calcul de spécificité :** Évaluation statistique de chaque marqueur en fonction de la catégorie émotionnelle et de son mode d'expression via `marker_specificity.py`.

### 2. Pipeline XLSX (`run_analysis_xlsx.py`)
Adapté pour les données tabulaires :
1. **Lecture et Extraction :** Lit l'ensemble des fichiers Excel présents dans `data/raw/xlsx/` et en extrait les marqueurs (`extract_markers_xlsx.py`).
2. **Standardisation :** Formate les catégories binaires XLSX vers un format compatible "Glozz".
3. **Calcul de spécificité :** Lance les calculs d'entropie pour mesurer la pertinence des marqueurs.

## Traitement du Langage Naturel (NLP)

L'extraction des marqueurs repose sur des techniques avancées de NLP, centralisées dans le module `nlp_utils.py`. 
Le pipeline supporte `SpaCy` et `Stanza`.

Les traitements incluent : tokenisation, lemmatisation, et le filtrage des stopwords.

## Évaluation Statistique et Analyse des Résultats

Le script `marker_specificity.py` mesure la force de l'association entre un marqueur linguistique, une émotion spécifique, ou un mode d'expression. 
L'approche méthodologique repose sur plusieurs concepts et tests statistiques rigoureux :

* **Entropie de Shannon :** Utilisée pour calculer l'entropie conditionnelle (H(Emotion|Marqueur) et H(Mode|Marqueur)). Une entropie faible indique qu'un marqueur est hautement spécifique (déterminé) à une émotion ou un mode particulier. À l'inverse, une entropie élevée indique une forte dispersion.
* **Test de Kruskal-Wallis :** Test non paramétrique appliqué pour vérifier s'il existe des différences significatives de dispersion des marqueurs entre les 4 modes d'expression de manière globale.
* **Test de Mann-Whitney U :** Test non paramétrique bilatéral (pour les comparaisons par paires) et unilatéral pour évaluer l'hypothèse principale selon laquelle certains modes s'appuient sur un vocabulaire plus hétérogène que d'autres.

### Aperçu des Résultats

Les analyses menées sur les corpus mettent en évidence des différences structurelles majeures selon le mode d'expression émotionnel :

1. **Dispersion lexicale supérieure pour les modes "Montrée" et "Suggérée" :** L'hypothèse selon laquelle ces modes présentent une entropie (dispersion lexicale) significativement plus élevée que les modes "Désignée" et "Comportementale" est **très fortement supportée** ($p < 0.001$).
2. **Spécificité des modes explicites :** Les modes "Comportementale" et "Désignée" s'appuient sur des marqueurs linguistiques plus contraints et spécifiques (entropie moyenne de ~1.17 et ~1.16). En revanche, le mode "Montrée" utilise un registre lexical beaucoup plus diversifié et ambigu (entropie moyenne de ~1.66).
3. **Comparaisons par paires :** Les tests révèlent des différences de dispersion hautement significatives ($p < 0.001$) entre le mode "Montrée" et l'ensemble des autres modes.

Les résultats complets, scores détaillés et les rapports de tests générés sont exportés dans le dossier `results/`.

## Utilisation

Les deux pipelines principaux s'exécutent en ligne de commande avec de multiples paramètres pour ajuster le comportement du traitement.

### Analyse Globale (Glozz)
```bash
# Exécuter le pipeline complet (parsing, extraction, spécificité) avec SpaCy
python scripts/run_analysis.py

# Exécuter uniquement une étape spécifique (ex: calcul d'entropie)
python scripts/run_analysis.py --step specificity

# Analyser un corpus en particulier avec Stanza, filtre des stopwords et un seuil de fréquence
python scripts/run_analysis.py --corpus CorpusCovid --lemmatizer stanza --remove-stopwords --min-freq 5
```

### Analyse Tabulaire (XLSX)
```bash
# Lancer l'analyse complète sur tous les fichiers XLSX bruts
python scripts/run_analysis_xlsx.py

# Analyser sans lemmatisation avec un seuil de fréquence modifié
python scripts/run_analysis_xlsx.py --no-lemma --min-freq 10
```

Pour consulter l'ensemble des arguments disponibles, vous pouvez utiliser la commande help :
```bash
python scripts/run_analysis.py --help
python scripts/run_analysis_xlsx.py --help
```
