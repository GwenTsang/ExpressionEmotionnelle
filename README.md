# ExpressionEmotionnelle

Corpus annotés et pipeline d'analyse pour l'étude des expressions émotionnelles (format Glozz).

## Contenu du dépôt


**Corpus** : Albert_dataset/, CorpusCovid/, LitteratureJeunesse/, PtitLibe/  
  Chaque sous-dossier attendu contient deux sous-dossiers `aa/` (annotations .aa XML) et `ac/` (texte brut .ac).
  
- glozz_parser.py — parse les fichiers `.aa` et `.ac` et produit `output/annotations.csv`.
- marker_extraction.py — extrait mots / ponctuations et (optionnellement) lemmes ; produit `output/markers.csv`.
- marker_specificity.py — calcule P(Classe|Marqueur), entropies (H), tests statistiques et exports dans `output/specificity_results/`.
- run_analysis.py — orchestrateur pour lancer les étapes (`parse`, `markers`, `specificity` ou `all`).
- top_markers.py — lit les CSV d'entropie et extrait/affiche les meilleurs marqueurs par classe/mode.
- output/ — dossier cible pour les CSV et les résultats.

## Résultat attendu (fichiers produits)

- output/annotations.csv  
  Colonnes produites (par glozz_parser) : corpus, file_id, unit_id, type, start_idx, end_idx, text_span, mode, categorie1, categorie2, remarque

- output/markers.csv  
  Dataframe "marqueur × annotation" (par marker_extraction) : colonnes standards + marker_type (word|lemma|punctuation) et marker_value

- output/specificity_results/entropy_per_marker_emotion.csv  
  Entropie et probabilités conditionnelles P(Emotion|marqueur) (cols P(...)), total_count, entropy, normalized_entropy, etc.

- output/specificity_results/entropy_per_marker_mode.csv  
  Entropie et probabilités pour les modes d'expression.

- output/specificity_results/entropy_by_mode_summary.csv  
  Résumé par mode (moyenne, médiane, écart-type).

- output/specificity_results/hypothesis_report.txt  
  Rapport texte détaillé du test d'hypothèse (Kruskal-Wallis, Mann-Whitney, statistiques descriptives).

- output/specificity_results/top_markers_*.csv  
  Exports produits par top_markers.py si demandé.

## Dépendances

- Python 3.8+
- pandas, numpy, scipy
- (optionnel pour lemmatisation) spaCy OR stanza
  - spaCy (recommandé pour CPU) :
    - pip install spacy
    - python -m spacy download fr_core_news_sm
  - stanza (GPU, CUDA requis) :
    - pip install stanza
    - pip install torch  (avec support CUDA)
    - stanza.download("fr")  (le script tente le téléchargement automatiquement)
- logging, argparse (stdlib)


Remarque importante sur Stanza : le backend stanza du script impose l'utilisation d'un GPU CUDA ; si vous n'avez pas de GPU CUDA, utilisez spaCy (`--lemmatizer spacy` ou laissez le réglage par défaut).

## Exemples d'utilisation

1) Pipeline complet (tous les corpus) — lemmatisation spaCy (par défaut) :

python run_analysis.py
  
Cela exécute successivement : parsing → extraction des marqueurs → calcul de spécificité. Les résultats sont écrits dans `output/`.

2) Exécuter une seule étape
   
- Parsing uniquement :
  - python run_analysis.py --step parse
    
- Extraction des marqueurs (à partir d'un annotations.csv existant) :
  - python run_analysis.py --step markers
    
- Spécificité uniquement (à partir d'un markers.csv existant) :
  - python run_analysis.py --step specificity


- Utiliser Stanza (GPU CUDA requis) :
  - python run_analysis.py --lemmatizer stanza
    
- Inclure unités avec features vides (Mode/Remarque vides) :
  - python run_analysis.py --include-empt
    
- Changer le dossier de sortie :
  - python run_analysis.py --output-dir results_dir

4) Appels directs aux scripts
- Générer annotations depuis les dossiers de corpus :
  - python glozz_parser.py --output output/annotations.csv
- Extraire marqueurs :
  - python marker_extraction.py -i output/annotations.csv -o output/markers.csv --lemmatizer spacy --batch-size 256
- Calculer spécificité :
  - python marker_specificity.py -i output/markers.csv -o output/specificity_results --min-freq 5
- Obtenir les top marqueurs :
  - python top_markers.py --emotion output/specificity_results/entropy_per_marker_emotion.csv --top 20

## Paramètres principaux (rapide)

- --include-empty : inclut les unités SitEmo/Autre même si certaines features (Mode/Remarque) sont vides.
- --no-lemma / --lemmatizer : contrôle la lemmatisation ; spaCy fonctionne en CPU, stanza nécessite GPU CUDA.
- --batch-size : taille du batch pour la lemmatisation (utile si gros corpus).
- --remove-stopwords : (marker_extraction) filtre une liste de stopwords FR très fréquents.
- --min-freq : fréquence minimale pour inclure un marqueur dans les calculs d'entropie.

## Notes sur les données et le format Glozz

- Les annotations attendues suivent le format Glozz (.aa pour XML d'annotations, .ac pour texte brut).
- Chaque unité annotée (unit) est lue avec ses positions start/end (offsets) pour extraire le segment textuel.
- Les scripts ciblent les types d'unités "SitEmo" et "Autre". Pour SitEmo, les features attendues incluent Mode, Categorie, Categorie2 ; pour Autre la feature Remarque est exploitée.

## Conseils et dépannage

- Si `glozz_parser.py` signale des fichiers .ac manquants, vérifiez que chaque `.aa` a son `.ac` correspondant dans le dossier `ac/`.
- Pour spaCy, si le modèle fr_core_news_sm n'est pas trouvé :
  - python -m spacy download fr_core_news_sm
- Pour Stanza, vérifiez `torch.cuda.is_available()` avant d'essayer le backend stanza ; sinon utilisez spaCy.
- En cas de lenteur : augmenter `--batch-size` pour spaCy (jusqu'à la mémoire disponible) ; pour stanza, attention à la VRAM GPU.

## Expérimentation et interprétation

- Les fichiers d'entropie contiennent des colonnes `P(...)` pour les probabilités conditionnelles P(Classe | Marqueur). Les marqueurs avec faible entropie sont plus ciblés (déterminés) pour une classe donnée.
- Le rapport d'hypothèse fournit un test global (Kruskal‑Wallis) et une comparaison unilatérale (Mann‑Whitney U) entre les groupes Montree+Suggeree vs Designee+Comportementale, avec mesures d'effet.
