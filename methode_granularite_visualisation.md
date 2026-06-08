# Methode pour analyser la granularite forme/lemme par visualisation

## Objectif global

L'objectif principal n'est pas seulement de determiner si la lemmatisation enrichit ou degrade l'analyse de specificite. L'objectif est de comprendre comment le changement de granularite entre formes observees, lemmes, corrections lexicales et entrees canoniques modifie la lecture emotionnelle des marqueurs.

L'hypothese centrale est la suivante :

> Un lexeme ou une forme observee `X` peut etre fortement associee a une emotion `Y`, tandis que le lemme `Lemme(X)` peut ne pas etre aussi fortement associe a cette meme emotion, parce qu'il regroupe plusieurs formes, contextes ou usages emotionnels.

Inversement, un lemme peut devenir fortement associe a une emotion alors qu'aucune forme de surface prise isolement ne passe le seuil de frequence ou ne parait suffisamment stable.

La methode doit donc distinguer au moins trois situations :

| Situation | Interpretation |
|---|---|
| La forme et le lemme sont specifiques a la meme emotion | La lemmatisation renforce ou stabilise un signal deja present. |
| La forme est specifique, mais le lemme est disperse | La lemmatisation dilue une specificite locale. |
| Le lemme est specifique, mais les formes seules ne le sont pas | La lemmatisation revele une famille morphologique ou lexicale pertinente. |

La visualisation, notamment sous forme de graphe, doit permettre de conserver simultanement les deux niveaux : le centre lexical canonique et les formes satellites qui peuvent avoir leurs propres specificites.

## Motivation specifique pour le mode `Designee`

Les resultats precedents suggerent que la lemmatisation est particulierement pertinente pour le mode `Designee`, car ce mode correspond souvent a une expression lexicale explicite de l'emotion : noms d'emotions, adjectifs emotionnels, verbes d'etat ou verbes d'expression.

Dans ce contexte, la lemmatisation peut servir a construire un lexique emotionnel structure. Une representation possible est un graphe dans lequel :

- un noeud central represente un lemme ou une entree canonique ;
- des noeuds satellites representent les formes de surface observees ;
- d'autres noeuds representent des corrections, entrees du lexique emotionnel, categories emotionnelles ou modes ;
- les aretes portent des poids : frequence, entropie, emotion dominante, backend, confiance de correction.

Cette structure permettrait de dire par exemple :

- le lemme `inquiéter` organise plusieurs formes liees a la peur ;
- une forme specifique autour de ce lemme peut etre plus distinctive que le lemme lui-meme ;
- certaines formes satellites doivent etre conservees comme indices distinctifs plutot que fusionnees sans nuance.

## Methode Python deja executee

Cette section documente precisement les commandes et analyses Python executees lors du diagnostic precedent. Elle doit servir de point de depart reproductible.

### Donnees analysees

Fichier d'entree :

```bash
data/pipeline_2/SimpleSitEmo.parquet
```

Dimensions observees :

```text
7302 lignes
8 colonnes
```

Colonnes principales :

```text
source_file
text_span
text_span_source
mode
emotion1
emotion2
emotion3
nature_linguistique
```

### Run 1 : SpaCy avec stopwords filtres

Commande executee :

```bash
python -m pipeline_2.run_analysis \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output-dir results/pipeline_2_compare_spacy \
  --lemmatizer spacy \
  --min-freq 3
```

Le filtrage des stopwords est actif par defaut dans `run_analysis.py`, tant que l'option `--keep-stopwords` n'est pas fournie.

Resultats principaux :

```text
Marqueurs extraits : 20 602
word : 9 551
lemma : 8 826
punctuation : 2 225
Marqueurs freq >= 3 pour H(emotion|marqueur) : 1 660
Entropie emotionnelle moyenne : 0.9287
```

### Run 2 : Stanza avec stopwords filtres

Commande executee :

```bash
python -m pipeline_2.run_analysis \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output-dir results/pipeline_2_compare_stanza \
  --lemmatizer stanza \
  --min-freq 3
```

Resultats principaux :

```text
Marqueurs extraits : 20 859
word : 9 551
lemma : 9 083
punctuation : 2 225
Marqueurs freq >= 3 pour H(emotion|marqueur) : 1 639
Entropie emotionnelle moyenne : 0.9372
```

### Run 3 : sans lemmatizer avec stopwords filtres

Commande executee :

```bash
python -m pipeline_2.run_analysis \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output-dir results/pipeline_2_compare_no_lemma \
  --no-lemma \
  --min-freq 3
```

Resultats principaux :

```text
Marqueurs extraits : 11 776
word : 9 551
punctuation : 2 225
Marqueurs freq >= 3 pour H(emotion|marqueur) : 882
Entropie emotionnelle moyenne : 0.9072
```

### Diagnostic comparatif Python

Le diagnostic comparatif a ensuite ete effectue en lisant les fichiers CSV generes dans les trois dossiers :

```text
results/pipeline_2_compare_spacy/markers.csv
results/pipeline_2_compare_stanza/markers.csv
results/pipeline_2_compare_no_lemma/markers.csv
results/pipeline_2_compare_*/specificity_results/entropy_per_marker_emotion.csv
results/pipeline_2_compare_*/specificity_results/entropy_per_marker_mode.csv
results/pipeline_2_compare_*/specificity_results/entropy_by_mode_summary.csv
```

Le script Python utilise etait structure autour des operations suivantes :

```python
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

runs = {
    "spacy": Path("results/pipeline_2_compare_spacy"),
    "stanza": Path("results/pipeline_2_compare_stanza"),
    "no_lemma": Path("results/pipeline_2_compare_no_lemma"),
}

for name, base in runs.items():
    markers = pd.read_csv(base / "markers.csv", encoding="utf-8-sig")
    entropy_emotion = pd.read_csv(
        base / "specificity_results" / "entropy_per_marker_emotion.csv",
        encoding="utf-8-sig",
    )
    entropy_mode = pd.read_csv(
        base / "specificity_results" / "entropy_per_marker_mode.csv",
        encoding="utf-8-sig",
    )

    rows_by_type = markers["marker_type"].value_counts()
    unique_by_type = (
        markers
        .drop_duplicates(["marker_type", "marker_value"])
        .groupby("marker_type")
        .size()
    )
    entropy_by_type = (
        entropy_emotion
        .groupby("marker_type")
        .agg(
            n=("marker_value", "size"),
            total_count_sum=("total_count", "sum"),
            mean_entropy=("entropy", "mean"),
            median_entropy=("entropy", "median"),
            mean_norm=("normalized_entropy", "mean"),
        )
    )
```

Les invariances des mots et ponctuations entre runs ont ete verifiees par comparaison directe :

```python
base_e = pd.read_csv(
    runs["no_lemma"] / "specificity_results" / "entropy_per_marker_emotion.csv",
    encoding="utf-8-sig",
)
base_nonlemma = (
    base_e[base_e.marker_type != "lemma"]
    .copy()
    .sort_values(["marker_type", "marker_value"])
    .reset_index(drop=True)
)

for name in ["spacy", "stanza"]:
    e = pd.read_csv(
        runs[name] / "specificity_results" / "entropy_per_marker_emotion.csv",
        encoding="utf-8-sig",
    )
    non = (
        e[e.marker_type != "lemma"]
        .copy()
        .sort_values(["marker_type", "marker_value"])
        .reset_index(drop=True)
    )

    cols = ["marker_value", "marker_type", "total_count", "entropy", "normalized_entropy"]
    exact_equal = base_nonlemma[cols].equals(non[cols])
```

Constat obtenu :

```text
Les lignes word et punctuation sont strictement identiques entre les trois runs.
Les differences entre runs avec et sans lemmatisation proviennent uniquement des lignes lemma ajoutees.
```

### Comparaison SpaCy/Stanza sur les lemmes

Le recouvrement SpaCy/Stanza a ete calcule sur les lemmes presents dans `entropy_per_marker_emotion.csv` avec `marker_type == "lemma"` :

```python
sp = pd.read_csv(
    runs["spacy"] / "specificity_results" / "entropy_per_marker_emotion.csv",
    encoding="utf-8-sig",
)
st = pd.read_csv(
    runs["stanza"] / "specificity_results" / "entropy_per_marker_emotion.csv",
    encoding="utf-8-sig",
)

sp_lem = sp[sp.marker_type == "lemma"].copy()
st_lem = st[st.marker_type == "lemma"].copy()

sp_set = set(sp_lem.marker_value)
st_set = set(st_lem.marker_value)

intersection = sp_set & st_set
spacy_only = sp_set - st_set
stanza_only = st_set - sp_set
jaccard = len(intersection) / len(sp_set | st_set)

merged = sp_lem.merge(
    st_lem,
    on=["marker_value", "marker_type"],
    suffixes=("_spacy", "_stanza"),
)

count_corr = merged["total_count_spacy"].corr(merged["total_count_stanza"])
entropy_corr = merged["entropy_spacy"].corr(merged["entropy_stanza"])
mean_abs_entropy_diff = (
    merged["entropy_spacy"] - merged["entropy_stanza"]
).abs().mean()
```

Resultats obtenus :

```text
Lemmes SpaCy freq >= 3 : 778
Lemmes Stanza freq >= 3 : 757
Intersection : 637
SpaCy seulement : 141
Stanza seulement : 120
Jaccard : 0.7094
Correlation des frequences sur lemmes communs : 0.9131
Correlation des entropies sur lemmes communs : 0.9632
Difference absolue moyenne d'entropie : 0.0662
```

### Comparaison lemme versus formes de surface frequentes

Une analyse exploratoire a aussi mesure si les lemmes presents au seuil etaient deja presents comme mots de surface frequents :

```python
no = pd.read_csv(
    runs["no_lemma"] / "specificity_results" / "entropy_per_marker_emotion.csv",
    encoding="utf-8-sig",
)
word_values = set(no[no.marker_type == "word"].marker_value)

for name in ["spacy", "stanza"]:
    e = pd.read_csv(
        runs[name] / "specificity_results" / "entropy_per_marker_emotion.csv",
        encoding="utf-8-sig",
    )
    lem = e[e.marker_type == "lemma"].copy()
    lem["also_word_freq_ge3"] = lem.marker_value.isin(word_values)

    summary = lem["also_word_freq_ge3"].value_counts()
    mean_also_word = lem[lem.also_word_freq_ge3]["entropy"].mean()
    mean_not_word = lem[~lem.also_word_freq_ge3]["entropy"].mean()
```

Resultats observes :

```text
SpaCy :
  lemmes deja presents comme word freq >= 3 : 467
  lemmes absents comme word freq >= 3 : 311
  H moyenne des lemmes deja words : 0.9918
  H moyenne des lemmes non words : 0.8943

Stanza :
  lemmes deja presents comme word freq >= 3 : 492
  lemmes absents comme word freq >= 3 : 265
  H moyenne des lemmes deja words : 0.9729
  H moyenne des lemmes non words : 0.9695
```

Cette analyse reste insuffisante pour mesurer la granularite forme/lemme, car elle compare seulement les valeurs de marqueurs et non les relations token par token entre formes de surface et lemmes.

### Tests de distribution

Les distributions d'entropie emotionnelle ont ete comparees par Mann-Whitney et Kolmogorov-Smirnov :

```python
for name in ["spacy", "stanza"]:
    e = pd.read_csv(
        runs[name] / "specificity_results" / "entropy_per_marker_emotion.csv",
        encoding="utf-8-sig",
    )
    lemma_values = e[e.marker_type == "lemma"]["entropy"].values
    nonlemma_values = e[e.marker_type != "lemma"]["entropy"].values

    u, p_mw = stats.mannwhitneyu(
        lemma_values,
        nonlemma_values,
        alternative="two-sided",
    )
    ks, p_ks = stats.ks_2samp(lemma_values, nonlemma_values)
```

Resultats :

```text
SpaCy :
  lemma n = 778, moyenne = 0.9528
  non-lemma n = 882, moyenne = 0.9067
  Mann-Whitney p = 0.211
  KS p = 0.702

Stanza :
  lemma n = 757, moyenne = 0.9717
  non-lemma n = 882, moyenne = 0.9067
  Mann-Whitney p = 0.0699
  KS p = 0.427
```

Comparaison entre runs complets :

```text
SpaCy versus Stanza :
  difference de moyenne = -0.0084
  Mann-Whitney p = 0.715
  KS p = 1.000

SpaCy versus sans lemmes :
  difference de moyenne = 0.0216
  Mann-Whitney p = 0.488
  KS p = 0.998

Stanza versus sans lemmes :
  difference de moyenne = 0.0300
  Mann-Whitney p = 0.320
  KS p = 0.974
```

Ces tests indiquent une forte stabilite globale, mais ils ne capturent pas les deplacements locaux de specificite entre une forme et son lemme.

### Matching lexical et fuzzy matching

Une table de frequences combinees des lemmes SpaCy et Stanza a ete construite :

```python
from pathlib import Path
import pandas as pd

out = Path("results/pipeline_2_compare_lexicon_matching")
out.mkdir(parents=True, exist_ok=True)

spacy = pd.read_csv("results/pipeline_2_compare_spacy/markers.csv", encoding="utf-8-sig")
stanza = pd.read_csv("results/pipeline_2_compare_stanza/markers.csv", encoding="utf-8-sig")

spacy_counts = spacy[spacy.marker_type.eq("lemma")]["marker_value"].value_counts()
stanza_counts = stanza[stanza.marker_type.eq("lemma")]["marker_value"].value_counts()

values = sorted(set(spacy_counts.index) | set(stanza_counts.index))
global_counts = pd.DataFrame({
    "marker_value": values,
    "spacy_count": [int(spacy_counts.get(v, 0)) for v in values],
    "stanza_count": [int(stanza_counts.get(v, 0)) for v in values],
})
global_counts["total_count"] = global_counts["spacy_count"] + global_counts["stanza_count"]
global_counts = global_counts.sort_values(
    ["total_count", "marker_value"],
    ascending=[False, True],
)

global_counts.to_csv(
    out / "pipeline_2_lemma_global_counts_spacy_stanza.csv",
    index=False,
)
```

Puis le script existant a ete lance :

```bash
python tools/match_marker_values_to_lexicon.py \
  --global-markers results/pipeline_2_compare_lexicon_matching/pipeline_2_lemma_global_counts_spacy_stanza.csv \
  --lexicon emotions/lexique_emotionnel.tsv \
  --outdir results/pipeline_2_compare_lexicon_matching \
  --marker-col marker_value
```

Resultats :

```text
Valeurs uniques : 3 089
Occurrences total_count : 17 909
Matchs exacts : 335 valeurs uniques / 3 522 occurrences
Reparations haute confiance : 162 valeurs uniques / 685 occurrences
Matched apres reparations : 497 valeurs uniques / 4 207 occurrences
Candidats de reparation : 395 lignes pour 220 marqueurs
```

Sorties produites :

```text
results/pipeline_2_compare_lexicon_matching/global_marker_value_counts_with_lexicon_repair.csv
results/pipeline_2_compare_lexicon_matching/global_marker_values_not_in_lexique_repair_candidates.csv
results/pipeline_2_compare_lexicon_matching/global_marker_values_not_in_lexique_high_confidence_candidates.csv
results/pipeline_2_compare_lexicon_matching/lexicon_repair_summary.csv
```

## Limites identifiees

### 1. Les runs avec lemmatizer sont des analyses mixtes

Dans l'etat actuel, un run avec SpaCy ou Stanza ajoute les lemmes aux mots de surface et aux ponctuations. Il ne remplace pas les mots de surface par leurs lemmes.

Cela signifie qu'un run avec lemmatizer contient :

```text
word + punctuation + lemma
```

et non :

```text
lemma uniquement
```

Cette limite est importante pour l'interpretation. Les resultats globaux des runs avec lemmatizer ne mesurent pas une lemmatisation pure, mais l'effet de l'ajout d'un niveau supplementaire de marqueurs.

### 2. Risque de double comptage analytique

Un meme segment peut contribuer a la fois par une forme de surface et par un lemme. Cette situation peut etre acceptable si l'on analyse differents types de marqueurs, mais elle ne doit pas etre confondue avec une representation canonique unique.

Pour l'analyse de granularite, il faut pouvoir comparer explicitement :

- surface seulement ;
- lemme seulement ;
- surface et lemme separes ;
- representation canonique unique.

### 3. Absence de table token-level

La pipeline actuelle ne conserve pas de relation explicite entre :

- une forme de surface ;
- son lemme SpaCy ;
- son lemme Stanza ;
- sa position dans le segment ;
- son contexte textuel ;
- ses emotions et modes associes.

Cette absence empeche de produire directement un graphe fiable forme/lemme. Elle limite aussi l'analyse des variantes flexionnelles, car on ne sait pas exactement quelles formes ont ete regroupees par chaque lemme.

### 4. Tokenisation heterogene

Les mots de surface sont extraits par regex, tandis que les lemmes sont produits par SpaCy ou Stanza. Les tokenisations peuvent diverger. Une analyse token-level devra documenter ces divergences.

### 5. Artefacts de lemmatisation

Des formes non lexicales ou discutables ont ete observees.

Exemples SpaCy :

```text
colèr
hont
fierter
dangereu
luire
```

Exemple Stanza :

```text
soi
```

Le cas `soi` est particulierement important, car il peut provenir de formes pronominales filtrees en surface, mais reintegrees apres lemmatisation si `soi` n'est pas dans la liste locale de stopwords.

### 6. Stopword leakage

Le filtrage des stopwords doit etre applique a la fois :

- avant lemmatisation, sur la forme de surface ;
- apres lemmatisation, sur le lemme produit.

Il faut egalement verifier que la liste de stopwords contient les formes reconstruites par les lemmatizers.

### 7. Fuzzy matching utile mais risque

Le fuzzy matching a permis de proposer des corrections plausibles :

| Forme observee | Candidat plausible |
|---|---|
| `colèr` | `colère` |
| `dangereu` | `dangereux` |
| `fierter` | `fierté` |
| `inquiète` | `inquiéter` |
| `effrayé` | `effrayer` |

Mais il a aussi produit des candidats dangereux :

| Forme observee | Candidat discutable |
|---|---|
| `pleurer` | `pleureur` |
| `mauvais` | `mauvaiseté` |
| `attendre` | `attendri` |
| `important` | `importun` |
| `heure` | `heureux` |

Conclusion : le fuzzy matching doit d'abord etre un outil d'annotation et de generation de candidats. Il ne doit pas automatiquement modifier la valeur canonique utilisee dans l'analyse de specificite sans validation.

### 8. Dilution possible de la specificite

La limite la plus importante pour ce nouveau projet est la suivante : un lemme peut diluer la specificite d'une forme.

Exemple abstrait :

```text
Forme X :
  emotion dominante = Colere
  entropie faible

Lemme(X) :
  formes regroupees = X, X1, X2, X3
  emotions presentes = Colere, Peur, Tristesse, Joie
  entropie plus elevee
```

Dans ce cas, la lemmatisation n'est pas simplement meilleure ou moins bonne. Elle change l'objet analyse. Elle passe d'un indice local a une famille morphologique ou lexicale.

La visualisation doit rendre ce changement visible.

## Objectif methodologique : visualiser la granularite

La methode proposee doit produire une visualisation qui permette d'inspecter les relations entre :

- formes de surface ;
- lemmes SpaCy ;
- lemmes Stanza ;
- entrees du lexique emotionnel ;
- corrections fuzzy candidates ;
- corrections validees ;
- emotions ;
- modes d'expression.

L'objectif n'est pas seulement de produire une image illustrative. La visualisation doit etre un instrument d'analyse.

Elle doit aider a repondre aux questions suivantes :

1. Quelles formes gravitent autour d'un lemme ?
2. Quelles formes sont plus specifiques emotionnellement que leur lemme ?
3. Quels lemmes revelent une famille emotionnelle coherente ?
4. Quels lemmes melangent plusieurs emotions et diluent les specificites locales ?
5. Quels lemmes sont propres a SpaCy ou a Stanza ?
6. Quels lemmes sont des artefacts probables ?
7. Quelles corrections fuzzy sont plausibles ?
8. Quelles corrections fuzzy doivent etre annotees manuellement ?
9. Le mode `Designee` forme-t-il des familles lexicales plus structurees que les autres modes ?
10. Quelles familles peuvent contribuer a un lexique emotionnel structure ?

## Modele de graphe propose

### Types de noeuds

| Type de noeud | Exemple | Role |
|---|---|---|
| `surface_form` | `inquiète` | forme observee dans le corpus |
| `lemma_spacy` | `inquiéter` | lemme produit par SpaCy |
| `lemma_stanza` | `inquiéter` | lemme produit par Stanza |
| `canonical_lemma` | `inquiéter` | forme canonique retenue ou candidate |
| `lexicon_entry` | `inquiétude` | entree du lexique emotionnel |
| `emotion` | `Peur` | emotion normalisee |
| `mode` | `Designee` | mode d'expression |
| `repair_candidate` | `colère` | candidat fuzzy |
| `artifact_flag` | `stopword_leakage` | signal de risque |

### Types d'aretes

| Type d'arete | Source | Cible | Poids ou attributs |
|---|---|---|---|
| `lemmatized_as` | surface form | lemma | backend, frequence |
| `canonicalized_as` | lemma ou surface | canonical lemma | methode, confiance |
| `exact_lexicon_match` | marker | lexicon entry | categorie emotionnelle |
| `fuzzy_candidate` | marker | repair candidate | distance, score, confiance |
| `associated_with_emotion` | marker ou lemma | emotion | frequence, probabilite, entropie |
| `observed_in_mode` | marker ou lemma | mode | frequence, probabilite |
| `has_artifact_flag` | marker ou lemma | artifact flag | type de suspicion |

### Attributs des noeuds

Chaque noeud lexical devrait idealement porter :

| Attribut | Description |
|---|---|
| `total_count` | frequence totale |
| `mode_counts` | distribution par mode |
| `emotion_counts` | distribution par emotion |
| `dominant_emotion` | emotion la plus frequente |
| `dominant_emotion_share` | part de l'emotion dominante |
| `emotion_entropy` | entropie emotionnelle |
| `mode_entropy` | entropie modale |
| `backend` | SpaCy, Stanza, surface, canonical |
| `lexicon_status` | exact, fuzzy, no_match, reviewed |
| `artifact_flags` | signaux de suspicion |

## Mesures de granularite a calculer

### Mesures forme/lemme

Pour chaque relation entre une forme et un lemme :

| Mesure | Interpretation |
|---|---|
| `surface_count` | frequence de la forme |
| `lemma_count` | frequence totale du lemme |
| `surface_entropy` | specificite de la forme |
| `lemma_entropy` | specificite du lemme |
| `delta_entropy = lemma_entropy - surface_entropy` | effet de dilution ou de concentration |
| `surface_dominant_emotion` | emotion dominante de la forme |
| `lemma_dominant_emotion` | emotion dominante du lemme |
| `dominant_emotion_changed` | indique si le regroupement change l'emotion dominante |
| `surface_share_in_lemma` | poids de la forme dans le lemme |

Interpretation de `delta_entropy` :

| Cas | Interpretation |
|---|---|
| `delta_entropy > 0` | le lemme est plus disperse que la forme ; possible dilution |
| `delta_entropy < 0` | le lemme est plus specifique que la forme ; possible consolidation |
| `delta_entropy proche de 0` | le regroupement conserve la specificite |

### Typologie analytique

Chaque relation forme/lemme pourrait etre classee dans une categorie :

| Categorie | Definition |
|---|---|
| `specificity_preserved` | la forme et le lemme gardent la meme emotion dominante et une entropie proche |
| `lemma_dilutes_surface` | la forme est specifique, le lemme est plus disperse |
| `lemma_reveals_family` | le lemme devient plus stable que les formes prises separement |
| `dominant_emotion_shift` | l'emotion dominante change entre forme et lemme |
| `backend_artifact` | le lemme est suspect ou non lexical |
| `stopword_leakage` | le lemme reintroduit un mot-outil |
| `needs_manual_review` | la relation ou correction demande une annotation |

## Focalisation sur `Designee`

Le mode `Designee` doit faire l'objet d'une vue dediee.

Questions specifiques :

1. Les lemmes du mode `Designee` correspondent-ils davantage au lexique emotionnel que ceux des autres modes ?
2. Les familles forme/lemme y sont-elles plus coherentes emotionnellement ?
3. Les formes satellites autour d'un lemme ont-elles tendance a partager la meme emotion ?
4. Le graphe du mode `Designee` permet-il d'identifier des noyaux lexicaux emotionnels structurants ?
5. Quelles formes doivent rester distinctes parce qu'elles sont plus specifiques que leur lemme ?

Sorties recommandees pour cette focalisation :

```text
designee_lemma_graph.graphml
designee_lemma_graph.gexf
designee_lemma_graph.json
designee_granularity_summary.csv
designee_manual_review_candidates.csv
```

Une visualisation interactive serait preferable a une image statique, par exemple avec :

- NetworkX pour construire le graphe ;
- PyVis pour une premiere visualisation HTML ;
- Plotly ou Dash si l'on veut filtrer dynamiquement par emotion, mode, backend ou statut ;
- Gephi si l'on veut explorer manuellement un graphe exporte en GraphML ou GEXF.

## Table token-level necessaire

Pour produire ce graphe proprement, il faut ajouter une table intermediaire token-level.

Nom possible :

```text
results/pipeline_2_granularity/token_lemmas.csv
```

Colonnes recommandees :

| Colonne | Description |
|---|---|
| `segment_id` | identifiant stable de la ligne source |
| `source_file` | fichier source |
| `text_span` | texte original |
| `mode` | mode normalise |
| `emotion` | emotion normalisee apres explosion |
| `token_index` | position du token |
| `char_start` | debut du token si disponible |
| `char_end` | fin du token si disponible |
| `surface` | forme observee |
| `surface_norm` | forme normalisee |
| `backend` | `spacy`, `stanza`, `regex` |
| `lemma` | lemme produit |
| `lemma_norm` | lemme normalise |
| `pos` | partie du discours si disponible |
| `is_alpha` | token alphabetique |
| `surface_is_stopword` | stopword en surface |
| `lemma_is_stopword` | stopword apres lemmatisation |
| `kept_for_analysis` | token conserve ou non |
| `drop_reason` | raison d'exclusion |

Cette table ne doit pas remplacer `markers.csv` immediatement. Elle doit d'abord servir de base d'audit et de visualisation.

## Tables derivees recommandees

### `surface_lemma_links.csv`

Une ligne par relation entre forme et lemme.

Colonnes possibles :

```text
backend
surface_norm
lemma_norm
surface_count
lemma_count
surface_entropy
lemma_entropy
delta_entropy
surface_dominant_emotion
lemma_dominant_emotion
dominant_emotion_changed
surface_share_in_lemma
mode
artifact_flags
granularity_class
```

### `lemma_family_summary.csv`

Une ligne par lemme ou entree canonique.

Colonnes possibles :

```text
backend
lemma_norm
n_surface_forms
surface_forms
total_count
dominant_surface
dominant_surface_share
dominant_emotion
dominant_emotion_share
emotion_entropy
mode_entropy
lexicon_status
n_exact_lexicon_links
n_fuzzy_candidates
n_artifact_flags
needs_manual_review
```

### `granularity_delta_by_marker.csv`

Une ligne par forme ayant un lien avec un lemme.

Objectif : identifier les cas ou la lemmatisation change fortement l'interpretation.

Colonnes importantes :

```text
surface_norm
lemma_norm
surface_entropy
lemma_entropy
delta_entropy
surface_dominant_emotion
lemma_dominant_emotion
surface_count
lemma_count
mode
review_priority
```

### `manual_annotation_candidates.csv`

Fichier destine a une validation humaine.

Le porteur du projet indique etre pret a contribuer aux annotations manuellement si necessaire. Cette disponibilite doit etre exploitee pour les cas ou la correction automatique serait fragile.

Colonnes possibles :

```text
candidate_id
surface_norm
lemma_norm
candidate_canonical
backend
mode
emotion_distribution
surface_count
lemma_count
delta_entropy
fuzzy_score
levenshtein_distance
lexicon_category
suggested_action
human_decision
human_canonical
human_note
annotator
annotation_date
```

Decisions humaines possibles :

| Decision | Effet |
|---|---|
| `accept_mapping` | accepter la canonicalisation proposee |
| `reject_mapping` | refuser la correction |
| `keep_surface_distinct` | conserver la forme comme marqueur distinct |
| `merge_strict_flexion` | fusionner seulement les flexions strictes |
| `merge_lexical_family` | fusionner une famille lexicale plus large |
| `drop_artifact` | exclure un artefact |
| `needs_context` | demander inspection de contextes textuels |

## Strategie de visualisation

### Vue 1 : graphe ego-centre par lemme

Pour un lemme donne :

- noeud central : lemme ;
- satellites : formes de surface ;
- couleur des satellites : emotion dominante ;
- taille des satellites : frequence ;
- epaisseur de l'arete : frequence de la relation surface -> lemme ;
- bordure ou icone : backend SpaCy/Stanza ;
- couleur du noeud central : emotion dominante du lemme ;
- halo ou marqueur : presence dans le lexique emotionnel.

Cette vue permet de voir immediatement si les satellites sont emotionnellement coherents ou heterogenes.

### Vue 2 : graphe global du mode `Designee`

Noeuds :

- lemmes ;
- formes ;
- emotions ;
- entrees du lexique emotionnel.

Filtres :

- emotion dominante ;
- entropie maximale ;
- frequence minimale ;
- backend ;
- statut lexical ;
- corrections validees ou non.

Objectif : identifier des noyaux lexicaux structurants pour le mode `Designee`.

### Vue 3 : carte de dilution

Representation possible :

- axe x : entropie de la forme ;
- axe y : entropie du lemme ;
- couleur : emotion dominante ;
- taille : frequence ;
- facettes : mode ou backend.

Les points au-dessus de la diagonale indiquent des cas ou le lemme est plus disperse que la forme. Les points sous la diagonale indiquent des cas ou le lemme est plus specifique que la forme.

### Vue 4 : tableau interactif de revue

Un tableau filtrable doit accompagner le graphe. Il est essentiel pour l'annotation manuelle.

Colonnes prioritaires :

```text
surface_norm
lemma_norm
mode
surface_count
lemma_count
surface_entropy
lemma_entropy
delta_entropy
surface_dominant_emotion
lemma_dominant_emotion
backend
lexicon_status
artifact_flags
suggested_action
human_decision
```

## Methodologie d'implementation proposee

### Etape 1 : reproduire les vues de base

Reproduire les trois runs deja executes :

```bash
python -m pipeline_2.run_analysis \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output-dir results/pipeline_2_compare_spacy \
  --lemmatizer spacy \
  --min-freq 3

python -m pipeline_2.run_analysis \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output-dir results/pipeline_2_compare_stanza \
  --lemmatizer stanza \
  --min-freq 3

python -m pipeline_2.run_analysis \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output-dir results/pipeline_2_compare_no_lemma \
  --no-lemma \
  --min-freq 3
```

### Etape 2 : produire une table token-level

Ajouter ou creer un script dedie, par exemple :

```bash
python -m pipeline_2.build_token_lemma_table \
  --input data/pipeline_2/SimpleSitEmo.parquet \
  --output results/pipeline_2_granularity/token_lemmas.csv \
  --lemmatizers spacy stanza \
  --remove-stopwords
```

Ce script devrait conserver les relations entre formes et lemmes sans encore appliquer de correction.

### Etape 3 : calculer les specificites par niveau

Calculer separement :

- specificite des formes de surface ;
- specificite des lemmes SpaCy ;
- specificite des lemmes Stanza ;
- specificite des formes canoniques validees si elles existent.

Une option possible serait d'etendre `marker_specificity.py` pour accepter :

```bash
--marker-view surface
--marker-view lemma_spacy
--marker-view lemma_stanza
--marker-view canonical
```

### Etape 4 : calculer les deltas forme/lemme

Construire `surface_lemma_links.csv` et `granularity_delta_by_marker.csv`.

Pseudo-code :

```python
surface_entropy = compute_entropy_for_view(token_table, marker_col="surface_norm")
lemma_entropy = compute_entropy_for_view(token_table, marker_col="lemma_norm")

links = (
    token_table
    .groupby(["backend", "surface_norm", "lemma_norm"])
    .size()
    .reset_index(name="surface_lemma_count")
)

links = links.merge(surface_entropy, on="surface_norm", how="left")
links = links.merge(lemma_entropy, on="lemma_norm", how="left", suffixes=("_surface", "_lemma"))

links["delta_entropy"] = links["entropy_lemma"] - links["entropy_surface"]
links["dominant_emotion_changed"] = (
    links["dominant_emotion_surface"] != links["dominant_emotion_lemma"]
)
```

### Etape 5 : annoter lexicalement

Reprendre la logique de `tools/match_marker_values_to_lexicon.py`, mais sans appliquer automatiquement les reparations.

Sorties attendues :

```text
lexicon_exact_matches.csv
lexicon_fuzzy_candidates.csv
manual_annotation_candidates.csv
```

La correction automatique doit rester optionnelle et strictement controlee.

### Etape 6 : construire le graphe

Exemple d'approche avec NetworkX :

```python
import networkx as nx

G = nx.Graph()

for _, row in surface_lemma_links.iterrows():
    surface_node = f"surface::{row.surface_norm}"
    lemma_node = f"lemma::{row.lemma_norm}"

    G.add_node(
        surface_node,
        node_type="surface_form",
        label=row.surface_norm,
        entropy=row.entropy_surface,
        dominant_emotion=row.dominant_emotion_surface,
        count=row.surface_count,
    )
    G.add_node(
        lemma_node,
        node_type="lemma",
        label=row.lemma_norm,
        entropy=row.entropy_lemma,
        dominant_emotion=row.dominant_emotion_lemma,
        count=row.lemma_count,
    )
    G.add_edge(
        surface_node,
        lemma_node,
        edge_type="lemmatized_as",
        backend=row.backend,
        weight=row.surface_lemma_count,
        delta_entropy=row.delta_entropy,
    )
```

Export possible :

```python
nx.write_graphml(G, "results/pipeline_2_granularity/lemma_graph.graphml")
nx.write_gexf(G, "results/pipeline_2_granularity/lemma_graph.gexf")
```

### Etape 7 : produire une visualisation interactive

Options possibles :

```text
PyVis : rapide pour un premier HTML exploratoire.
Plotly/Dash : plus robuste pour filtrer et inspecter.
Gephi : utile pour exploration manuelle apres export GEXF ou GraphML.
```

La premiere implementation peut commencer par PyVis :

```python
from pyvis.network import Network

net = Network(height="900px", width="100%", notebook=False)
net.from_nx(G)
net.show("results/pipeline_2_granularity/lemma_graph.html")
```

Une implementation plus avancee devrait permettre :

- filtrage par mode ;
- filtrage par emotion dominante ;
- filtrage par backend ;
- filtrage par entropie ;
- affichage des distributions emotionnelles au survol ;
- selection des cas a annoter.

## Place de l'annotation manuelle

L'annotation manuelle est pertinente et probablement necessaire.

Le porteur du projet est pret a contribuer aux annotations si necessaire. La methode doit donc prevoir une boucle semi-automatique :

1. le script identifie les cas ambigus ;
2. il produit un fichier de revue ;
3. l'annotateur valide, rejette ou modifie les propositions ;
4. les decisions humaines sont reinserees dans une table de mappings valides ;
5. une vue canonique est calculee uniquement a partir de ces mappings valides.

Les cas prioritaires pour annotation sont :

- forte frequence ;
- fort `delta_entropy` ;
- changement d'emotion dominante ;
- candidat fuzzy haute confiance mais semantiquement ambigu ;
- lemme exclusif a un backend ;
- lemme non lexical ;
- cas du mode `Designee` pertinents pour le lexique emotionnel structure.

## Produit attendu

Le resultat final de cette methode ne doit pas etre seulement un fichier corrige. Il doit comprendre :

1. une analyse de specificite par niveau de granularite ;
2. une table des relations formes -> lemmes ;
3. une mesure de dilution ou de consolidation de la specificite ;
4. un graphe exploratoire ;
5. une liste de cas a annoter ;
6. une table de corrections validees ;
7. une comparaison avant/apres canonicalisation ;
8. une attention particuliere au mode `Designee`.

## Conclusion methodologique

La lemmatisation doit etre traitee comme un changement de niveau d'analyse. Elle peut :

- enrichir la couverture en regroupant des variantes ;
- reveler des familles lexicales emotionnelles ;
- aider a structurer un lexique, notamment pour le mode `Designee` ;
- mais aussi diluer la specificite d'une forme locale ;
- introduire des artefacts propres a un backend ;
- reintegrer des stopwords sous forme lemmatisee ;
- ou produire des corrections fuzzy trompeuses.

La visualisation par graphe est donc un objectif methodologique pertinent. Elle permet de representer le lemme comme un centre lexical sans effacer les formes qui gravitent autour de lui. Elle rend observable la tension entre deux niveaux : le lexeme distinctif local et la famille lemmatique ou canonique plus generale.

La methode recommandee est comparative, traçable et semi-automatique. Elle doit conserver les formes originales, documenter les lemmes, calculer les differences de specificite, proposer des corrections sans les imposer, puis utiliser l'annotation manuelle pour les cas ou la decision automatique serait fragile.
