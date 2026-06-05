# Audit technique : unités SitEmo sans mode et alignement Glozz/XLSX

Remarque : le fichier xlsx a été supprimé mais il est téléchargeable avec :

```bash
wget https://github.com/GwenTsang/Eval-EMOTYC/raw/refs/heads/main/golds/emotexttokids_gold_flat.xlsx
```

Ce mémo documente les vérifications effectuées sur le corpus Glozz de
`data/raw/glozz/` et sur le fichier aplati `emotexttokids_gold_flat.xlsx`.
L'objectif était de comprendre le statut des unités `SitEmo` sans `Mode` dans
Glozz et de vérifier si le fichier XLSX aplati permettait d'inférer une version
plus récente ou plus complète des annotations.

## Données examinées

- Corpus Glozz brut : `data/raw/glozz/`
- Parseur Glozz : `analysis_pipeline/glozz_parser.py`
- Exports existants :
  - `results/glozz/specificity_results_spacy/annotations.csv`
  - `results/glozz/specificity_results_stanza/annotations.csv`
- Corpus XLSX aplati : `emotexttokids_gold_flat.xlsx`

## Hypothèse 1 : le parseur Glozz raterait certaines annotations de mode

Vérification effectuée directement dans les XML `.aa`, indépendamment des CSV
déjà exportés.

Résultats :

- Annotations Glozz extraites : 8 633.
- Unités `SitEmo` : 7 096.
- Unités `SitEmo` sans `Mode` : 164.
- Toutes les unités `SitEmo` possèdent une feature XML nommée exactement
  `Mode`.
- Aucune variante de nom de feature de mode n'a été trouvée (`mode`, `Mode `,
  etc.).
- Pour les 164 unités concernées, la feature `Mode` existe mais sa valeur est
  vide.

Conclusion : rien n'indique que `analysis_pipeline/glozz_parser.py` rate des modes
existants. Les 164 absences de mode semblent présentes dans les fichiers Glozz
eux-mêmes.

## Hypothèse 2 : les 164 unités SitEmo sans mode seraient textuellement vides

Vérification effectuée sur les offsets Glozz et sur les valeurs des features XML.

Résultats :

- Spans textuels vides parmi les 164 unités : 0.
- Longueur minimale du span après trim : 3 caractères.
- Longueur médiane : 28,5 caractères.
- Longueur maximale : 150 caractères.
- Les 164 unités ont toutes la même signature de features :
  `Mode`, `Type`, `Categorie`, `Type2`, `Categorie2`, `Nature`, `Declencheur`.
- `Categorie` non vide : 0/164.
- `Categorie2` non vide : 0/164.
- `Type`, `Type2`, `Declencheur` non vides : 0/164.
- `Nature` non vide : 11/164.
- Les 153 autres unités n'ont aucune feature renseignée, au-delà du type
  d'unité `SitEmo` et de leurs offsets.

Conclusion : ces unités ne sont pas textuellement vides, mais elles sont presque
toutes vides annotationnellement. Aucune des 164 n'a d'émotion renseignée dans
`Categorie` ou `Categorie2`.

## Hypothèse 3 : toutes les lignes émotionnelles du XLSX aplati auraient un mode

Vérification effectuée sur `emotexttokids_gold_flat.xlsx`.

Résultats :

- Lignes XLSX : 27 911.
- Colonne texte utilisée : `TEXT`.
- Lignes `Emo=1` : 5 374.
- Lignes avec au moins un mode parmi `Comportementale`, `Designee`, `Montree`,
  `Suggeree` : 4 407.
- Lignes `Emo=1` sans aucun mode : 967.
- Lignes `Emo=0` avec un mode : 0.

Conclusion : l'hypothèse selon laquelle toutes les phrases émotionnelles du XLSX
ont un mode est fausse pour ce fichier. Il existe 967 lignes `Emo=1` sans mode.

## Hypothèse 4 : les phrases du XLSX seraient absentes du Glozz brut

Méthode :

- Alignement de chaque valeur `TEXT` du XLSX sur les textes `.ac`.
- Recherche exacte après normalisation des espaces.
- Pour les cas restants, normalisation contrôlée des guillemets/citations pour
  gérer les différences entre citations conservées dans les `.ac` et citations
  supprimées ou déplacées dans le XLSX.
- Pas de découpage phrastique automatique, afin d'éviter les erreurs autour des
  ellipses (`...`) et des titres collés au texte suivant.

Résultats :

- Lignes XLSX alignées exactement après normalisation des espaces : 27 901/27 911.
- Lignes supplémentaires alignées après normalisation des guillemets : 10/27 911.
- Lignes réellement non alignées : 0/27 911.
- Lignes avec plus d'une occurrence globale possible : 831, surtout à cause de
  textes courts ou génériques.

Conclusion : toutes les valeurs `TEXT` du XLSX ont été retrouvées dans les textes
bruts Glozz, en tenant compte de variations typographiques limitées.

## Hypothèse 5 : un SitEmo sans mode du Glozz serait repris dans le XLSX avec un mode

Méthode :

- Alignement de chaque ligne `TEXT` du XLSX sur les offsets bruts des fichiers
  `.ac`.
- Croisement offset-level avec les 164 unités `SitEmo` sans mode.
- Pour éviter les faux positifs, le test principal n'était pas une simple
  recherche du span dans le XLSX. Une ligne XLSX devait couvrir l'intervalle
  Glozz de l'unité dans le même fichier `.ac`.
- Pour chaque ligne couvrante, vérification des autres unités `SitEmo` avec mode
  déjà présentes dans la même ligne.

Résultats :

- `SitEmo` sans mode couverts par une ligne `TEXT` du XLSX : 161/164.
- `SitEmo` sans mode non couverts par une ligne `TEXT` unique : 3/164.
- Parmi les 161 couverts, au moins un mode est présent dans la ligne XLSX dans
  161/161 cas.
- Dans tous ces cas, le ou les modes de la ligne XLSX sont déjà expliqués par au
  moins une autre unité `SitEmo` avec mode dans la même ligne.
- Cas forts où une ligne XLSX aurait un mode sans autre `SitEmo` modal dans la
  même ligne : 0.
- Cas où un mode XLSX ne serait pas expliqué par les autres `SitEmo` modaux de
  la ligne : 0.

Conclusion : oui, 161 unités `SitEmo` sans mode du Glozz se retrouvent dans des
lignes `TEXT` du XLSX. Cependant, aucun cas ne force à conclure que ces unités
avaient reçu un mode dans une version plus récente du corpus. Les modes de ces
lignes sont toujours explicables par d'autres unités `SitEmo` déjà modales dans
le même contexte.

Les 3 unités sans mode non couvertes par une ligne `TEXT` unique sont :

- `Albert38-A3_premiers_conges_payes` :
  `de travailler moins longtemps et de bénéficier de congés payés`
- `Albert4-Trump_Clinton_coup` :
  `d'avoir effacé volontairement 33.000 e-mails confidentiels de sa boîte personnelle`
  ; le XLSX coupe `33.000` en deux lignes (`33.` puis `000 e-mails...`).
- `Albert46-A3_Arabie_Saoudite` :
  `fui son pays`

## Implication technique

Le filtrage des `SitEmo` sans mode dans `analysis_pipeline/extract_markers_glozz.py` est
cohérent avec l'objectif actuel d'analyser les marqueurs par mode d'expression :
ces 164 unités ne fournissent pas de label de mode exploitable. En revanche, ces
unités couvrent des spans textuels réels et doivent être conservées pour une éventuelle
correction et complétion du corpus.
