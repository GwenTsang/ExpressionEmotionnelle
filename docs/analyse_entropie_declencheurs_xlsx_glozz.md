# Analyse : effet du span declencheur sur les entropies Glozz/XLSX

## Objet du compte rendu

Ce document rend compte des analyses menees pour tester l'hypothese suivante :

> Le fait de restreindre les segments Glozz au span textuel `Declencheur`
> rapproche-t-il les scores d'entropie obtenus sur le corpus Glozz de ceux
> obtenus sur le corpus XLSX ?

La question a ete testee a deux niveaux :

- une version specifique, centree sur le mode `Designee` ;
- une version plus generale, couvrant les quatre modes et, en complement, les
  emotions.

Les resultats soutiennent nettement l'hypothese pour les scores **par mode**.
Ils sont plus mitiges pour les scores **par emotion**, surtout parce que le
corpus XLSX devient tres petit pour plusieurs emotions apres filtrage.

## Donnees et scripts utilises

Les analyses s'appuient sur les fichiers deja produits par le run existant de
`run_pipelines.sh`.

Données :

- `data/pipeline_1/SimpleSitEmo_xlsx.parquet`
- `data/pipeline_1/SimpleSitEmo_glozz.parquet`
- `data/pipeline_2/SimpleSitEmo_xlsx.parquet`
- `data/pipeline_2/SimpleSitEmo_glozz.parquet`
- `results/pipeline_1/markers.csv`
- `results/pipeline_2/markers.csv`

Fonctions reutilisees :

- `pipeline_1.marker_specificity.compute_conditional_entropy`
- `pipeline_1.marker_specificity.compute_entropy_by_mode`
- `pipeline_2.marker_specificity.compute_conditional_entropy`
- `pipeline_2.marker_specificity.compute_entropy_by_mode`

Parametre d'entropie :

- `min_freq = 3`, comme dans le script de specificite.

## Difference entre pipeline 1 et pipeline 2

La difference principale examinee ici est la suivante :

- `pipeline_1/build_simplesitemo_glozz.py` conserve comme `text_span` le segment
  annote complet dans Glozz ;
- `pipeline_2/build_simplesitemo_glozz.py` utilise prioritairement la feature
  Glozz `Declencheur` comme `text_span`.

Dans les parquets observes :

```text
pipeline_1 XLSX  : 491 unites
pipeline_1 Glozz : 6811 unites

pipeline_2 XLSX  : 491 unites, text_span_source = segment_complet pour 491/491
pipeline_2 Glozz : 6811 unites, text_span_source = declencheur pour 6808/6811
```

La pipeline 2 ne modifie donc pas les spans XLSX ; elle modifie presque tous les
spans Glozz.

### Nuance importante : filtrage des stopwords

La comparaison brute `pipeline_1` vs `pipeline_2` ne mesure pas uniquement
l'effet du passage au declencheur. Elle confond au moins deux changements :

- le passage du segment Glozz complet au span `Declencheur` ;
- le filtrage par defaut des stopwords dans `pipeline_2/extract_markers.py`,
  alors que `pipeline_1/extract_markers.py` les conserve par defaut.

Pour isoler au mieux l'effet du declencheur, deux comparaisons ont ete faites :

- comparaison brute : `pipeline_1` telle quelle vs `pipeline_2` telle quelle ;
- comparaison controlee : `pipeline_1` apres suppression des stopwords avec
  `pipeline_2.nlp_utils.FR_STOPWORDS` vs `pipeline_2`.

La comparaison controlee est celle qui doit etre privilegiee pour interpreter
l'effet propre du passage au span `Declencheur`.

## Definition des entropies

Le script `marker_specificity.py` calcule l'entropie de Shannon :

```text
H(C | x) = - somme_c P(C = c | marqueur = x) * log2(P(C = c | marqueur = x))
```

Deux familles d'analyses ont ete distinguees.

### Entropie emotionnelle par mode

1. Pour chaque corpus separement, calcul de `H(Emotion | marqueur)`.
2. Pour chaque mode, selection des marqueurs presents dans ce mode.
3. Moyenne des entropies emotionnelles de ces marqueurs.

C'est exactement la logique de `compute_entropy_by_mode`.

Cette analyse repond a la question :

> Les marqueurs presents dans un mode donne sont-ils plus ou moins
> specifiques d'une emotion ?

### Entropie modale par emotion

Une analyse symetrique a ete ajoutee :

1. Pour chaque corpus separement, calcul de `H(Mode | marqueur)`.
2. Pour chaque emotion, selection des marqueurs presents avec cette emotion.
3. Moyenne des entropies modales de ces marqueurs.

Cette analyse repond a la question :

> Les marqueurs presents avec une emotion donnee sont-ils plus ou moins
> specifiques d'un mode ?

Cette seconde analyse est informative, mais elle est plus fragile car plusieurs
emotions sont tres faiblement representees dans le XLSX.

## Metriques de rapprochement

Pour chaque groupe compare, deux distances XLSX/Glozz ont ete calculees :

- `absdiff_mean` : difference absolue entre la moyenne XLSX et la moyenne Glozz ;
- `wasserstein` : distance de Wasserstein entre les distributions de marqueurs
  individuelles.

Un passage a la pipeline 2 est considere comme un rapprochement si la distance
XLSX/Glozz diminue.

## Tailles observees

Nombre de lignes de marqueurs apres separation XLSX/Glozz :

```text
pipeline_1 brute
  XLSX  : 4433 lignes de marqueurs
  Glozz : 61126 lignes de marqueurs

pipeline_1 controlee stopwords
  XLSX  : 2269 lignes de marqueurs
  Glozz : 31809 lignes de marqueurs

pipeline_2
  XLSX  : 2269 lignes de marqueurs
  Glozz : 18333 lignes de marqueurs
```

Nombre de marqueurs retenus pour `H(Emotion | marqueur)` :

```text
pipeline_1 brute
  XLSX  : 310 marqueurs, entropie moyenne = 0.6216
  Glozz : 2989 marqueurs, entropie moyenne = 1.1904

pipeline_1 controlee stopwords
  XLSX  : 171 marqueurs, entropie moyenne = 0.5238
  Glozz : 2751 marqueurs, entropie moyenne = 1.1005

pipeline_2
  XLSX  : 171 marqueurs, entropie moyenne = 0.5238
  Glozz : 1488 marqueurs, entropie moyenne = 0.9198
```

Nombre de marqueurs retenus pour `H(Mode | marqueur)` :

```text
pipeline_1 brute
  XLSX  : 310 marqueurs, entropie moyenne = 0.5574
  Glozz : 2989 marqueurs, entropie moyenne = 0.7478

pipeline_1 controlee stopwords
  XLSX  : 171 marqueurs, entropie moyenne = 0.3782
  Glozz : 2751 marqueurs, entropie moyenne = 0.6985

pipeline_2
  XLSX  : 171 marqueurs, entropie moyenne = 0.3782
  Glozz : 1488 marqueurs, entropie moyenne = 0.4153
```

Ces chiffres montrent deja que la restriction aux declencheurs reduit fortement
le nombre de marqueurs Glozz retenus et abaisse les entropies Glozz.

## Resultat specifique : mode `Designee`

La premiere hypothese testee portait sur le mode `Designee`.

### Comparaison brute

```text
pipeline_1 brute
  XLSX  : n = 29, moyenne = 0.830379, mediane = 0.811278
  Glozz : n = 1342, moyenne = 1.256302, mediane = 1.295462
  ecart des moyennes = 0.425923
  distance de Wasserstein = 0.507258
  KS p = 0.000008
  Mann-Whitney p = 0.002098

pipeline_2
  XLSX  : n = 11, moyenne = 0.541794, mediane = 0.439497
  Glozz : n = 420, moyenne = 0.535310, mediane = 0.000000
  ecart des moyennes = 0.006484
  distance de Wasserstein = 0.165212
  KS p = 0.809776
  Mann-Whitney p = 0.829639
```

La comparaison brute soutient fortement l'hypothese : la moyenne Glozz devient
presque identique a la moyenne XLSX pour `Designee` en pipeline 2.

### Comparaison controlee stopwords

```text
pipeline_1 controlee stopwords
  XLSX  : n = 11, moyenne = 0.541794, mediane = 0.439497
  Glozz : n = 1154, moyenne = 1.077934, mediane = 0.985228
  ecart des moyennes = 0.536140
  distance de Wasserstein = 0.536140
  KS p = 0.012456
  Mann-Whitney p = 0.015387

pipeline_2
  XLSX  : n = 11, moyenne = 0.541794, mediane = 0.439497
  Glozz : n = 420, moyenne = 0.535310, mediane = 0.000000
  ecart des moyennes = 0.006484
  distance de Wasserstein = 0.165212
  KS p = 0.809776
  Mann-Whitney p = 0.829639
```

Apres controle du filtrage des stopwords, l'effet est encore plus net. La
distance moyenne XLSX/Glozz passe de `0.536140` a `0.006484`.

La mediane appelle toutefois une nuance : en pipeline 2, la mediane Glozz vaut
`0.000000`, ce qui indique que beaucoup de marqueurs Glozz du mode `Designee`
deviennent parfaitement specifiques d'une emotion. La moyenne, la distance de
Wasserstein et les tests non parametriques indiquent neanmoins un rapprochement
global clair.

## Resultat general par mode

Cette section generalise l'analyse aux quatre modes. La comparaison controlee
stopwords est la plus pertinente.

### Valeurs controlees par mode

`H(Emotion | marqueur)`, moyenne des marqueurs presents dans chaque mode :

| Mode | XLSX p1 filtree | Glozz p1 filtree | XLSX p2 | Glozz p2 |
|---|---:|---:|---:|---:|
| Comportementale | 0.394163 | 1.110022 | 0.394163 | 0.882210 |
| Designee | 0.541794 | 1.077934 | 0.541794 | 0.535310 |
| Montree | 0.533618 | 1.561042 | 0.533618 | 1.342859 |
| Suggeree | 0.628371 | 1.248188 | 0.628371 | 1.129588 |

Nombre de marqueurs distincts dans les distributions controlees :

| Mode | XLSX | Glozz p1 filtree | Glozz p2 |
|---|---:|---:|---:|
| Comportementale | 33 | 1521 | 557 |
| Designee | 11 | 1154 | 420 |
| Montree | 163 | 495 | 316 |
| Suggeree | 54 | 2268 | 1058 |

### Distances controlees par mode

| Mode | ecart p1 filtree | ecart p2 | delta | effet |
|---|---:|---:|---:|---|
| Comportementale | 0.715859 | 0.488047 | -0.227812 | rapproche |
| Designee | 0.536140 | 0.006484 | -0.529656 | rapproche |
| Montree | 1.027424 | 0.809241 | -0.218183 | rapproche |
| Suggeree | 0.619817 | 0.501217 | -0.118600 | rapproche |

Synthese controlee :

```text
nombre de modes rapproches     : 4/4
nombre de modes eloignes       : 0/4
ecart moyen XLSX/Glozz p1      : 0.724810
ecart moyen XLSX/Glozz p2      : 0.451247
RMSE p1                        : 0.748277
RMSE p2                        : 0.534865
Wasserstein moyen p1           : 0.724810
Wasserstein moyen p2           : 0.490929
```

Conclusion pour les modes : la restriction aux spans `Declencheur` rapproche
systematiquement les scores Glozz des scores XLSX.

### Comparaison brute par mode

La comparaison brute, bien que confondue par le changement de filtrage des
stopwords, donne la meme orientation.

| Mode | ecart p1 brute | ecart p2 | delta | effet |
|---|---:|---:|---:|---|
| Comportementale | 0.653593 | 0.488047 | -0.165546 | rapproche |
| Designee | 0.425923 | 0.006484 | -0.419439 | rapproche |
| Montree | 1.145513 | 0.809241 | -0.336273 | rapproche |
| Suggeree | 0.573527 | 0.501217 | -0.072310 | rapproche |

Synthese brute :

```text
nombre de modes rapproches     : 4/4
ecart moyen XLSX/Glozz p1      : 0.699639
ecart moyen XLSX/Glozz p2      : 0.451247
Wasserstein moyen p1           : 0.719973
Wasserstein moyen p2           : 0.490929
```

## Resultat general par emotion

L'analyse par emotion repose sur `H(Mode | marqueur)`. Elle est plus fragile :
dans le XLSX, plusieurs emotions n'ont qu'un nombre infime de marqueurs apres
filtrage.

### Valeurs controlees par emotion

| Emotion | n XLSX | XLSX | n Glozz p1 | Glozz p1 | n Glozz p2 | Glozz p2 |
|---|---:|---:|---:|---:|---:|---:|
| Colere | 167 | 0.387295 | 1585 | 0.839570 | 679 | 0.477901 |
| Degout | 87 | 0.292436 | 113 | 0.829202 | 61 | 0.524546 |
| Joie | 10 | 0.506633 | 1114 | 0.755410 | 589 | 0.501652 |
| Peur | 4 | 0.840079 | 1481 | 0.830946 | 632 | 0.450855 |
| Surprise | 4 | 1.284086 | 769 | 0.737165 | 343 | 0.565893 |
| Tristesse | 16 | 0.735903 | 1150 | 0.750776 | 597 | 0.452173 |
| Culpabilite | 1 | 1.250244 | 50 | 0.852436 | 26 | 0.537152 |
| Fierte | 1 | 1.250244 | 575 | 0.694006 | 265 | 0.428711 |
| Jalousie | 2 | 0.000000 | 13 | 0.507757 | 4 | 0.725350 |

### Distances controlees par emotion

| Emotion | ecart p1 filtree | ecart p2 | delta | effet |
|---|---:|---:|---:|---|
| Colere | 0.452274 | 0.090605 | -0.361669 | rapproche |
| Culpabilite | 0.397807 | 0.713091 | 0.315284 | eloigne |
| Degout | 0.536766 | 0.232109 | -0.304657 | rapproche |
| Fierte | 0.556238 | 0.821533 | 0.265295 | eloigne |
| Jalousie | 0.507757 | 0.725350 | 0.217593 | eloigne |
| Joie | 0.248777 | 0.004981 | -0.243796 | rapproche |
| Peur | 0.009133 | 0.389225 | 0.380091 | eloigne |
| Surprise | 0.546922 | 0.718194 | 0.171272 | eloigne |
| Tristesse | 0.014872 | 0.283730 | 0.268858 | eloigne |

Synthese controlee :

```text
nombre d'emotions rapprochees  : 3/9
nombre d'emotions eloignees    : 6/9
ecart moyen XLSX/Glozz p1      : 0.363394
ecart moyen XLSX/Glozz p2      : 0.442091
RMSE p1                        : 0.418921
RMSE p2                        : 0.529093
Wasserstein moyen p1           : 0.430433
Wasserstein moyen p2           : 0.475067
```

Sur l'ensemble des emotions comparables, la restriction aux declencheurs
eloigne donc legerement Glozz de XLSX.

Cette conclusion doit etre lue avec prudence. Les emotions suivantes sont trop
peu representees dans le XLSX pour soutenir une interpretation robuste :

```text
Culpabilite : n XLSX = 1
Fierte      : n XLSX = 1
Jalousie    : n XLSX = 2
Peur        : n XLSX = 4
Surprise    : n XLSX = 4
```

Si l'on ne retient que les emotions un peu representees dans XLSX, par exemple
`n >= 10`, on obtient :

```text
Colere     : rapproche
Degout     : rapproche
Joie       : rapproche
Tristesse  : eloigne
```

Dans ce sous-ensemble moins fragile, l'effet est donc plutot favorable, mais il
n'est pas uniforme.

### Comparaison brute par emotion

La comparaison brute donne une image egalement mitigee :

```text
nombre d'emotions rapprochees  : 4/9
nombre d'emotions eloignees    : 5/9
ecart moyen XLSX/Glozz p1      : 0.265581
ecart moyen XLSX/Glozz p2      : 0.442091
Wasserstein moyen p1           : 0.384190
Wasserstein moyen p2           : 0.475067
```

Cette comparaison brute est cependant moins interpretable, puisqu'elle confond
l'effet du declencheur avec le changement de filtrage des stopwords.

## Interpretation

### Pour les modes

La conclusion est forte : la restriction aux declencheurs rapproche Glozz de
XLSX pour les quatre modes.

L'effet est particulierement marque pour `Designee`, ou l'ecart de moyenne
controle passe de `0.536140` a `0.006484`.

Pour les autres modes, le rapprochement est moins spectaculaire mais
systematique :

- `Comportementale` : l'ecart baisse de `0.227812` ;
- `Montree` : l'ecart baisse de `0.218183` ;
- `Suggeree` : l'ecart baisse de `0.118600`.

L'ordre croissant des entropies Glozz en pipeline 2 devient :

```text
Designee < Comportementale < Suggeree < Montree
```

L'ordre XLSX en pipeline 2 est :

```text
Comportementale < Montree < Designee < Suggeree
```

Les ordres ne sont donc pas identiques, mais les niveaux absolus sont plus
proches en pipeline 2 qu'en pipeline 1 controlee.

### Pour les emotions

La conclusion generale est moins stable. Le passage au declencheur rapproche
nettement certaines emotions (`Colere`, `Degout`, `Joie`), mais eloigne
plusieurs emotions rares ou peu representees dans le XLSX.

Cette instabilite tient a deux raisons principales :

1. Le XLSX est petit une fois l'analyse faite par emotion.
2. Pour certaines emotions, une seule entropie de marqueur XLSX peut fortement
   orienter la moyenne.

Il serait donc imprudent de conclure que le declencheur eloigne vraiment les
scores emotionnels en general. La conclusion la plus rigoureuse est :

- l'effet est favorable pour les emotions suffisamment representees dans XLSX,
  mais pas uniformement ;
- l'analyse par emotion necessite un corpus XLSX plus grand ou un seuil minimal
  de representation plus strict.

## Limites methodologiques

1. Les marqueurs ne sont pas des observations independantes au sens statistique
   strict. Un meme marqueur peut contribuer a plusieurs groupes s'il apparait
   dans plusieurs modes ou emotions.
2. Les tests non parametriques rapportes pour `Designee` comparent des
   distributions de marqueurs, mais ils ne resolvent pas la dependance
   structurelle entre marqueurs, segments et corpus.
3. Le corpus XLSX est petit. C'est particulierement critique pour l'analyse par
   emotion.
4. Le controle stopwords a ete fait par suppression des lignes de marqueurs
   deja extraites dans `pipeline_1`. Cela correspond au changement de filtrage
   observe, mais ce n'est pas une rerun complete de toute la pipeline 1.
5. `pipeline_2` utilise le declencheur pour 6808 unites Glozz sur 6811 ; trois
   unites restent fondees sur le segment complet.
6. Les distances de moyenne et de Wasserstein sont descriptives. Elles mesurent
   un rapprochement numerique des distributions, pas une equivalence
   annotationnelle complete entre les corpus.

## Conclusion

Si l'on parle des scores produits par la logique principale de
`marker_specificity.py`, c'est-a-dire `H(Emotion | marqueur)` agrege par mode,
la reponse est nette :

> Se restreindre au span `Declencheur` rapproche les scores d'entropie Glozz des
> scores XLSX pour les quatre modes.

L'effet reste visible apres controle du filtrage des stopwords, ce qui indique
qu'il ne s'agit pas seulement d'un artefact du changement de filtrage lexical.

Pour l'analyse symetrique par emotion, la reponse est plus nuancee :

> Le rapprochement n'est pas global sur les neuf emotions comparables, mais les
> emotions suffisamment representees dans le XLSX tendent plutot a se
> rapprocher.

La conclusion robuste est donc la suivante :

- **par mode** : effet de rapprochement clair et systematique ;
- **par emotion** : effet heterogene, fortement contraint par la petite taille
  du XLSX.
