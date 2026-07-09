# Rapport de comparaison des corpus : proportions d'émotions et de modes

Ce rapport examine le script `tools/compute_corpus_label_proportions.py` et les
résultats produits dans `results/corpus_label_proportions.csv`. Il compare deux
ensembles :

- **CyberAggAdo** : annotations issues de
  `data/raw/xlsx/CyberAdoAgg_gold_global_total_latest.xlsx`.
- **TTK-Glozz** : annotations issues de `results/glozz/annotations.csv`, qui
  agrègent les sous-corpus Glozz `PtitLibe`, `LitteratureJeunesse`,
  `Albert_dataset` et `CorpusCovid`.

## 1. Ce que calcule le script

Le script produit trois mesures distinctes :

| Mesure | Dénominateur | Interprétation |
|:---|:---|:---|
| `emotion_assignment_share` | Toutes les assignations de catégories émotionnelles | Mesure de composition : une unité multi-étiquette contribue une fois par émotion. Les pourcentages somment à 100 % par corpus. |
| `emotion_unit_presence` | Toutes les unités annotées comme émotionnelles | Mesure de présence par unité : une même unité peut être comptée dans plusieurs émotions. Les pourcentages peuvent donc dépasser 100 % au total. |
| `mode_unit_share` | Toutes les unités avec un mode valide | Distribution des modes. Dans TTK-Glozz, les unités `Autre` n'ont pas de mode et sont exclues de ce dénominateur. |

Chargement des données :

- **CyberAggAdo** : le script ne conserve que les lignes `Emo == 1`, lit les
  spans `span1` à `span4`, exige un mode et au moins une catégorie, puis sépare
  les catégories multiples notées avec `+`. Les doublons de span dans une même
  ligne sont fusionnés par `(source_row, text_span, mode)` en conservant l'union
  des émotions.
- **TTK-Glozz** : le script lit les unités `SitEmo` avec `categorie1` et
  `categorie2`, exige un mode pour ces unités, puis ajoute les unités `Autre`
  comme émotion `Autre` sans mode.

La relance du script donne :

| Corpus | Unités émotionnelles | Assignations émotionnelles | Unités avec mode |
|:---|---:|---:|---:|
| CyberAggAdo | 491 | 572 | 491 |
| TTK-Glozz | 8 243 | 8 937 | 6 811 |

TTK-Glozz est donc beaucoup plus volumineux. Dans ce corpus, les 1 432 unités
`Autre` contribuent aux proportions émotionnelles, mais pas aux proportions de
mode.

## 2. Structure multi-étiquette

Ici, une unité est dite **multi-étiquette** lorsqu'elle porte plusieurs
catégories émotionnelles. Par exemple, dans CyberAggAdo, une cellule comme
`Dégoût + Colère` compte comme une unité multi-étiquette. Dans TTK-Glozz, cela
correspond aux unités `SitEmo` qui ont à la fois `categorie1` et `categorie2`
renseignées avec deux émotions distinctes.

Le calcul ci-dessous inclut `Autre`, puisque le script traite `Autre` comme une
catégorie émotionnelle dans les mesures d'émotion. Cela a deux effets :

- une unité annotée seulement `Autre` compte comme une unité simple, pas comme
  une unité multi-étiquette ;
- une unité qui combine `Autre` avec une autre émotion compte comme
  multi-étiquette. Ce cas existe dans CyberAggAdo, mais pas dans TTK-Glozz où
  les unités `Autre` sont séparées et sans mode.

| Corpus | Unités multi-étiquettes | Part des unités | Assignations par unité |
|:---|---:|---:|---:|
| CyberAggAdo | 77 / 491 | 15,68 % | 1,165 |
| TTK-Glozz | 694 / 8 243 | 8,42 % | 1,084 |

Si l'on retire `Autre`, le tableau change :

| Corpus | Unités conservées | Assignations non-`Autre` | Unités multi-étiquettes non-`Autre` | Part des unités conservées | Assignations par unité conservée |
|:---|---:|---:|---:|---:|---:|
| CyberAggAdo | 459 | 505 | 46 | 10,02 % | 1,100 |
| TTK-Glozz | 6 811 | 7 505 | 694 | 10,19 % | 1,102 |

La lecture change donc nettement : **avec `Autre`, CyberAggAdo paraît plus
multi-étiqueté ; sans `Autre`, les deux corpus ont un taux de multi-étiquetage
très proche**. La différence initiale vient surtout du statut particulier de
`Autre` : 67 unités CyberAggAdo contiennent `Autre`, dont 35 en combinaison avec
une autre émotion, tandis que les 1 432 unités `Autre` de TTK-Glozz sont des
unités séparées à étiquette unique.

## 3. Comparaison des émotions

### 3.1 Composition par assignation émotionnelle

| Émotion | CyberAggAdo | TTK-Glozz | Écart Cyber - TTK |
|:---|---:|---:|---:|
| Colère | 59,44 % | 19,26 % | +40,18 pts |
| Dégoût | 16,78 % | 0,90 % | +15,89 pts |
| Autre | 11,71 % | 16,02 % | -4,31 pts |
| Joie | 5,42 % | 13,68 % | -8,27 pts |
| Tristesse | 2,62 % | 10,63 % | -8,01 pts |
| Jalousie | 1,05 % | 0,10 % | +0,95 pt |
| Surprise | 0,87 % | 13,80 % | -12,92 pts |
| Peur | 0,70 % | 16,62 % | -15,92 pts |
| Culpabilité | 0,52 % | 0,28 % | +0,24 pt |
| Fierté | 0,52 % | 3,01 % | -2,49 pts |
| Embarras | 0,35 % | 2,66 % | -2,31 pts |
| Admiration | 0,00 % | 3,04 % | -3,04 pts |

CyberAggAdo est très concentré sur deux catégories : **Colère** et **Dégoût**
représentent ensemble 76,22 % des assignations émotionnelles. Cette concentration
est cohérente avec la nature du corpus, centré sur des échanges de
cyberagression et de cyberharcèlement.

TTK-Glozz est nettement plus diversifié : **Colère**, **Peur**, **Autre**,
**Surprise**, **Joie** et **Tristesse** se situent toutes entre 10 % et 20 % des
assignations, à l'exception de `Autre` qui reste dans la même zone. Le **Dégoût**
y est presque absent, alors qu'il est la deuxième catégorie de CyberAggAdo.

### 3.2 Présence par unité

La mesure par unité confirme le même contraste :

| Émotion | CyberAggAdo | TTK-Glozz |
|:---|---:|---:|
| Colère | 69,25 % | 20,88 % |
| Dégoût | 19,55 % | 0,97 % |
| Peur | 0,81 % | 18,02 % |
| Surprise | 1,02 % | 14,96 % |
| Joie | 6,31 % | 14,84 % |
| Tristesse | 3,06 % | 11,52 % |
| Autre | 13,65 % | 17,37 % |

CyberAggAdo fait apparaître la colère dans près de sept unités sur dix. À
l'inverse, TTK-Glozz distribue la présence émotionnelle entre plusieurs émotions
de base, notamment peur, surprise, joie et tristesse.

### 3.3 Émotions de base, complexes et `Autre`

| Groupe | CyberAggAdo | TTK-Glozz |
|:---|---:|---:|
| Émotions de base | 85,84 % | 74,88 % |
| Émotions complexes | 2,45 % | 9,10 % |
| Autre | 11,71 % | 16,02 % |

TTK-Glozz contient une part plus importante d'émotions complexes, notamment
**Admiration**, **Fierté** et **Embarras**. CyberAggAdo, lui, mobilise presque
exclusivement les émotions de base, avec une domination très forte de la colère.

## 4. Comparaison des modes d'expression

| Mode | CyberAggAdo | TTK-Glozz | Écart Cyber - TTK |
|:---|---:|---:|---:|
| Montrée | 70,26 % | 20,91 % | +49,36 pts |
| Désignée | 11,41 % | 24,78 % | -13,38 pts |
| Suggérée | 11,00 % | 33,02 % | -22,02 pts |
| Comportementale | 7,33 % | 21,29 % | -13,96 pts |

Le contraste le plus massif concerne le mode **Montrée** : il représente plus de
70 % des unités CyberAggAdo, contre environ 21 % dans TTK-Glozz. Cela suggère
que les émotions de CyberAggAdo passent fortement par la forme de l'énoncé :
ponctuation expressive, graphies, interjections, intensification, formulations
directes ou agressives.

TTK-Glozz présente une distribution plus équilibrée. Le mode **Suggérée** y est
majoritaire, suivi de **Désignée** et **Comportementale**. Cette répartition est
compatible avec des textes narratifs, journalistiques ou explicatifs, où
l'émotion est souvent inférée depuis une situation décrite, nommée lexicalement,
ou portée par la description d'un comportement.

## 5. Interprétation comparative

1. **CyberAggAdo est un corpus émotionnellement polarisé.**  
   La colère y domine très fortement, et le dégoût est beaucoup plus présent que
   dans TTK-Glozz. Les autres émotions de base, en particulier peur, surprise,
   joie et tristesse, sont marginales.

2. **TTK-Glozz est plus diversifié.**  
   Les émotions y couvrent un spectre plus large. La peur, la surprise, la joie
   et la tristesse y occupent une place comparable ou supérieure à la colère,
   alors que ces catégories sont faibles dans CyberAggAdo.

3. **Le mode d'expression distingue fortement les corpus.**  
   CyberAggAdo est dominé par l'émotion montrée. TTK-Glozz repose davantage sur
   l'émotion suggérée, désignée ou comportementale. La différence de genre et de
   situation d'énonciation est probablement déterminante : messages d'interaction
   conflictuelle d'un côté, textes plus narratifs ou informatifs de l'autre.

4. **Les labels `Autre` ne sont pas strictement comparables.**  
   Dans CyberAggAdo, `Autre` est une catégorie portée par des spans avec mode.
   Dans TTK-Glozz, une partie importante de `Autre` correspond à des unités
   séparées sans mode. Cela rend la comparaison de `Autre` utile descriptivement,
   mais plus fragile que celle des onze catégories émotionnelles principales.

## 6. Limites et recommandations

- Les résultats sont **descriptifs** : le script ne réalise pas de test
  statistique ni de calcul de taille d'effet.
- Les proportions comparent des **unités annotées**, pas des messages, des
  documents, des phrases ou des tokens. Elles ne doivent donc pas être lues comme
  une fréquence émotionnelle brute dans les corpus complets.
- TTK-Glozz est un agrégat de sous-corpus hétérogènes. Une analyse ultérieure
  devrait produire les mêmes proportions par sous-corpus (`PtitLibe`,
  `LitteratureJeunesse`, `Albert_dataset`, `CorpusCovid`) pour vérifier si la
  diversité observée est homogène ou portée par un sous-ensemble.
- Pour comparer finement les modes, il serait utile de produire une variante
  `SitEmo` uniquement, excluant `Autre` aussi des proportions émotionnelles, afin
  d'aligner complètement le périmètre émotion/mode entre les deux corpus.

## 7. Conclusion

Le script est cohérent avec son objectif : il sépare clairement composition des
assignations, présence par unité et distribution des modes. Les résultats
montrent une opposition nette entre les corpus. **CyberAggAdo** est petit,
fortement multi-étiqueté, polarisé vers **Colère/Dégoût** et dominé par le mode
**Montrée**. **TTK-Glozz** est beaucoup plus volumineux, plus diversifié dans ses
catégories émotionnelles, et repose davantage sur les modes **Suggérée**,
**Désignée** et **Comportementale**.
