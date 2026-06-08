# Rapport de Corrélation entre HATE et Modes d'Expression (CyberAdoAgg)

Ce rapport présente l'analyse statistique de la relation entre les catégories d'agressivité de la colonne `HATE` (OAG, CAG, NAG) et les 4 modes d'expression émotionnelle dans le jeu de données [CyberAdoAgg_gold_global_total_latest.xlsx](file:///workspaces/workspace/ExpressionEmotionnelle/data/raw/xlsx/CyberAdoAgg_gold_global_total_latest.xlsx).

> [!NOTE]
> Les 4 modes analysés sont :
> - **Suggeree** (Émotion suggérée)
> - **Montree** (Émotion montrée/exprimée ouvertement)
> - **Comportementale** (Émotion se manifestant par le comportement)
> - **Designee** (Émotion désignée nominativement)

---

## 1. Distribution des Modes d'Expression par Catégorie HATE

Le tableau ci-dessous indique le pourcentage de présence de chaque mode d'expression au sein des différentes catégories de messages (`HATE`).

| Catégorie HATE | Suggeree | Montree | Comportementale | Designee |
| :--- | :---: | :---: | :---: | :---: |
| **CAG** | 3.23% | 21.51% | 1.08% | 2.15% |
| **NAG** | 7.28% | 20.69% | 5.36% | 6.13% |
| **OAG** | 7.49% | **52.93%** | 4.92% | **8.67%** |

---

## 2. Analyse de Corrélation Linéaire (Pearson r / Coefficient Phi)

En encodant la colonne `HATE` en variables binaires (one-hot encoding), nous mesurons la force et la direction de la corrélation linéaire avec chaque mode binaire.

### Coefficients de Corrélation ($r$)
| Catégorie HATE | Suggeree | Montree | Comportementale | Designee |
| :--- | :---: | :---: | :---: | :---: |
| **CAG** | -0.05 | -0.13 | -0.06 | -0.07 |
| **NAG** | 0.01 | -0.26 | 0.03 | -0.03 |
| **OAG** | 0.03 | **0.33** | 0.02 | 0.07 |

### Degré de Significativité ($p$-values)
| Catégorie HATE | Suggeree | Montree | Comportementale | Designee |
| :--- | :---: | :---: | :---: | :---: |
| **CAG** | 0.1356 | **0.0003** | 0.0835 | **0.0496** |
| **NAG** | 0.7758 | **0.0000** | 0.4768 | 0.4810 |
| **OAG** | 0.4835 | **0.0000** | 0.6520 | 0.0517 |

---

## 3. Tests d'Association Globale (Chi-Deux & V de Cramér)

Pour évaluer si un mode d'expression est associé globalement au type d'agressivité `HATE`, nous utilisons le test d'indépendance du Chi-Deux. L'intensité de cette association est mesurée par le **V de Cramér**.

| Mode | Statistique $\chi^2$ | $p$-value | Degrés de liberté | V de Cramér | Association Significative ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Montree** | 83.9174 | < 0.0001 | 2 | 0.3278 | **Oui** |
| **Designee** | 5.4489 | 0.0656 | 2 | 0.0835 | Non (Proche du seuil) |
| **Comportementale** | 3.0722 | 0.2152 | 2 | 0.0627 | Non |
| **Suggeree** | 2.2431 | 0.3258 | 2 | 0.0536 | Non |

---

## 4. Conclusions et Interprétation

1. **Le mode "Montree" est fortement associé à l'agressivité** :
   - C'est le seul mode d'expression montrant une association globale statistiquement significative et forte (V de Cramér = **0.3278**, $p < 0.0001$).
   - Plus de la moitié des messages avec agressivité explicite (**52.93%** de OAG) affichent des émotions montrées, contre seulement ~21% pour les messages non agressifs (NAG) ou implicitement agressifs (CAG).
   - On observe une corrélation positive significative avec **OAG** ($r = +0.33$) et négative avec **NAG** ($r = -0.26$) et **CAG** ($r = -0.13$).
2. **Le mode "Designee" montre une tendance faible** :
   - Ce mode a une présence légèrement plus forte dans les messages OAG (8.67%) par rapport à NAG (6.13%) et CAG (2.15%).
   - La corrélation négative avec **CAG** est tout juste significative ($r = -0.07$, $p = 0.0496$). L'association globale reste cependant non significative au seuil standard de 5% ($p = 0.0656$).
3. **Absence de lien pour les modes "Suggeree" et "Comportementale"** :
   - Ces deux modes d'expression d'émotions n'ont aucun lien statistique significatif avec le fait qu'un message soit agressif (explicite ou implicite) ou non agressif.

---

## 5. Fichiers de Résultats Générés

Les résultats ont été sauvegardés sous forme de fichiers CSV dans le répertoire des résultats :
- [distribution_modes_par_hate.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/distribution_modes_par_hate.csv) : Proportions de présence des modes.
- [correlation_pearson_hate_modes.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/correlation_pearson_hate_modes.csv) : Matrice des corrélations de Pearson (Phi).
- [correlation_pvalue_hate_modes.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/correlation_pvalue_hate_modes.csv) : Valeurs de significativité des corrélations ($p$-values).
- [association_chisquare_cramersv_hate_modes.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/association_chisquare_cramersv_hate_modes.csv) : Résultats des tests du Chi-Deux et V de Cramér.
