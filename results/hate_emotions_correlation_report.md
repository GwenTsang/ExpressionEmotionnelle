# Rapport de Corrélation entre HATE et Émotions (CyberAdoAgg)

Ce rapport présente l'analyse statistique de la relation entre les catégories d'agressivité de la colonne `HATE` (OAG, CAG, NAG) et les 12 émotions annotées dans le jeu de données [CyberAdoAgg_gold_global_total_latest.xlsx](file:///workspaces/workspace/ExpressionEmotionnelle/data/raw/xlsx/CyberAdoAgg_gold_global_total_latest.xlsx).

> [!NOTE]
> Le dataset contient **781 lignes valides** après filtrage des valeurs manquantes dans la colonne `HATE`.
> Les catégories de `HATE` évaluées sont :
> - **OAG** (Overtly Aggressive / Agressivité explicite) : 427 lignes
> - **CAG** (Covertly Aggressive / Agressivité implicite) : 93 lignes
> - **NAG** (Non Aggressive / Non agressif) : 261 lignes

---

## 1. Distribution des Émotions par Catégorie HATE

Le tableau ci-dessous indique la proportion (en pourcentage) de présence de chaque émotion au sein des différentes catégories de messages (`HATE`).

| Catégorie HATE | Admiration | Autre | Colere | Culpabilite | Degout | Embarras | Fierte | Jalousie | Joie | Peur | Surprise | Tristesse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAG** | 0.0% | 6.45% | 17.20% | 0.0% | 3.23% | 0.0% | 0.0% | 0.0% | 3.23% | 0.0% | 1.08% | 0.0% |
| **NAG** | 0.0% | 4.60% | 25.29% | 1.15% | 1.15% | 0.0% | 1.15% | 0.77% | 3.07% | 0.38% | 0.0% | 3.07% |
| **OAG** | 0.0% | 10.30% | 48.48% | 0.0% | 17.33% | 0.47% | 0.0% | 0.94% | 4.45% | 0.70% | 0.94% | 1.41% |

> [!TIP]
> - La **Colère** est l'émotion dominante, particulièrement dans les messages **OAG** où elle est présente dans près de la moitié des cas (**48.48%**).
> - Le **Dégoût** est également très présent dans les messages **OAG** (**17.33%**), mais presque inexistant dans les autres catégories.

---

## 2. Analyse de Corrélation Linéaire (Pearson r / Coefficient Phi)

En encodant la colonne `HATE` en variables binaires (one-hot encoding), nous pouvons calculer le coefficient de corrélation de Pearson (équivalent au coefficient Phi pour deux variables binaires) pour mesurer la force et la direction de l'association.

### Coefficients de Corrélation ($r$)
| Catégorie HATE | Admiration | Autre | Colere | Culpabilite | Degout | Embarras | Fierte | Jalousie | Joie | Peur | Surprise | Tristesse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAG** | NaN | -0.02 | -0.15 | -0.02 | -0.09 | -0.02 | -0.02 | -0.03 | -0.01 | -0.03 | 0.02 | -0.05 |
| **NAG** | NaN | -0.09 | -0.17 | 0.09 | -0.21 | -0.04 | 0.09 | -0.00 | -0.03 | -0.01 | -0.06 | 0.07 |
| **OAG** | NaN | 0.10 | 0.26 | -0.07 | 0.26 | 0.05 | -0.07 | 0.02 | 0.03 | 0.03 | 0.04 | -0.03 |

### Degré de Significativité ($p$-values)
| Catégorie HATE | Admiration | Autre | Colere | Culpabilite | Degout | Embarras | Fierte | Jalousie | Joie | Peur | Surprise | Tristesse |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAG** | NaN | 0.5726 | **0.0000** | 0.5241 | **0.0174** | 0.6032 | 0.5241 | 0.3666 | 0.7425 | 0.4616 | 0.5757 | 0.1655 |
| **NAG** | NaN | **0.0144** | **0.0000** | **0.0143** | **0.0000** | 0.3164 | **0.0143** | 0.9965 | 0.4246 | 0.7209 | 0.1123 | 0.0577 |
| **OAG** | NaN | **0.0072** | **0.0000** | 0.0568 | **0.0000** | 0.1978 | 0.0568 | 0.5542 | 0.3319 | 0.4136 | 0.2543 | 0.3708 |

> [!NOTE]
> Les coefficients pour **Admiration** sont marqués `NaN` car cette émotion n'est présente dans aucun message de l'ensemble du dataset (écart-type nul).

---

## 3. Tests d'Association Globale (Chi-Deux & V de Cramér)

Pour évaluer si la distribution d'une émotion varie de manière significative globalement en fonction des 3 catégories de `HATE`, nous appliquons le test d'indépendance du Chi-Deux. L'intensité de cette association est mesurée par le **V de Cramér** (compris entre 0 et 1).

| Émotion | Statistique $\chi^2$ | $p$-value | Degrés de liberté | V de Cramér | Association Significative ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Colere** | 55.1250 | < 0.0001 | 2 | 0.2657 | **Oui** |
| **Degout** | 51.7835 | < 0.0001 | 2 | 0.2575 | **Oui** |
| **Autre** | 7.5377 | 0.0231 | 2 | 0.0982 | **Oui** |
| **Culpabilite** | 6.0001 | 0.0498 | 2 | 0.0877 | **Oui** |
| **Fierte** | 6.0001 | 0.0498 | 2 | 0.0877 | **Oui** |
| **Tristesse** | 4.4625 | 0.1074 | 2 | 0.0756 | Non |
| **Surprise** | 2.5488 | 0.2796 | 2 | 0.0571 | Non |
| **Embarras** | 1.6623 | 0.4355 | 2 | 0.0461 | Non |
| **Joie** | 0.9489 | 0.6222 | 2 | 0.0349 | Non |
| **Jalousie** | 0.8791 | 0.6443 | 2 | 0.0335 | Non |
| **Peur** | 0.8679 | 0.6480 | 2 | 0.0333 | Non |
| **Admiration** | NaN | NaN | NaN | NaN | Non (Constante vide) |

---

## 4. Conclusions et Interprétation

1. **Association forte et positive avec l'agressivité explicite (OAG)** :
   - La **Colère** ($r = +0.26$, $p < 0.001$) et le **Dégoût** ($r = +0.26$, $p < 0.001$) sont significativement plus fréquents dans les messages contenant de l'agressivité explicite (OAG).
   - L'association globale pour ces deux émotions est statistiquement très robuste (V de Cramér de ~0.26).
2. **Corrélation négative avec l'agressivité implicite (CAG) et l'absence d'agressivité (NAG)** :
   - La **Colère** montre des corrélations négatives significatives avec CAG ($r = -0.15$) et NAG ($r = -0.17$).
   - Le **Dégoût** présente une corrélation négative significative avec CAG ($r = -0.09$) et surtout NAG ($r = -0.21$).
3. **Émotions exclusives aux messages non agressifs (NAG)** :
   - Bien que rares, la **Culpabilité** ($p = 0.0498$) et la **Fierté** ($p = 0.0498$) ne sont observées que dans les messages **NAG** (1.15% chacun).
4. **Autres émotions** :
   - L'émotion **Autre** présente une faible corrélation positive avec OAG et négative avec NAG.
   - Les autres émotions (Tristesse, Joie, Peur, Surprise, Jalousie, Embarras) n'ont pas d'association statistiquement significative avec le type d'agressivité.
   - L'**Admiration** n'est pas représentée dans ce dataset.

---

## 5. Fichiers de Résultats Générés

Les résultats ont été sauvegardés sous forme de fichiers CSV dans le répertoire des résultats :
- [distribution_emotions_par_hate.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/distribution_emotions_par_hate.csv) : Proportions de présence des émotions.
- [correlation_pearson_hate_emotions.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/correlation_pearson_hate_emotions.csv) : Matrice des coefficients de corrélation de Pearson (Phi).
- [correlation_pvalue_hate_emotions.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/correlation_pvalue_hate_emotions.csv) : Valeurs de significativité des corrélations ($p$-values).
- [association_chisquare_cramersv_hate_emotions.csv](file:///workspaces/workspace/ExpressionEmotionnelle/results/correlation/association_chisquare_cramersv_hate_emotions.csv) : Résultats des tests du Chi-Deux et V de Cramér.
