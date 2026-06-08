# Analyse des variantes flexionnelles et des artefacts de lemmatisation

## Objectif du rapport

Objectif central : distinguer ce qui relève d'un enrichissement morphologique utile de ce qui relève d'artefacts introduits par le lemmatizer ou par une correction fuzzy.

## Contexte empirique

Trois runs comparatifs ont été lancés sur `data/pipeline_2/SimpleSitEmo.parquet`, avec élimination des stopwords dans les trois cas :

| Run | Dossier de sortie | Lemmatisation | Stopwords |
|---|---|---:|---:|
| SpaCy | `results/pipeline_2_compare_spacy` | oui, `spacy` | filtrés |
| Stanza | `results/pipeline_2_compare_stanza` | oui, `stanza` | filtrés |
| Sans lemmatizer | `results/pipeline_2_compare_no_lemma` | non | filtrés |

La pipeline actuelle extrait toujours les mots de surface et la ponctuation. Quand un lemmatizer est activé, elle ajoute les lemmes comme un troisième type de marqueur, avec `marker_type = "lemma"` et `marker_value = <lemme>`. La spécificité est ensuite calculée sur tous les marqueurs présents dans `markers.csv`, sauf filtrage explicite effectué en dehors de `run_analysis.py`.

Conséquence importante : les runs SpaCy et Stanza ne sont pas des analyses exclusivement lemmatisées. Ce sont des analyses mixtes `word + punctuation + lemma`.

## Résultats observés

### Volumétrie des marqueurs

| Run | Lignes de marqueurs | `word` | `lemma` | `punctuation` | Marqueurs avec fréquence >= 3 |
|---|---:|---:|---:|---:|---:|
| Sans lemmatizer | 11 776 | 9 551 | 0 | 2 225 | 882 |
| SpaCy | 20 602 | 9 551 | 8 826 | 2 225 | 1 660 |
| Stanza | 20 859 | 9 551 | 9 083 | 2 225 | 1 639 |

Le gain principal de la lemmatisation est donc une augmentation du nombre de marqueurs analysables au seuil `min_freq = 3`.

### Entropie émotionnelle moyenne

| Run | Nombre de marqueurs freq >= 3 | H emotion moyenne | H emotion médiane |
|---|---:|---:|---:|
| Sans lemmatizer | 882 | 0.907 | 0.920 |
| SpaCy | 1 660 | 0.928 | 0.920 |
| Stanza | 1 639 | 0.937 | 0.920 |

Les distributions globales d'entropie changent peu. L'ajout de lemmes enrichit la couverture, mais ne modifie pas fortement la structure générale de l'analyse.

### Entropie émotionnelle par type de marqueur

| Run | Type | Nombre freq >= 3 | H emotion moyenne |
|---|---|---:|---:|
| Sans lemmatizer | word | 871 | 0.895 |
| Sans lemmatizer | punctuation | 11 | 1.832 |
| SpaCy | word | 871 | 0.895 |
| SpaCy | lemma | 778 | 0.953 |
| SpaCy | punctuation | 11 | 1.832 |
| Stanza | word | 871 | 0.895 |
| Stanza | lemma | 757 | 0.972 |
| Stanza | punctuation | 11 | 1.832 |

Les lignes `word` et `punctuation` sont strictement identiques entre les trois runs. Les différences observées dans les runs avec lemmatizer viennent uniquement des lignes `lemma`.

### Entropie par mode

| Mode | Sans lemmatizer | SpaCy | Stanza |
|---|---:|---:|---:|
| Comportementale | 0.854 | 0.907 | 0.919 |
| Désignée | 0.563 | 0.582 | 0.602 |
| Montrée | 1.222 | 1.231 | 1.237 |
| Suggérée | 1.147 | 1.145 | 1.156 |

La hiérarchie reste stable : `Désignée` conserve une entropie plus faible, tandis que `Montrée` et `Suggérée` restent plus dispersés.

Le test de Kruskal-Wallis reste significatif dans les trois runs. La différence qualitative notable concerne la comparaison `Montrée` versus `Suggérée` :

| Run | `Montrée` vs `Suggérée` |
|---|---|
| Sans lemmatizer | non significatif, `p = 0.11` |
| SpaCy | significatif, `p = 0.02` |
| Stanza | significatif, `p = 0.02` |

Ce point doit être traité comme un signal à investiguer, pas comme une preuve que la lemmatisation améliore automatiquement l'analyse.

## Comparaison SpaCy versus Stanza

Sur les lemmes passant le seuil de fréquence pour l'entropie émotionnelle :

| Mesure | Valeur |
|---|---:|
| Lemmes SpaCy freq >= 3 | 778 |
| Lemmes Stanza freq >= 3 | 757 |
| Intersection | 637 |
| Lemmes seulement SpaCy | 141 |
| Lemmes seulement Stanza | 120 |
| Jaccard SpaCy/Stanza | 0.709 |
| Corrélation des fréquences sur lemmes communs | 0.913 |
| Corrélation des entropies sur lemmes communs | 0.963 |
| Différence absolue moyenne d'entropie sur lemmes communs | 0.066 |

La structure globale est donc très proche entre SpaCy et Stanza, mais les divergences locales sont importantes pour l'interprétation des marqueurs.

Exemples de divergences ou de formes problématiques observées :

| Cas | Backend | Observation |
|---|---|---|
| `colère` | SpaCy | le lemme observé peut devenir `colèr`, forme tronquée ou non standard |
| `fierté` | SpaCy | le lemme observé peut devenir `fierter`, forme non interprétable comme lemme nominal |
| `honte` | SpaCy | le lemme observé peut devenir `hont` |
| `lui`, `elle` | SpaCy | certains pronoms peuvent être ramenés à `luire` ou `lui` selon le contexte |
| `se` | Stanza | peut devenir `soi`, qui n'est pas filtré par la liste actuelle de stopwords |
| `colère`, `honte`, `fierté` | Stanza | généralement plus lisible que SpaCy sur ces exemples |

Ces observations ne signifient pas que Stanza est toujours supérieur. Elles indiquent que l'analyse doit distinguer deux dimensions : stabilité statistique globale et qualité interprétative locale.

## Diagnostic lexical fuzzy

Une expérience exploratoire a été conduite avec `tools/match_marker_values_to_lexicon.py` sur les lemmes SpaCy et Stanza combinés.

Sorties produites :

- `results/pipeline_2_compare_lexicon_matching/pipeline_2_lemma_global_counts_spacy_stanza.csv`
- `results/pipeline_2_compare_lexicon_matching/global_marker_value_counts_with_lexicon_repair.csv`
- `results/pipeline_2_compare_lexicon_matching/global_marker_values_not_in_lexique_repair_candidates.csv`
- `results/pipeline_2_compare_lexicon_matching/global_marker_values_not_in_lexique_high_confidence_candidates.csv`
- `results/pipeline_2_compare_lexicon_matching/lexicon_repair_summary.csv`

Résultat synthétique :

| Statut | Valeurs uniques | Occurrences |
|---|---:|---:|
| Match exact avec le lexique | 335 | 3 522 |
| Réparation haute confiance | 162 | 685 |
| Sans match | 2 592 | 13 702 |
| Total | 3 089 | 17 909 |

Le fuzzy matching récupère donc une part réelle de marqueurs émotionnels ou para-émotionnels, mais cette part reste minoritaire. Il ne doit pas être utilisé comme mécanisme de correction global sans audit.

Exemples de réparations plausibles :

| Forme observée | Candidat | Commentaire |
|---|---|---|
| `colèr` | `colère` | correction plausible d'un artefact SpaCy |
| `dangereu` | `dangereux` | correction plausible d'une forme tronquée |
| `fierter` | `fierté` | correction plausible d'un artefact SpaCy |
| `inquiète` | `inquiéter` ou `inquiet` | correction plausible, mais ambiguë selon l'objectif |
| `amusé` | `amuser` | correction plausible si l'on veut canonicaliser vers l'infinitif |
| `effrayé` | `effrayer` | correction plausible si l'on veut canonicaliser vers l'infinitif |

Exemples de réparations discutables ou dangereuses :

| Forme observée | Candidat | Risque |
|---|---|---|
| `pleurer` | `pleureur` | transformation sémantiquement discutable |
| `mauvais` | `mauvaiseté` | candidat lexical rare, risque de surcorrection |
| `attendre` | `attendri` | proximité orthographique mais relation sémantique non fiable |
| `important` | `importun` | faux positif plausible par distance |
| `heure` | `heureux` | faux positif par préfixe |
| `marre` | `marrer` | peut inverser ou déplacer la valeur émotionnelle |

Conclusion : Levenshtein et fuzzy matching doivent produire des candidats, pas décider seuls de la forme canonique utilisée dans l'analyse de spécificité.

## Définir l'enrichissement morphologique

Un cas devrait être considéré comme un enrichissement morphologique utile si plusieurs conditions convergent.

Critères forts :

1. Le lemme regroupe plusieurs formes de surface réellement observées dans le corpus.
2. Les formes regroupées appartiennent à la même famille morphologique ou lexicale identifiable.
3. Le regroupement fait passer un marqueur au seuil de fréquence ou stabilise ses fréquences sans changer artificiellement son sens.
4. Le lemme est interprétable par un lecteur humain.
5. Le lemme est confirmé par au moins un des éléments suivants :
   - match exact dans un lexique contrôlé ;
   - accord entre SpaCy et Stanza ;
   - correction haute confiance validée ;
   - appartenance claire à une famille morphologique visible dans les formes de surface.
6. L'entropie obtenue après regroupement reste cohérente avec la distribution des formes de surface.

Exemples attendus d'enrichissement utile :

- regrouper `inquiète`, `inquiéter`, `inquiétant`, `inquiétude` si la méthode choisie accepte un regroupement lexical large ;
- regrouper `amusé`, `amusante`, `amusent`, `amuser` si la méthode vise une famille verbale ou adjectivale ;
- regrouper `dangereux`, `dangereuse`, `danger` seulement si la méthode accepte un regroupement dérivationnel plus large, ce qui doit être explicitement documenté.

Il faut distinguer deux niveaux :

| Niveau | Exemple | Interprétation |
|---|---|---|
| Flexion stricte | `dangereux` / `dangereuse` | même adjectif, variation de genre ou nombre |
| Dérivation ou famille lexicale | `danger` / `dangereux` | même famille, mais changement de catégorie grammaticale |

Ces deux niveaux ne doivent pas nécessairement être fusionnés dans la même analyse.

## Définir l'artefact de lemmatisation

Un cas doit être suspecté comme artefact si l'un ou plusieurs des signaux suivants sont présents :

1. Le lemme n'est pas une forme française interprétable : `colèr`, `hont`, `fierter`.
2. Le lemme est exclusif à un backend et très fréquent.
3. Le lemme regroupe des formes de surface sémantiquement hétérogènes.
4. Le fuzzy matching propose une correction fondée seulement sur une proximité orthographique courte.
5. La correction modifie la catégorie lexicale sans justification explicite.
6. Le candidat appartient au lexique émotionnel mais le marqueur observé n'est pas émotionnel dans le contexte.
7. La correction augmente ou diminue fortement l'entropie d'un marqueur sans explication lexicale claire.

Ces signaux ne doivent pas nécessairement entraîner une suppression automatique. Ils doivent au minimum produire une annotation d'audit.

## Limite actuelle de la pipeline

La pipeline actuelle ne conserve pas suffisamment d'information pour analyser rigoureusement les variantes flexionnelles.

Actuellement :

- les mots de surface sont extraits via regex ;
- les lemmes sont extraits séparément via SpaCy ou Stanza ;
- les deux listes sont ajoutées dans `markers.csv` comme des marqueurs distincts ;
- aucune relation token par token n'est conservée entre forme de surface et lemme ;
- aucun identifiant de token, offset ou position dans `text_span` n'est enregistré ;
- les différences de tokenisation entre regex, SpaCy et Stanza ne sont pas documentées dans les sorties.

Pour analyser les variantes, il faut idéalement conserver une table token-level avant explosion émotionnelle.

## Table token-level recommandée

Une future implémentation devrait produire une table intermédiaire, par exemple `token_lemmas.csv`, avant ou en parallèle de `markers.csv`.

Colonnes possibles :

| Colonne | Rôle |
|---|---|
| `segment_id` | identifiant stable de la ligne source |
| `source_file` | document source |
| `text_span` | texte original |
| `mode` | mode normalisé |
| `emotion` | émotion normalisée, après explosion si nécessaire |
| `token_index` | position du token dans le segment |
| `char_start` | offset de début si disponible |
| `char_end` | offset de fin si disponible |
| `surface` | forme de surface originale |
| `surface_norm` | forme normalisée, par exemple casefold |
| `backend` | `spacy`, `stanza`, `regex`, ou autre |
| `lemma` | lemme produit par le backend |
| `lemma_norm` | lemme normalisé |
| `pos` | partie du discours si disponible |
| `is_alpha` | indicateur alphabétique |
| `surface_is_stopword` | stopword avant lemmatisation |
| `lemma_is_stopword` | stopword après lemmatisation |
| `kept_for_analysis` | indique si le token est retenu |
| `drop_reason` | stopword, ponctuation, non alpha, vide, artefact, etc. |

Cette table permettrait de calculer explicitement :

- quelles formes de surface sont regroupées par chaque lemme ;
- quel backend produit quel lemme ;
- quels lemmes réintroduisent des stopwords ;
- quels regroupements changent les fréquences et les entropies ;
- quels cas nécessitent une correction ou une validation.

## Analyse des variantes à produire

Une sortie centrale devrait être un fichier du type `lemma_variant_summary.csv`.

Colonnes recommandées :

| Colonne | Description |
|---|---|
| `backend` | SpaCy, Stanza ou autre |
| `lemma` | lemme analysé |
| `surface_forms` | liste des formes regroupées |
| `n_surface_forms` | nombre de formes de surface distinctes |
| `total_count` | fréquence totale du lemme |
| `max_surface_count` | fréquence de la forme dominante |
| `dominant_surface` | forme de surface la plus fréquente |
| `dominant_surface_share` | part de la forme dominante |
| `count_gain_vs_surface_max` | gain de fréquence par regroupement |
| `crosses_min_freq_threshold` | indique si le lemme passe le seuil grâce au regroupement |
| `emotion_entropy_lemma` | entropie du lemme |
| `emotion_entropy_surface_weighted` | entropie pondérée des formes de surface |
| `entropy_delta` | différence entre regroupement et formes de surface |
| `modes_present` | modes où le lemme apparaît |
| `emotions_present` | émotions où le lemme apparaît |
| `lexicon_status` | exact, repaired, no_match, etc. |
| `artifact_flags` | indicateurs de suspicion |

Exemples d'indicateurs utiles :

- `single_surface_only` : le lemme ne regroupe aucune variante. Dans ce cas, il n'enrichit pas morphologiquement l'analyse.
- `backend_only_high_count` : le lemme est fréquent mais n'apparaît que dans un backend.
- `nonlexical_form` : le lemme semble tronqué ou non lexical.
- `stopword_leakage` : le lemme est un mot-outil ou assimilé.
- `large_entropy_delta` : le regroupement modifie fortement la dispersion émotionnelle.
- `ambiguous_fuzzy_candidate` : plusieurs candidats proches existent.

## Comparaisons de sensibilité à prévoir

Pour distinguer enrichissement et bruit, il ne suffit pas de produire un seul résultat canonique. Il faut comparer plusieurs vues.

Vues pertinentes :

| Vue | Description |
|---|---|
| `surface_only` | mots de surface et ponctuations, sans lemmes |
| `lemma_spacy_only` | lemmes SpaCy uniquement, hors mots de surface |
| `lemma_stanza_only` | lemmes Stanza uniquement, hors mots de surface |
| `mixed_current_spacy` | comportement actuel avec mots, ponctuation et lemmes SpaCy |
| `mixed_current_stanza` | comportement actuel avec mots, ponctuation et lemmes Stanza |
| `canonical_exact_lexicon` | formes ramenées seulement aux matchs exacts du lexique |
| `canonical_reviewed_repairs` | formes ramenées aux corrections validées |
| `canonical_no_fuzzy` | canonicalisation sans fuzzy matching |
| `canonical_fuzzy_annotated_only` | fuzzy enregistré, mais non appliqué aux fréquences |

Mesures de comparaison :

- nombre de marqueurs au seuil `min_freq`;
- nombre de marqueurs émotionnels interprétables ;
- entropie moyenne et médiane ;
- entropie par mode ;
- rang des top marqueurs spécifiques ;
- recouvrement des top-k marqueurs entre vues ;
- corrélation de rang des entropies ;
- variations des tests statistiques ;
- nombre de cas gagnés par regroupement morphologique ;
- nombre de cas suspectés comme artefacts.

Les tests de significativité doivent être complétés par des mesures d'effet et de stabilité. Les distributions partagent des observations ou des marqueurs liés ; les p-values seules ne suffisent pas.

## Politique de fuzzy matching

Le script `tools/match_marker_values_to_lexicon.py` est une bonne base pour générer des candidats de réparation. Il utilise notamment :

- normalisation textuelle ;
- suppression optionnelle des accents ;
- distance de Levenshtein bornée ;
- ratio de similarité ;
- contrainte de préfixe ;
- classement par confiance.

Cependant, la méthode doit éviter de transformer ce mécanisme en correction automatique globale.

Options d'intégration possibles :

1. **Annotation uniquement**
   Le fuzzy matching produit des colonnes de diagnostic mais ne change jamais `marker_value`.

2. **Correction haute confiance avec validation**
   Les candidats `high` sont proposés dans un fichier de validation. Seuls les mappings validés sont appliqués.

3. **Correction automatique très restreinte**
   Correction automatique seulement si :
   - distance <= 1 ou transformation clairement accentuelle ;
   - préfixe long partagé ;
   - un seul candidat existe ;
   - le candidat est dans le lexique ;
   - la forme observée est non lexicale ou clairement tronquée ;
   - aucun faux positif évident n'est détecté.

4. **Correction contrôlée par dictionnaire**
   Une table `manual_canonical_mapping.csv` ou `reviewed_lexicon_repairs.csv` définit explicitement les substitutions autorisées.

5. **Correction exploratoire non utilisée dans la spécificité principale**
   Les corrections servent à documenter les artefacts et à produire une analyse secondaire, mais pas le résultat principal.

La recommandation prudente est de commencer par les options 1 et 2.

## Colonnes de canonicalisation recommandées

Il est préférable de ne pas écraser `marker_value`. Une future implémentation devrait ajouter des colonnes séparées :

| Colonne | Rôle |
|---|---|
| `marker_value` | valeur originale conservée |
| `marker_type` | `word`, `lemma`, `punctuation`, etc. |
| `canonical_marker_value` | valeur utilisée pour une vue canonique |
| `canonical_source` | `surface`, `lemma_spacy`, `lemma_stanza`, `lexicon_exact`, `reviewed_repair`, etc. |
| `canonical_confidence` | `exact`, `high_reviewed`, `manual`, `none`, etc. |
| `canonical_action` | keep, replace, drop, flag |
| `canonical_note` | justification courte ou identifiant de règle |

Cette séparation permet de calculer la spécificité sur différentes colonnes sans perdre la traçabilité.

## Stopwords et lemmatisation

Le filtrage des stopwords doit être appliqué à deux niveaux :

1. sur la forme de surface ;
2. sur le lemme produit.

Le cas `soi` illustre le risque. `se` est dans la liste locale de stopwords, mais Stanza peut produire `soi`, qui n'est pas filtré dans l'état actuel. Une implémentation robuste devrait :

- enrichir la liste de stopwords avec des lemmes pronominaux et formes reconstruites ;
- produire un rapport `stopword_leakage.csv`;
- distinguer les stopwords supprimés avant et après lemmatisation ;
- vérifier les lemmes très fréquents à forte entropie et faible valeur interprétative.

## Risques méthodologiques

### Double comptage

Le comportement actuel ajoute les lemmes aux mots de surface. Un même segment peut donc contribuer deux fois à l'analyse sous deux formes différentes. Ce n'est pas forcément incorrect si l'objectif est d'analyser plusieurs types de marqueurs, mais cela ne doit pas être présenté comme une analyse lemmatisée pure.

### Circularité lexicale

Le lexique émotionnel peut aider à annoter les marqueurs, mais il ne doit pas choisir une correction en fonction de l'émotion observée dans le corpus. Utiliser l'étiquette émotionnelle comme critère de correction introduirait un biais circulaire dans l'analyse de spécificité.

Il est acceptable d'utiliser les émotions pour auditer a posteriori les corrections, mais pas pour sélectionner automatiquement la correction.

### Faux positifs fuzzy

La proximité orthographique ne garantit pas une relation morphologique. Les exemples `attendre` -> `attendri`, `important` -> `importun` et `heure` -> `heureux` montrent que des candidats haute confiance peuvent être méthodologiquement inacceptables.

### Fusion de flexion et dérivation

La méthode doit décider si elle regroupe seulement les variantes flexionnelles strictes ou aussi les familles dérivationnelles.

Ces deux options répondent à des objectifs différents :

- flexion stricte : plus conservatrice, moins de bruit ;
- famille lexicale large : plus informative, mais plus de décisions interprétatives.

Il peut être pertinent de produire les deux vues.

### Dépendance au backend

SpaCy et Stanza ne produisent pas toujours les mêmes lemmes. Il faut mesurer :

- accords token par token ;
- désaccords fréquents ;
- lemmes exclusifs à un backend ;
- formes non lexicales par backend ;
- impact sur les marqueurs qui passent le seuil `min_freq`.

## Sorties recommandées

Une implémentation complète pourrait produire les fichiers suivants :

| Fichier | Objectif |
|---|---|
| `token_lemmas.csv` | table token-level avec surface, lemme, backend, stopwords et offsets |
| `lemma_variant_summary.csv` | résumé des variantes regroupées par lemme |
| `lemmatizer_backend_comparison.csv` | comparaison SpaCy/Stanza |
| `lemmatizer_artifact_candidates.csv` | lemmes suspects ou non lexicaux |
| `stopword_leakage.csv` | stopwords réintroduits après lemmatisation |
| `lexicon_exact_matches.csv` | matchs exacts au lexique émotionnel |
| `lexicon_fuzzy_candidates.csv` | candidats fuzzy non appliqués |
| `reviewed_canonical_mapping.csv` | mappings validés pour correction |
| `specificity_sensitivity_summary.csv` | comparaison des vues de spécificité |
| `specificity_delta_by_marker.csv` | changements d'entropie par marqueur |
| `analyse_des_variantes_report.md` | rapport final automatisé |

## Paramètres d'implémentation à explorer

Paramètres possibles :

| Paramètre | Rôle |
|---|---|
| `--marker-view` | `surface`, `lemma`, `mixed`, `canonical` |
| `--lemmatizer` | `spacy`, `stanza`, `both`, `none` |
| `--lemma-backend-comparison` | active une comparaison détaillée des backends |
| `--canonical-map` | chemin vers une table de mappings validés |
| `--fuzzy-repair` | `off`, `annotate`, `high-only`, `reviewed-only` |
| `--lexicon` | chemin vers le lexique émotionnel |
| `--strict-flexion-only` | interdit les regroupements dérivationnels |
| `--include-derivational-family` | autorise des familles lexicales plus larges |
| `--min-freq` | seuil de fréquence pour la spécificité |
| `--artifact-report` | produit un rapport des lemmes suspects |

Ces paramètres ne doivent pas tous être implémentés immédiatement. Ils définissent un espace de conception pertinent.

## Méthode minimale recommandée

Pour une première implémentation robuste, la méthode minimale pourrait être :

1. Ajouter une sortie token-level conservant surface, lemme, backend et stopword status.
2. Produire `lemma_variant_summary.csv`.
3. Ajouter une option de spécificité `marker_type == lemma` ou `marker_view == lemma`.
4. Produire une comparaison `surface_only` versus `lemma_only`.
5. Annoter les lemmes par match exact avec le lexique émotionnel.
6. Générer des candidats fuzzy, sans les appliquer automatiquement.
7. Produire un rapport des artefacts probables.
8. Laisser la correction canonique à une étape validée ou à un mapping explicite.

Cette version permettrait déjà de répondre à la question centrale : quels lemmes enrichissent réellement l'analyse, et lesquels introduisent du bruit.

## Méthode avancée possible

Une version plus ambitieuse pourrait ajouter :

- comparaison token-level SpaCy/Stanza sur les mêmes segments ;
- score de confiance morphologique ;
- distinction flexion stricte versus famille lexicale ;
- mapping canonique validé ;
- analyses de sensibilité automatiques sur plusieurs vues ;
- rapport des marqueurs dont la spécificité change fortement après canonicalisation ;
- audit manuel assisté des candidats fuzzy ;
- génération de tables prêtes pour visualisation.

Cette version serait plus coûteuse, mais plus solide pour une publication ou une analyse méthodologique détaillée.

## Critères de décision pour retenir une correction

Une correction devrait être retenue pour l'analyse principale seulement si elle satisfait un niveau de preuve suffisant.

Critères possibles :

| Niveau | Condition | Utilisation recommandée |
|---|---|---|
| Exact | forme observée dans le lexique | utilisable en annotation |
| Backend agreement | SpaCy et Stanza produisent le même lemme | utilisable comme signal de confiance |
| Reviewed high | candidat fuzzy haute confiance validé | utilisable dans une vue canonique |
| Fuzzy high non validé | candidat plausible mais non relu | annotation seulement |
| Fuzzy medium | candidat incertain | audit seulement |
| Backend-only suspect | lemme fréquent exclusif à un backend | audit ou exclusion selon cas |
| Stopword leakage | lemme mot-outil réintroduit | exclusion probable |
| Nonlexical | forme tronquée ou non lexicale | correction ou exclusion après validation |

## Formulation opérationnelle pour l'implémentation

L'implémentation devrait permettre de répondre automatiquement aux questions suivantes :

1. Quels lemmes ajoutés par SpaCy ou Stanza passent le seuil de fréquence ?
2. Parmi eux, lesquels regroupent réellement plusieurs formes de surface ?
3. Quels lemmes n'apportent aucun regroupement morphologique ?
4. Quels lemmes sont propres à un seul backend ?
5. Quels lemmes correspondent exactement au lexique émotionnel ?
6. Quels lemmes peuvent être réparés par fuzzy matching, et avec quelle confiance ?
7. Quels candidats fuzzy sont probablement faux positifs ?
8. Quels lemmes réintroduisent des stopwords ?
9. Quels marqueurs changent fortement d'entropie après regroupement ?
10. Les conclusions par mode changent-elles entre surface, lemme et représentation canonique ?

Le résultat attendu n'est pas seulement un `markers.csv` corrigé. Il faut aussi produire les preuves permettant de juger la qualité de cette correction.

## Conclusion

L'intégration d'une analyse des variantes flexionnelles est méthodologiquement pertinente. Elle permettrait de mesurer précisément la contribution des lemmatizers à l'analyse de spécificité : augmentation de couverture, regroupement de variantes, stabilisation de fréquences, mais aussi introduction de formes bruitées.

La correction par fuzzy matching est également pertinente, mais seulement comme outil d'annotation et de génération de candidats dans un premier temps. Les résultats observés montrent qu'elle récupère des formes utiles, mais qu'elle produit aussi des faux positifs suffisamment plausibles pour biaiser l'analyse si elle est appliquée automatiquement.

La méthode recommandée est donc comparative et traçable :

- conserver les formes originales ;
- enregistrer les lemmes avec leur backend ;
- documenter les variantes regroupées ;
- annoter les matchs lexicaux exacts ;
- proposer des corrections fuzzy sans les imposer ;
- appliquer seulement les corrections validées dans une vue canonique séparée ;
- comparer systématiquement les résultats de spécificité entre les vues.

Cette approche laisse la possibilité d'explorer plusieurs niveaux de normalisation tout en préservant la distinction essentielle entre enrichissement morphologique et artefact du lemmatizer.
