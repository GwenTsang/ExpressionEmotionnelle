# Comparaison des 19 colonnes binaires

## Fichiers

- Reference : `../emotexttokids_gold_flat.xlsx`
- Genere : `data/raw/xlsx/TTK-Glozz_19_FlatColumns.parquet`

## Alignement TEXT

- `reference_rows`: 27911
- `generated_rows`: 29150
- `reference_unique_TEXT`: 27493
- `generated_unique_TEXT`: 28664
- `exact_TEXT_matched_occurrences`: 26186
- `exact_TEXT_reference_missing_occurrences`: 1725
- `exact_TEXT_generated_extra_occurrences`: 2964
- `space_normalized_TEXT_matched_occurrences`: 26186
- `space_normalized_TEXT_reference_missing_occurrences`: 1725
- `space_normalized_TEXT_generated_extra_occurrences`: 2964
- `reference_total_TEXT_chars`: 2900065
- `generated_total_TEXT_chars`: 2935052
- `generated_minus_reference_TEXT_chars`: 34987
- `generated_minus_reference_TEXT_chars_pct`: 1.206

## Totaux par colonne

| column | reference_positive | generated_positive | delta_generated_minus_reference | relative_delta_pct |
| --- | --- | --- | --- | --- |
| Colere | 1180 | 1218 | 38 | 3.22 |
| Degout | 52 | 55 | 3 | 5.769 |
| Joie | 888 | 919 | 31 | 3.491 |
| Peur | 1047 | 1073 | 26 | 2.483 |
| Surprise | 824 | 822 | -2 | -0.243 |
| Tristesse | 673 | 682 | 9 | 1.337 |
| Admiration | 211 | 210 | -1 | -0.474 |
| Culpabilite | 19 | 19 | 0 | 0.0 |
| Embarras | 162 | 164 | 2 | 1.235 |
| Fierte | 202 | 206 | 4 | 1.98 |
| Jalousie | 7 | 6 | -1 | -14.286 |
| Autre | 1270 | 1285 | 15 | 1.181 |
| Comportementale | 1242 | 1252 | 10 | 0.805 |
| Designee | 1499 | 1518 | 19 | 1.268 |
| Montree | 971 | 952 | -19 | -1.957 |
| Suggeree | 1909 | 1924 | 15 | 0.786 |
| Emo | 5374 | 5502 | 128 | 2.382 |
| Base | 4133 | 4231 | 98 | 2.371 |
| Complexe | 568 | 573 | 5 | 0.88 |

## Comparaison ligne a ligne sur TEXT exact

| match_scope | column | matched_rows | mismatched_rows | agreement_pct | reference_1_generated_0 | reference_0_generated_1 |
| --- | --- | --- | --- | --- | --- | --- |
| exact_TEXT_occurrence | Colere | 26186 | 2 | 99.992 | 1 | 1 |
| exact_TEXT_occurrence | Degout | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Joie | 26186 | 1 | 99.996 | 1 | 0 |
| exact_TEXT_occurrence | Peur | 26186 | 1 | 99.996 | 1 | 0 |
| exact_TEXT_occurrence | Surprise | 26186 | 3 | 99.989 | 1 | 2 |
| exact_TEXT_occurrence | Tristesse | 26186 | 2 | 99.992 | 2 | 0 |
| exact_TEXT_occurrence | Admiration | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Culpabilite | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Embarras | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Fierte | 26186 | 1 | 99.996 | 1 | 0 |
| exact_TEXT_occurrence | Jalousie | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Autre | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Comportementale | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Designee | 26186 | 0 | 100.0 | 0 | 0 |
| exact_TEXT_occurrence | Montree | 26186 | 2 | 99.992 | 1 | 1 |
| exact_TEXT_occurrence | Suggeree | 26186 | 2 | 99.992 | 2 | 0 |
| exact_TEXT_occurrence | Emo | 26186 | 4 | 99.985 | 3 | 1 |
| exact_TEXT_occurrence | Base | 26186 | 4 | 99.985 | 3 | 1 |
| exact_TEXT_occurrence | Complexe | 26186 | 1 | 99.996 | 1 | 0 |

## Comparaison ligne a ligne apres normalisation des espaces

| match_scope | column | matched_rows | mismatched_rows | agreement_pct | reference_1_generated_0 | reference_0_generated_1 |
| --- | --- | --- | --- | --- | --- | --- |
| space_normalized_TEXT_occurrence | Colere | 26186 | 2 | 99.992 | 1 | 1 |
| space_normalized_TEXT_occurrence | Degout | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Joie | 26186 | 1 | 99.996 | 1 | 0 |
| space_normalized_TEXT_occurrence | Peur | 26186 | 1 | 99.996 | 1 | 0 |
| space_normalized_TEXT_occurrence | Surprise | 26186 | 3 | 99.989 | 1 | 2 |
| space_normalized_TEXT_occurrence | Tristesse | 26186 | 2 | 99.992 | 2 | 0 |
| space_normalized_TEXT_occurrence | Admiration | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Culpabilite | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Embarras | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Fierte | 26186 | 1 | 99.996 | 1 | 0 |
| space_normalized_TEXT_occurrence | Jalousie | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Autre | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Comportementale | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Designee | 26186 | 0 | 100.0 | 0 | 0 |
| space_normalized_TEXT_occurrence | Montree | 26186 | 2 | 99.992 | 1 | 1 |
| space_normalized_TEXT_occurrence | Suggeree | 26186 | 2 | 99.992 | 2 | 0 |
| space_normalized_TEXT_occurrence | Emo | 26186 | 4 | 99.985 | 3 | 1 |
| space_normalized_TEXT_occurrence | Base | 26186 | 4 | 99.985 | 3 | 1 |
| space_normalized_TEXT_occurrence | Complexe | 26186 | 1 | 99.996 | 1 | 0 |
