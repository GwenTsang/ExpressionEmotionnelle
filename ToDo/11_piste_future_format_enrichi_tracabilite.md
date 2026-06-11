# Piste future 11 : definir un format enrichi pour audit et tracabilite

- Objectif : evaluer l'interet d'un format Parquet plus riche que `SimpleSitEmo.parquet`.
- Exemple de nom : `annotations_enriched.parquet`.
- Statut : piste future, distincte du schema minimal.

Colonnes candidates :

```text
source_file
corpus
file_id
unit_id
type
start_idx
end_idx
segments
text_span
mode
emotion
emotion_rank
remarque
source_path
```

Usage vise :

- audits ;
- verifications de parsing ;
- realignements Glozz/XLSX ;
- analyses plus fines.

Ce format enrichi ne remplacerait pas necessairement le `SimpleSitEmo` minimal.
