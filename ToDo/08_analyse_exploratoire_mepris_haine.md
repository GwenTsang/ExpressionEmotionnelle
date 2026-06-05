# Tache 08 : produire une analyse exploratoire `MeprisHaine`

- Objectif : analyser separement les cas hors schema correspondant probablement a `mépris / haine`.
- Portee : analyse exploratoire distincte des analyses principales.
- Attention : ne pas attribuer automatiquement `mépris / haine` a tous les spans `Autre`.

Audit XLSX deja execute :

```text
Autre=1 : 62 lignes
mépris / haine=1 : 344 lignes
Autre=1 ET mépris / haine=1 : 47 lignes
Autre=1 ET mépris / haine=0 : 15 lignes

Spans contenant Autre : 67
Spans Autre dans une ligne mépris / haine=1 : 51
Spans Autre dans une ligne mépris / haine=0 : 16
```

Regle proposee pour XLSX :

```text
si spanN_cat contient Autre ET mépris / haine == 1:
    out_of_schema_affect = "MeprisHaine"
    out_of_schema_affect_source = "xlsx_row_flag_inferred"
```

Comme `mépris / haine` est une colonne de ligne, cette attribution reste inferee et non strictement span-level.

Audit Glozz deja execute :

```text
Unites Glozz type Autre : 1537
Remarque = Haine/haine/Mépris/mépris : 147
```

Regle proposee pour Glozz :

```text
si type == "Autre" et remarque.lower() in {"haine", "mépris", "mepris"}:
    out_of_schema_affect = "MeprisHaine"
    out_of_schema_affect_source = "glozz_remarque_exact"
```

Tests attendus :

- les 51 spans XLSX `Autre` dans des lignes `mépris / haine=1` sont marques `MeprisHaine` avec provenance inferee ;
- les remarques Glozz `Haine`, `haine`, `Mépris`, `mépris` sont marquees `MeprisHaine` avec provenance exacte.
