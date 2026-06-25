# Tache 08 : produire une analyse exploratoire `MeprisHaine` ou bien séparément `mépris` et `haine`

- Objectif : analyser separement les cas hors schema correspondant probablement a `mépris / haine`.
- Portee : analyse exploratoire distincte des analyses principales.

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

Dans l'idéal, cela permet de créer un nouveau dataset avec une colonne `text` et une colonne binaire `MeprisHaine`.
On peut aussi faire une colonne `text` et une colonne 