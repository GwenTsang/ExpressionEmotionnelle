Dans la fonction `_extract_target_unit_record`, modifier le dictionnaire initial et l'assignation :
```python
    record = {
        # ... [vos clés existantes]
        "mode": None,
        "type1": None,
        "categorie1": None,
        "type2": None,
        "categorie2": None,
        "nature": None,
        "declencheur": None,
        "remarque": None,
    }

    if unit_type == "SitEmo":
        record["mode"] = features.get("Mode")
        record["type1"] = features.get("Type")
        record["categorie1"] = features.get("Categorie")
        record["type2"] = features.get("Type2")
        record["categorie2"] = features.get("Categorie2")
        record["nature"] = features.get("Nature")
        record["declencheur"] = features.get("Declencheur")
```
Dans la fonction `_merge_discontinuous_records`, n'oubliez pas de fusionner ces nouveaux champs :
```python
    if merged["type"] == "SitEmo":
        merged["mode"] = _merge_feature_values(ordered, "mode", unit_ids=unit_ids, file_id=file_id)
        merged["type1"] = _merge_feature_values(ordered, "type1", unit_ids=unit_ids, file_id=file_id)
        merged["categorie1"] = _merge_feature_values(ordered, "categorie1", unit_ids=unit_ids, file_id=file_id)
        merged["type2"] = _merge_feature_values(ordered, "type2", unit_ids=unit_ids, file_id=file_id)
        merged["categorie2"] = _merge_feature_values(ordered, "categorie2", unit_ids=unit_ids, file_id=file_id)
        merged["nature"] = _merge_feature_values(ordered, "nature", unit_ids=unit_ids, file_id=file_id)
        merged["declencheur"] = _merge_feature_values(ordered, "declencheur", unit_ids=unit_ids, file_id=file_id)
```
