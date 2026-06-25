#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glozz_parser.py — Parsing de corpus annotés au format Glozz (.aa/.ac)

Ce script parcourt les quatre sous-corpus, associe chaque fichier
d'annotation XML (.aa) à son fichier texte brut (.ac), et extrait
toutes les unités d'annotation de type SitEmo et Autre.

Colonnes produites :
    corpus, file_id, unit_id, type, start_idx, end_idx, text_span,
    source_unit_ids, segments, is_discontinuous, discontinuous_group_id,
    mode, categorie1, categorie2, nature, remarque
"""

import os
import sys
import argparse
import json
import re
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Chemins des quatre corpus (relatifs au dossier du script)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data", "raw", "glozz")
CORPUS_DIRS = {
    "Albert_dataset": os.path.join(_DATA_DIR, "Albert_dataset"),
    "CorpusCovid": os.path.join(_DATA_DIR, "CorpusCovid"),
    "LitteratureJeunesse": os.path.join(_DATA_DIR, "LitteratureJeunesse"),
    "PtitLibe": os.path.join(_DATA_DIR, "PtitLibe"),
}

# Types d'annotation à extraire
TARGET_TYPES = {"SitEmo", "Autre"}

# ---------------------------------------------------------------------------
# Fonctions de parsing
# ---------------------------------------------------------------------------


def _get_feature_value(feature_node) -> Optional[str]:
    """Extrait la valeur d'un nœud <feature>.

    Le format Glozz encode les valeurs comme texte direct :
        <feature name="Mode">Suggeree</feature>
    ou vide :
        <feature name="Mode"/>

    Retourne None si le nœud est vide ou le texte est absent.
    """
    if feature_node is None:
        return None
    text = feature_node.text
    if text is None or text.strip() == "":
        return None
    return text.strip()


def _clean_span(text: str) -> str:
    """Normalise les retours ligne sans modifier le contenu lexical."""
    return text.replace("\n", " ").replace("\r", " ")


def _normalize_values(text: Optional[str]) -> Optional[str]:
    """Normalise, corrige les typos et dé-duplique les valeurs multiples."""
    if not text:
        return None
    # Splitter sur /, ; et ,
    parts = re.split(r'[/;,]', text)
    normalized = []
    for p in parts:
        p = p.strip().lower()
        if not p:
            continue
        # Corrections
        if p == "emour":
            p = "amour"
        elif p == "stresse":
            p = "stress"
        elif p == "amour (apprécier)":
            p = "amour"
        elif p == "déterminsation":
            p = "détermination"
        
        # Ajout aux valeurs uniques
        if p not in normalized:
            normalized.append(p)
            
    if not normalized:
        return None
    return ", ".join(normalized)


def _merge_feature_values(
    records: list[dict],
    column: str,
    *,
    unit_ids: list[str],
    file_id: str,
) -> Optional[str]:
    values = [
        record.get(column)
        for record in records
        if record.get(column) is not None and str(record.get(column)).strip() != ""
    ]
    unique_values = list(dict.fromkeys(values))
    if len(unique_values) > 1:
        print(
            "Conflit de valeurs '%s' dans le discontinu %s (%s) : %s"
            % (column, "+".join(unit_ids), file_id, unique_values),
            file=sys.stderr,
        )
    return unique_values[0] if unique_values else None


def _extract_target_unit_record(
    unit,
    *,
    raw_text: str,
    file_id: str,
    corpus_name: str,
) -> Optional[dict]:
    """Extrait une unité cible Glozz continue sous forme d'enregistrement."""
    type_node = unit.find("./characterisation/type")
    if type_node is None:
        return None
    unit_type = (type_node.text or "").strip()
    if unit_type not in TARGET_TYPES:
        return None

    unit_id = unit.get("id", "")

    start_node = unit.find("./positioning/start/singlePosition")
    end_node = unit.find("./positioning/end/singlePosition")
    if start_node is None or end_node is None:
        print("Unité %s sans position dans %s — ignorée" % (unit_id, file_id))
        return None

    try:
        start_idx = int(start_node.get("index", "-1"))
        end_idx = int(end_node.get("index", "-1"))
    except (ValueError, TypeError):
        print("Index invalide pour l'unité %s dans %s" % (unit_id, file_id))
        return None

    if 0 <= start_idx <= end_idx <= len(raw_text):
        text_span = raw_text[start_idx:end_idx]
    else:
        print(
            "Offsets hors limites [%d:%d] pour l'unité %s dans %s "
            "(longueur texte=%d)"
            % (start_idx, end_idx, unit_id, file_id, len(raw_text))
        )
        text_span = ""

    features = {}
    feature_set = unit.find("./characterisation/featureSet")
    if feature_set is not None:
        for feature in feature_set.findall("feature"):
            feat_name = feature.get("name", "")
            feat_value = _get_feature_value(feature)
            features[feat_name] = feat_value

    record = {
        "corpus": corpus_name,
        "file_id": file_id,
        "unit_id": unit_id,
        "type": unit_type,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "text_span": _clean_span(text_span),
        "source_unit_ids": json.dumps([unit_id], ensure_ascii=False),
        "segments": json.dumps([[start_idx, end_idx]], ensure_ascii=False),
        "is_discontinuous": False,
        "discontinuous_group_id": None,
        "mode": None,
        "categorie1": None,
        "categorie2": None,
        "nature": None,
        "declencheur": None,
        "remarque": None,
    }

    if unit_type == "SitEmo":
        record["mode"] = features.get("Mode")
        record["categorie1"] = features.get("Categorie")
        record["categorie2"] = features.get("Categorie2")
        record["nature"] = features.get("Nature")
        record["declencheur"] = features.get("Declencheur")
    elif unit_type == "Autre":
        record["remarque"] = _normalize_values(features.get("Remarque"))

    return record


def _discontinue_components(root, unit_records: dict[str, dict], file_id: str):
    """Construit les composantes d'unités reliées par des relations Discontinue."""
    graph: dict[str, set[str]] = {}

    for relation in root.findall(".//relation"):
        type_node = relation.find("./characterisation/type")
        relation_type = (type_node.text or "").strip() if type_node is not None else ""
        if relation_type != "Discontinue":
            continue

        terms = [
            term.get("id")
            for term in relation.findall("./positioning/term")
            if term.get("id")
        ]
        target_terms = [term_id for term_id in terms if term_id in unit_records]
        if len(target_terms) < 2:
            continue

        term_types = {unit_records[term_id]["type"] for term_id in target_terms}
        if len(term_types) > 1:
            print(
                "Relation Discontinue mixte ignorée dans %s : %s"
                % (file_id, target_terms),
                file=sys.stderr,
            )
            continue

        for term_id in target_terms:
            graph.setdefault(term_id, set())
        for left, right in zip(target_terms, target_terms[1:]):
            graph[left].add(right)
            graph[right].add(left)

    components = []
    seen = set()
    for unit_id in sorted(graph):
        if unit_id in seen:
            continue
        stack = [unit_id]
        seen.add(unit_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(component) > 1:
            components.append(component)

    return components


def _merge_discontinuous_records(records: list[dict], *, file_id: str) -> dict:
    """Fusionne les segments d'une annotation discontinue en une seule ligne."""
    ordered = sorted(records, key=lambda record: (record["start_idx"], record["end_idx"]))
    unit_ids = [record["unit_id"] for record in ordered]
    segments = [[record["start_idx"], record["end_idx"]] for record in ordered]
    group_id = "discontinuous:" + "+".join(unit_ids)
    text_span = " ".join(
        record["text_span"].strip()
        for record in ordered
        if record["text_span"].strip() != ""
    )

    merged = {
        "corpus": ordered[0]["corpus"],
        "file_id": ordered[0]["file_id"],
        "unit_id": group_id,
        "type": ordered[0]["type"],
        "start_idx": min(record["start_idx"] for record in ordered),
        "end_idx": max(record["end_idx"] for record in ordered),
        "text_span": text_span,
        "source_unit_ids": json.dumps(unit_ids, ensure_ascii=False),
        "segments": json.dumps(segments, ensure_ascii=False),
        "is_discontinuous": True,
        "discontinuous_group_id": group_id,
        "mode": None,
        "categorie1": None,
        "categorie2": None,
        "nature": None,
        "declencheur": None,
        "remarque": None,
    }

    if merged["type"] == "SitEmo":
        merged["mode"] = _merge_feature_values(ordered, "mode", unit_ids=unit_ids, file_id=file_id)
        merged["categorie1"] = _merge_feature_values(ordered, "categorie1", unit_ids=unit_ids, file_id=file_id)
        merged["categorie2"] = _merge_feature_values(ordered, "categorie2", unit_ids=unit_ids, file_id=file_id)
        merged["nature"] = _merge_feature_values(ordered, "nature", unit_ids=unit_ids, file_id=file_id)
        merged["declencheur"] = _merge_feature_values(ordered, "declencheur", unit_ids=unit_ids, file_id=file_id)
    elif merged["type"] == "Autre":
        merged["remarque"] = _merge_feature_values(
            ordered, "remarque", unit_ids=unit_ids, file_id=file_id
        )

    return merged


def parse_aa_ac_pair(
    aa_filepath: str,
    ac_filepath: str,
    corpus_name: str = "",
) -> list[dict]:
    """Parse une paire de fichiers .aa (annotations) et .ac (texte brut).

    Extrait toutes les unités de type SitEmo et Autre avec leurs
    métadonnées et le segment textuel correspondant.

    Parameters
    ----------
    aa_filepath : str
        Chemin vers le fichier d'annotation XML (.aa).
    ac_filepath : str
        Chemin vers le fichier texte brut correspondant (.ac).
    corpus_name : str
        Nom du corpus source (pour la colonne 'corpus').

    Returns
    -------
    list[dict]
        Liste de dictionnaires, un par unité extraite.
    """
    # --- Lecture du texte brut ---
    try:
        with open(ac_filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print("Fichier .ac manquant : %s" % ac_filepath)
        return []
    except Exception as e:
        print("Erreur lecture .ac %s : %s" % (ac_filepath, e))
        return []

    # --- Parsing XML ---
    try:
        tree = ET.parse(aa_filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print("Erreur XML dans %s : %s" % (aa_filepath, e))
        return []
    except Exception as e:
        print("Erreur lecture .aa %s : %s" % (aa_filepath, e))
        return []

    file_id = os.path.basename(aa_filepath).replace(".aa", "")
    unit_records = {}

    for unit in root.findall(".//unit"):
        record = _extract_target_unit_record(
            unit,
            raw_text=raw_text,
            file_id=file_id,
            corpus_name=corpus_name,
        )
        if record is not None:
            unit_records[record["unit_id"]] = record

    discontinuous_components = _discontinue_components(root, unit_records, file_id)
    discontinuous_unit_ids = {
        unit_id
        for component in discontinuous_components
        for unit_id in component
    }

    extracted = [
        record
        for unit_id, record in unit_records.items()
        if unit_id not in discontinuous_unit_ids
    ]
    for component in discontinuous_components:
        records = [unit_records[unit_id] for unit_id in component]
        extracted.append(_merge_discontinuous_records(records, file_id=file_id))

    return extracted


def process_corpus(corpus_dir: str, corpus_name: str) -> pd.DataFrame:
    """Parcourt un corpus (dossiers aa/ et ac/) et extrait les annotations.

    Parameters
    ----------
    corpus_dir : str
        Chemin vers le dossier racine du corpus (contenant aa/ et ac/).
    corpus_name : str
        Nom du corpus pour la colonne 'corpus'.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant toutes les annotations extraites.
    """
    aa_dir = os.path.join(corpus_dir, "aa")
    ac_dir = os.path.join(corpus_dir, "ac")

    if not os.path.isdir(aa_dir):
        print("Dossier aa/ introuvable : %s" % aa_dir)
        return pd.DataFrame()
    if not os.path.isdir(ac_dir):
        print("Dossier ac/ introuvable : %s" % ac_dir)
        return pd.DataFrame()

    all_records = []
    aa_files = sorted(
        f for f in os.listdir(aa_dir) if f.endswith(".aa")
    )
    n_files = len(aa_files)
    n_missing = 0
    n_errors = 0

    for aa_filename in aa_files:
        aa_path = os.path.join(aa_dir, aa_filename)
        ac_filename = aa_filename.replace(".aa", ".ac")
        ac_path = os.path.join(ac_dir, ac_filename)

        if not os.path.isfile(ac_path):
            print("Fichier .ac manquant pour %s" % aa_filename)
            n_missing += 1
            continue

        records = parse_aa_ac_pair(aa_path, ac_path, corpus_name)
        if records is None:
            n_errors += 1
            continue
        all_records.extend(records)

    print(
        "Corpus '%s' : %d annotations extraites (%d fichiers manquants, "
        "%d erreurs)"
        % (
            corpus_name,
            len(all_records),
            n_missing,
            n_errors,
        )
    )

    return pd.DataFrame(all_records)


def process_all_corpora(
    corpus_dirs: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Agrège les annotations de tous les corpus.

    Parameters
    ----------
    corpus_dirs : dict[str, str], optional
        Dictionnaire {nom_corpus: chemin}. Par défaut : CORPUS_DIRS.

    Returns
    -------
    pd.DataFrame
        DataFrame consolidé de toutes les annotations.
    """
    if corpus_dirs is None:
        corpus_dirs = CORPUS_DIRS

    all_dfs = []
    for corpus_name, corpus_dir in corpus_dirs.items():
        if not os.path.isdir(corpus_dir):
            print("Dossier corpus introuvable : %s (%s)" % (corpus_dir, corpus_name))
            continue
        df = process_corpus(corpus_dir, corpus_name)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("Aucune annotation extraite de tous les corpus.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    print(
        "Total : %d annotations extraites de %d corpus"
        % (len(result), len(all_dfs))
    )
    return result


def export_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Exporte le DataFrame dans un fichier CSV encodé en UTF-8.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à exporter.
    output_path : str
        Chemin du fichier de sortie.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Résultats exportés : %s (%d lignes)" % (output_path, len(df)))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse les corpus Glozz et extrait les annotations émotionnelles."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "glozz", "annotations.csv"),
        help="Chemin du fichier CSV de sortie (défaut: ../results/glozz/annotations.csv)",
    )
    args = parser.parse_args()

    print("=== Démarrage du parsing des corpus Glozz ===")
    df = process_all_corpora()

    if df.empty:
        print("df empty, Arrêt.")
        sys.exit(1)

    # Résumé statistique rapide
    print("--- Résumé ---")
    print("Lignes totales      : %d" % len(df))
    print("Par corpus          : %s" % df["corpus"].value_counts().to_dict())
    print("Par type            : %s" % df["type"].value_counts().to_dict())
    if "mode" in df.columns:
        print(
            "Par mode (SitEmo)   : %s"
            % df.loc[df["type"] == "SitEmo", "mode"].value_counts(dropna=False).to_dict()
        )
    if "categorie1" in df.columns:
        print(
            "Par catégorie1      : %s"
            % df.loc[df["type"] == "SitEmo", "categorie1"]
            .value_counts(dropna=False)
            .to_dict(),
        )

    export_to_csv(df, args.output)

print()

if __name__ == "__main__":
    main()
