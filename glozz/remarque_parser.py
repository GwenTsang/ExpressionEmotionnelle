#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remarque_parser.py — Parsing ciblé des éléments "Remarque" dans les fichiers Glozz

Ce module parcourt les sous-corpus Glozz, associe chaque fichier d'annotation (.aa)
à son fichier de texte brut (.ac), et extrait uniquement les unités d'annotation
de type "Autre" pour en extraire la feature "Remarque".

Contrairement à glozz_parser.process_all_corpora(), ce module ne parse que les
unités pertinentes (type "Autre") et ignore le traitement des relations Discontinue,
ce qui le rend significativement plus léger pour les cas d'usage centrés sur les
remarques.

Colonnes produites :
    corpus, file_id, unit_id, start_idx, end_idx, text_span, remarque
"""

import os
import re
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any
import pandas as pd

# Chemins par défaut
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data", "raw", "glozz")
_DEFAULT_OUTPUT = os.path.join(_SCRIPT_DIR, "..", "results", "glozz", "remarques.csv")

CORPUS_DIRS = {
    "Albert_dataset": os.path.join(_DATA_DIR, "Albert_dataset"),
    "CorpusCovid": os.path.join(_DATA_DIR, "CorpusCovid"),
    "LitteratureJeunesse": os.path.join(_DATA_DIR, "LitteratureJeunesse"),
    "PtitLibe": os.path.join(_DATA_DIR, "PtitLibe"),
}

def _get_feature_value(feature_node) -> Optional[str]:
    """Extrait la valeur d'un nœud <feature>."""
    if feature_node is None:
        return None
    text = feature_node.text
    if text is None or text.strip() == "":
        return None
    return text.strip()

def _clean_span(text: str) -> str:
    """Normalise le span textuel."""
    return text.replace("\n", " ").replace("\r", " ").strip()


def _normalize_remarque(text: Optional[str]) -> Optional[str]:
    """Normalise, corrige les typos et dé-duplique les valeurs de remarque.

    Les valeurs multiples (séparées par /, ; ou ,) sont normalisées
    individuellement, dé-dupliquées, puis recombinées.
    """
    if not text:
        return None
    parts = re.split(r'[/;,]', text)
    normalized = []
    for p in parts:
        p = p.strip().lower()
        if not p:
            continue
        # Corrections de typos connues
        typo_map = {
            "emour": "amour",
            "stresse": "stress",
            "amour (apprécier)": "amour",
            "déterminsation": "détermination",
        }
        p = typo_map.get(p, p)
        if p not in normalized:
            normalized.append(p)
    if not normalized:
        return None
    return ", ".join(normalized)

def parse_remarques_from_pair(aa_filepath: str, ac_filepath: str, corpus_name: str) -> List[Dict[str, Any]]:
    """Parse une paire de fichiers .aa/.ac et extrait les remarques des unités 'Autre'."""
    # Lecture du texte brut (.ac)
    try:
        with open(ac_filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        print(f"Erreur lecture .ac {ac_filepath} : {e}", file=sys.stderr)
        return []

    # Lecture XML (.aa)
    try:
        tree = ET.parse(aa_filepath)
        root = tree.getroot()
    except Exception as e:
        print(f"Erreur lecture XML .aa {aa_filepath} : {e}", file=sys.stderr)
        return []

    file_id = os.path.basename(aa_filepath).replace(".aa", "")
    records = []

    # Recherche des unités
    for unit in root.findall(".//unit"):
        type_node = unit.find("./characterisation/type")
        if type_node is None:
            continue
        unit_type = (type_node.text or "").strip()
        
        # On cible uniquement "Autre"
        if unit_type.lower() != "autre":
            continue

        unit_id = unit.get("id", "")
        
        # Positionnement
        start_node = unit.find("./positioning/start/singlePosition")
        end_node = unit.find("./positioning/end/singlePosition")
        if start_node is None or end_node is None:
            continue

        try:
            start_idx = int(start_node.get("index", "-1"))
            end_idx = int(end_node.get("index", "-1"))
        except (ValueError, TypeError):
            continue

        if 0 <= start_idx <= end_idx <= len(raw_text):
            text_span = raw_text[start_idx:end_idx]
        else:
            text_span = ""

        # Extraction de la feature Remarque (insensible à la casse sur le nom de feature)
        remarque_val = None
        feature_set = unit.find("./characterisation/featureSet")
        if feature_set is not None:
            for feature in feature_set.findall("feature"):
                name = feature.get("name", "")
                if name.lower() == "remarque":
                    remarque_val = _normalize_remarque(_get_feature_value(feature))
                    break

        records.append({
            "corpus": corpus_name,
            "file_id": file_id,
            "unit_id": unit_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text_span": _clean_span(text_span),
            "remarque": remarque_val,
        })

    return records

def parse_all_remarques(corpus_dirs: Dict[str, str] = CORPUS_DIRS) -> pd.DataFrame:
    """Parcourt tous les corpus et extrait toutes les remarques."""
    all_records = []
    
    for corpus_name, corpus_dir in corpus_dirs.items():
        aa_dir = os.path.join(corpus_dir, "aa")
        ac_dir = os.path.join(corpus_dir, "ac")
        
        if not os.path.isdir(aa_dir) or not os.path.isdir(ac_dir):
            print(f"Dossiers aa/ ou ac/ manquants dans {corpus_dir}", file=sys.stderr)
            continue
            
        aa_files = sorted(f for f in os.listdir(aa_dir) if f.endswith(".aa"))
        for aa_filename in aa_files:
            aa_path = os.path.join(aa_dir, aa_filename)
            ac_filename = aa_filename.replace(".aa", ".ac")
            ac_path = os.path.join(ac_dir, ac_filename)
            
            if os.path.isfile(ac_path):
                records = parse_remarques_from_pair(aa_path, ac_path, corpus_name)
                all_records.extend(records)
            else:
                print(f"Fichier .ac manquant pour {aa_path}", file=sys.stderr)
                
    return pd.DataFrame(all_records)

def get_unique_remarques(df: pd.DataFrame) -> List[str]:
    """Retourne la liste triée des remarques uniques non vides."""
    if "remarque" not in df.columns or df.empty:
        return []
    non_empty = df["remarque"].dropna().str.strip()
    non_empty = non_empty[non_empty != ""]
    return sorted(list(non_empty.unique()), key=lambda s: s.lower())

def main():
    parser = argparse.ArgumentParser(
        description="Parse spécifiquement les éléments Remarque de type Autre dans les corpus Glozz."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=_DEFAULT_OUTPUT,
        help=f"Chemin du fichier CSV de sortie (défaut: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("=== Démarrage du parsing des éléments Remarque ===")
    df = parse_all_remarques()

    if df.empty:
        print("Aucune donnée extraite.")
        sys.exit(1)

    print("\n=== Statistiques d'extraction ===")
    print(f"Total d'unités 'Autre' extraites : {len(df)}")
    print(f"Par corpus :")
    for corpus, count in df["corpus"].value_counts().items():
        print(f"  - {corpus} : {count}")
    
    # Nombre de remarques non-nulles / non-vides
    non_empty_remarques = df["remarque"].dropna().str.strip().ne("").sum()
    print(f"Nombre d'unités avec une remarque non vide : {non_empty_remarques} / {len(df)}")

    # Valeurs uniques de remarque avec proportions (ordre décroissant)
    total_units = len(df)
    remarque_counts = df["remarque"].dropna().str.strip()
    remarque_counts = remarque_counts[remarque_counts != ""].value_counts()
    print(f"\n=== Valeurs uniques de Remarque ({len(remarque_counts)}) ===")
    for val, count in remarque_counts.items():
        proportion = (count / total_units) * 100 if total_units > 0 else 0
        print(f"  - {val} ({proportion:.1f}%)")

    # Sauvegarde CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nRésultats exportés avec succès dans : {args.output}")

if __name__ == "__main__":
    main()
