#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_simplesitemo_glozz.py — Extraction Glozz → SimpleSitEmo.

Utilise le parser Glozz existant (analysis_pipeline.glozz_parser) pour
extraire les annotations SitEmo des quatre corpus et produire un
DataFrame conforme au schéma SimpleSitEmo.

Schéma produit :
    source_file, text_span, mode, emotion1, emotion2, emotion3,
    nature_linguistique
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .emotion_taxonomy import normalize_emotion, normalize_mode

# ── Constantes ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")

_DEFAULT_OUTPUT = os.path.join(_PROJECT_DIR, "data", "SimpleSitEmo_glozz.parquet")

# Colonnes du schéma SimpleSitEmo
SCHEMA_COLUMNS = [
    "source_file", "text_span", "mode",
    "emotion1", "emotion2", "emotion3",
    "nature_linguistique",
]


# ── Extraction ────────────────────────────────────────────────────────────


def build_simplesitemo_glozz() -> pd.DataFrame:
    """Construit le DataFrame SimpleSitEmo à partir des corpus Glozz.

    Utilise ``analysis_pipeline.glozz_parser.process_all_corpora()`` pour
    lire les 4 corpus, puis filtre et transforme les unités SitEmo.

    Returns
    -------
    pd.DataFrame
        DataFrame conforme au schéma SimpleSitEmo.
    """
    # Import du parser existant (chemin relatif au projet)
    sys.path.insert(0, _PROJECT_DIR)
    from glozz import process_all_corpora

    print("=== Extraction Glozz → SimpleSitEmo ===")
    raw_df = process_all_corpora()

    if raw_df.empty:
        print("Aucune annotation extraite — arrêt.", file=sys.stderr)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    print(f"\nAnnotations brutes : {len(raw_df)}")
    print(f"  Par type : {raw_df['type'].value_counts().to_dict()}")

    # Filtrer : garder uniquement SitEmo (exclure Autre)
    sitemo_df = raw_df[raw_df["type"] == "SitEmo"].copy()
    n_autre = len(raw_df[raw_df["type"] == "Autre"])
    print(f"  Autre exclues : {n_autre}")

    # Vérifier les modes manquants
    n_no_mode = sitemo_df["mode"].isna().sum()
    if n_no_mode > 0:
        print(f"  SitEmo sans mode : {n_no_mode} — exclus")
        sitemo_df = sitemo_df[sitemo_df["mode"].notna()].copy()

    print(f"  SitEmo retenues : {len(sitemo_df)}")

    # Construire le DataFrame SimpleSitEmo
    records: list[dict] = []
    for _, row in sitemo_df.iterrows():
        mode = normalize_mode(row["mode"])
        emotion1 = normalize_emotion(row.get("categorie1"))
        emotion2 = normalize_emotion(row.get("categorie2"))

        # nature_linguistique depuis la colonne 'nature' (ajoutée au parser)
        nature = row.get("nature")
        if nature is not None and (isinstance(nature, float) and pd.isna(nature)):
            nature = None
        elif nature is not None:
            nature = str(nature).strip() or None

        records.append({
            "source_file": row["corpus"],
            "text_span": row["text_span"],
            "mode": mode,
            "emotion1": emotion1,
            "emotion2": emotion2,
            "emotion3": None,  # Toujours null pour Glozz
            "nature_linguistique": nature,
        })

    result = pd.DataFrame(records, columns=SCHEMA_COLUMNS)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction Glozz → SimpleSitEmo parquet."
    )
    parser.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUTPUT,
        help=f"Fichier parquet de sortie (défaut : {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    result = build_simplesitemo_glozz()

    if result.empty:
        print("Aucune unité — pas de fichier écrit.", file=sys.stderr)
        sys.exit(1)

    # Écriture parquet
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_parquet(args.output, index=False, engine="pyarrow")
    print(f"\n✓ Écrit {len(result)} unités dans {args.output}")

    # Résumé
    print(f"  Unités totales       : {len(result)}")
    print(f"  Sources              : {result['source_file'].value_counts().to_dict()}")
    print(f"  Modes                : {result['mode'].value_counts(dropna=False).to_dict()}")
    print(f"  emotion1             : {result['emotion1'].value_counts(dropna=False).to_dict()}")
    n_with_e2 = result["emotion2"].notna().sum()
    print(f"  Avec emotion2        : {n_with_e2}")
    n_nature = result["nature_linguistique"].notna().sum()
    print(f"  nature_linguistique  : {n_nature}/{len(result)} renseignée")


if __name__ == "__main__":
    main()
