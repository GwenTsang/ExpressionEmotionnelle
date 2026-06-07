#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_simplesitemo_xlsx.py — Extraction XLSX → SimpleSitEmo.

Lit le fichier CyberAdoAgg_gold_global_total_latest.xlsx et produit un
DataFrame conforme au schéma SimpleSitEmo (une ligne par segment
émotionnel annoté).

Schéma produit :
    source_file, text_span, text_span_source, mode, emotion1,
    emotion2, emotion3, nature_linguistique
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd

from .emotion_taxonomy import normalize_emotion, normalize_mode

# ── Constantes ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")

_DEFAULT_INPUT = os.path.join(
    _PROJECT_DIR, "data", "raw", "xlsx",
    "CyberAdoAgg_gold_global_total_latest.xlsx",
)
_DEFAULT_OUTPUT = os.path.join(_PROJECT_DIR, "data", "SimpleSitEmo_xlsx.parquet")

_SOURCE_FILE_LABEL = "CyberAggAdo"
_MAX_SPANS = 4
_MAX_EMOTIONS = 3

# Colonnes du schéma SimpleSitEmo
SCHEMA_COLUMNS = [
    "source_file", "text_span", "text_span_source", "mode",
    "emotion1", "emotion2", "emotion3",
    "nature_linguistique",
]


# ── Extraction ────────────────────────────────────────────────────────────


def _parse_emotions(cat_value: object) -> list[str]:
    """Parse la colonne spanN_cat (potentiellement multi-émotion via ' + ').

    Retourne une liste de labels canoniques (dédupliqués, max 3).
    """
    if cat_value is None or (isinstance(cat_value, float) and pd.isna(cat_value)):
        return []
    raw = str(cat_value).strip()
    if not raw:
        return []

    parts = [p.strip() for p in raw.split(" + ")]
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        canon = normalize_emotion(part)
        if canon is not None and canon not in seen:
            normalized.append(canon)
            seen.add(canon)
    return normalized[:_MAX_EMOTIONS]


def _extract_spans_from_row(row: pd.Series) -> list[dict]:
    """Extrait les unités SimpleSitEmo depuis une ligne XLSX (Emo == 1)."""
    units: list[dict] = []
    for n in range(1, _MAX_SPANS + 1):
        text = row.get(f"span{n}_text")
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        text = str(text).strip()
        if not text:
            continue

        cat = row.get(f"span{n}_cat")
        raw_mode = row.get(f"span{n}_mode")
        nature = row.get(f"nature_linguistique_span_{n}")

        mode = normalize_mode(raw_mode)
        emotions = _parse_emotions(cat)

        # nature_linguistique : garder tel quel, None si NaN
        if nature is not None and (isinstance(nature, float) and pd.isna(nature)):
            nature = None
        elif nature is not None:
            nature = str(nature).strip() or None

        unit: dict = {
            "source_file": _SOURCE_FILE_LABEL,
            "text_span": text,
            "text_span_source": "segment_complet",
            "mode": mode,
            "emotion1": emotions[0] if len(emotions) > 0 else None,
            "emotion2": emotions[1] if len(emotions) > 1 else None,
            "emotion3": emotions[2] if len(emotions) > 2 else None,
            "nature_linguistique": nature,
            # Clé interne pour la fusion — sera supprimée ensuite
            "_source_idx": row.name,
        }
        units.append(unit)
    return units


def _merge_duplicate_spans(units: list[dict]) -> list[dict]:
    """Fusionne les doublons (même text_span + mode) au sein d'une même ligne source.

    Combine les émotions (dédupliquées, max 3) et conserve la première
    nature_linguistique non-nulle.
    """
    from collections import defaultdict

    # Grouper par (_source_idx, text_span, mode)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for u in units:
        key = (u["_source_idx"], u["text_span"], u["mode"])
        groups[key].append(u)

    merged: list[dict] = []
    for (_src_idx, text_span, mode), group in groups.items():
        # Collecter toutes les émotions uniques
        all_emotions: list[str] = []
        seen: set[str] = set()
        nature: Optional[str] = None

        for u in group:
            for ek in ("emotion1", "emotion2", "emotion3"):
                e = u.get(ek)
                if e is not None and e not in seen:
                    all_emotions.append(e)
                    seen.add(e)
            if nature is None and u.get("nature_linguistique") is not None:
                nature = u["nature_linguistique"]

        all_emotions = all_emotions[:_MAX_EMOTIONS]

        merged.append({
            "source_file": _SOURCE_FILE_LABEL,
            "text_span": text_span,
            "text_span_source": "segment_complet",
            "mode": mode,
            "emotion1": all_emotions[0] if len(all_emotions) > 0 else None,
            "emotion2": all_emotions[1] if len(all_emotions) > 1 else None,
            "emotion3": all_emotions[2] if len(all_emotions) > 2 else None,
            "nature_linguistique": nature,
        })

    return merged


def build_simplesitemo_xlsx() -> pd.DataFrame:
    """Construit le DataFrame SimpleSitEmo depuis le fichier XLSX par défaut."""
    print(f"Lecture du fichier XLSX : {_DEFAULT_INPUT}")
    df = pd.read_excel(_DEFAULT_INPUT)
    print(f"  {len(df)} lignes lues, colonnes : {len(df.columns)}")

    # Filtrer Emo == 1
    emo_df = df[df["Emo"] == 1].copy()
    print(f"  {len(emo_df)} lignes avec Emo == 1")

    # Extraire toutes les unités span
    all_units: list[dict] = []
    for idx, row in emo_df.iterrows():
        units = _extract_spans_from_row(row)
        all_units.extend(units)

    print(f"  {len(all_units)} unités extraites (avant fusion des doublons)")

    # Fusionner les doublons (même text_span + mode dans la même ligne)
    merged = _merge_duplicate_spans(all_units)
    print(f"  {len(merged)} unités après fusion des doublons")

    result = pd.DataFrame(merged, columns=SCHEMA_COLUMNS)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction XLSX → SimpleSitEmo parquet."
    )
    parser.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUTPUT,
        help=f"Fichier parquet de sortie (défaut : {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    result = build_simplesitemo_xlsx()

    # Écriture parquet
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_parquet(args.output, index=False, engine="pyarrow")
    print(f"\n✓ Écrit {len(result)} unités dans {args.output}")

    # Résumé
    print(f"  Unités totales       : {len(result)}")
    print(f"  source_file unique   : {result['source_file'].unique().tolist()}")
    print(f"  Modes                : {result['mode'].value_counts(dropna=False).to_dict()}")
    print(f"  emotion1             : {result['emotion1'].value_counts(dropna=False).to_dict()}")
    n_with_e2 = result["emotion2"].notna().sum()
    n_with_e3 = result["emotion3"].notna().sum()
    print(f"  Avec emotion2        : {n_with_e2}")
    print(f"  Avec emotion3        : {n_with_e3}")
    n_nature = result["nature_linguistique"].notna().sum()
    print(f"  nature_linguistique  : {n_nature}/{len(result)} renseignée")


if __name__ == "__main__":
    main()
