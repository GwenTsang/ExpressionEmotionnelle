#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_simplesitemo.py — Fusion XLSX + Glozz → SimpleSitEmo.parquet.

Lit les deux fichiers parquet intermédiaires (XLSX et Glozz), valide le
schéma et les valeurs, puis produit le fichier unifié SimpleSitEmo.parquet.

Pipeline 2 : le schéma inclut une colonne ``text_span_source`` qui
indique si le text_span provient du déclencheur ou du segment complet.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .emotion_taxonomy import ALL_EMOTIONS, MODES

# ── Constantes ────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")

_DEFAULT_XLSX = os.path.join(_PROJECT_DIR, "data", "SimpleSitEmo_xlsx.parquet")
_DEFAULT_GLOZZ = os.path.join(_PROJECT_DIR, "data", "SimpleSitEmo_glozz.parquet")
_DEFAULT_OUTPUT = os.path.join(_PROJECT_DIR, "data", "SimpleSitEmo.parquet")

EXPECTED_COLUMNS = [
    "source_file", "text_span", "text_span_source", "mode",
    "emotion1", "emotion2", "emotion3",
    "nature_linguistique",
]


# ── Validation ────────────────────────────────────────────────────────────


def _validate_schema(df: pd.DataFrame, label: str) -> None:
    """Vérifie que le DataFrame a exactement les colonnes attendues."""
    actual = list(df.columns)
    if actual != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual)
        extra = set(actual) - set(EXPECTED_COLUMNS)
        msg = f"Schéma invalide pour {label}."
        if missing:
            msg += f" Colonnes manquantes : {missing}."
        if extra:
            msg += f" Colonnes en trop : {extra}."
        raise ValueError(msg)


def _validate_values(df: pd.DataFrame, label: str) -> None:
    """Vérifie que les valeurs d'émotion et de mode sont canoniques."""
    # Valider les modes
    valid_modes = set(MODES)
    mode_values = df["mode"].dropna().unique()
    invalid_modes = set(mode_values) - valid_modes
    if invalid_modes:
        raise ValueError(
            f"{label} : modes non canoniques trouvés : {invalid_modes}"
        )

    # Valider les émotions
    valid_emotions = set(ALL_EMOTIONS)
    for col in ("emotion1", "emotion2", "emotion3"):
        col_values = df[col].dropna().unique()
        invalid = set(col_values) - valid_emotions
        if invalid:
            raise ValueError(
                f"{label} : {col} contient des valeurs non canoniques : {invalid}"
            )


def _check_source_collision(
    xlsx_df: pd.DataFrame, glozz_df: pd.DataFrame
) -> None:
    """Vérifie qu'il n'y a pas de collision de source_file entre les deux sources."""
    xlsx_sources = set(xlsx_df["source_file"].unique())
    glozz_sources = set(glozz_df["source_file"].unique())
    collision = xlsx_sources & glozz_sources
    if collision:
        raise ValueError(
            f"Collision de source_file entre XLSX et Glozz : {collision}"
        )


# ── Fusion ────────────────────────────────────────────────────────────────


def merge_simplesitemo(
    xlsx_path: str, glozz_path: str
) -> pd.DataFrame:
    """Fusionne les deux fichiers parquet SimpleSitEmo.

    Parameters
    ----------
    xlsx_path : str
        Chemin vers SimpleSitEmo_xlsx.parquet.
    glozz_path : str
        Chemin vers SimpleSitEmo_glozz.parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame unifié conforme au schéma SimpleSitEmo.
    """
    print(f"Lecture XLSX  : {xlsx_path}")
    xlsx_df = pd.read_parquet(xlsx_path, engine="pyarrow")
    print(f"  {len(xlsx_df)} unités")

    # XLSX n'a pas de colonne text_span_source : on l'ajoute par défaut
    if "text_span_source" not in xlsx_df.columns:
        xlsx_df["text_span_source"] = "segment_complet"
        # Remettre les colonnes dans l'ordre attendu
        xlsx_df = xlsx_df[EXPECTED_COLUMNS]

    print(f"Lecture Glozz : {glozz_path}")
    glozz_df = pd.read_parquet(glozz_path, engine="pyarrow")
    print(f"  {len(glozz_df)} unités")

    # Validations
    print("\nValidation des schémas…")
    _validate_schema(xlsx_df, "XLSX")
    _validate_schema(glozz_df, "Glozz")
    print("  ✓ Schémas conformes")

    print("Vérification des collisions source_file…")
    _check_source_collision(xlsx_df, glozz_df)
    print("  ✓ Pas de collision")

    print("Validation des valeurs (émotions / modes)…")
    _validate_values(xlsx_df, "XLSX")
    _validate_values(glozz_df, "Glozz")
    print("  ✓ Valeurs canoniques")

    # Concaténation
    result = pd.concat([xlsx_df, glozz_df], ignore_index=True)
    print(f"\n  Total fusionné : {len(result)} unités")

    # Résumé text_span_source
    if "text_span_source" in result.columns:
        print(f"  text_span_source : {result['text_span_source'].value_counts().to_dict()}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusion XLSX + Glozz → SimpleSitEmo.parquet."
    )
    parser.add_argument(
        "--xlsx",
        default=_DEFAULT_XLSX,
        help=f"Parquet XLSX (défaut : {_DEFAULT_XLSX})",
    )
    parser.add_argument(
        "--glozz",
        default=_DEFAULT_GLOZZ,
        help=f"Parquet Glozz (défaut : {_DEFAULT_GLOZZ})",
    )
    parser.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUTPUT,
        help=f"Parquet de sortie (défaut : {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    result = merge_simplesitemo(args.xlsx, args.glozz)

    # Écriture parquet
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_parquet(args.output, index=False, engine="pyarrow")
    print(f"\n✓ Écrit {len(result)} unités dans {args.output}")

    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ SimpleSitEmo UNIFIÉ")
    print("=" * 60)
    print(f"  Unités totales           : {len(result)}")
    print(f"  Sources                  : {result['source_file'].value_counts().to_dict()}")
    print(f"  Modes                    : {result['mode'].value_counts(dropna=False).to_dict()}")
    print(f"  emotion1                 : {result['emotion1'].value_counts(dropna=False).to_dict()}")
    n_e2 = result["emotion2"].notna().sum()
    n_e3 = result["emotion3"].notna().sum()
    print(f"  Avec emotion2            : {n_e2}")
    print(f"  Avec emotion3            : {n_e3}")
    n_nature = result["nature_linguistique"].notna().sum()
    print(f"  nature_linguistique      : {n_nature}/{len(result)} renseignée")


if __name__ == "__main__":
    main()
