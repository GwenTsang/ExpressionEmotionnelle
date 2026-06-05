#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contrat de schéma pour les marqueurs SimpleSitEmo normalisés.

Version simplifiée du marker_contract d'analysis_pipeline :
- La colonne condition est « emotion » (pas « categorie1 »).
- Pas de normalize_existing_marker_table() ni xlsx_markers_to_normalized()
  car l'extracteur gère toute la normalisation.
"""

from __future__ import annotations

import sys
from typing import Iterable

import pandas as pd

from .emotion_taxonomy import EMOTIONS, MODES


# ── Colonnes requises ─────────────────────────────────────────────────────

MARKER_COLUMNS: list[str] = ["marker_value", "marker_type"]

NORMALIZED_MARKER_COLUMNS: list[str] = MARKER_COLUMNS + ["type", "emotion", "mode"]


# ── Exception ─────────────────────────────────────────────────────────────


class MarkerSchemaError(ValueError):
    """Erreur levée quand une table de marqueurs ne respecte pas le schéma."""


# ── Fonctions de validation ───────────────────────────────────────────────


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    table_name: str,
) -> None:
    """Vérifie que *columns* sont toutes présentes dans *df*.

    Raises
    ------
    MarkerSchemaError
        Si au moins une colonne est absente.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise MarkerSchemaError(
            f"{table_name}: colonnes requises manquantes: {missing}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )


def validate_normalized_markers(
    df: pd.DataFrame,
    *,
    table_name: str = "normalized marker table",
    require_condition_columns: bool = True,
) -> None:
    """Valide le schéma minimal consommé par l'analyse de spécificité.

    Parameters
    ----------
    df : pd.DataFrame
        Table de marqueurs à valider.
    table_name : str
        Nom affiché dans les messages d'erreur.
    require_condition_columns : bool
        Si True, vérifie aussi la présence de « type », « emotion » et
        « mode » et contrôle leurs valeurs.
    """
    columns = NORMALIZED_MARKER_COLUMNS if require_condition_columns else MARKER_COLUMNS
    require_columns(df, columns, table_name)

    # Avertissement sur les NaN dans marker_value
    if df["marker_value"].isna().any():
        n_nan = int(df["marker_value"].isna().sum())
        print(
            f"{table_name}: marker_value contient {n_nan} valeur(s) NaN",
            file=sys.stderr,
        )

    if require_condition_columns:
        known_emotions = set(EMOTIONS + ["Autre"])
        unknown_emotions = sorted(
            str(v)
            for v in df["emotion"].dropna().unique()
            if v not in known_emotions
        )
        unknown_modes = sorted(
            str(v) for v in df["mode"].dropna().unique() if v not in MODES
        )
        if unknown_emotions:
            raise MarkerSchemaError(
                f"{table_name}: valeurs emotion inconnues: {unknown_emotions}"
            )
        if unknown_modes:
            raise MarkerSchemaError(
                f"{table_name}: valeurs mode inconnues: {unknown_modes}"
            )
