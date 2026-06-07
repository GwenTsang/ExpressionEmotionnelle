#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vocabulaire canonique des émotions et modes d'expression.

Les labels canoniques sont **accentués** (Colère, Dégoût, Désignée, etc.)
conformément au schéma SimpleSitEmo.  Ce module normalise les deux formes
attestées dans les données (accentuée XLSX, non-accentuée Glozz) vers
ces canoniques via un simple dictionnaire direct.
"""

from __future__ import annotations

from typing import Optional


# ── Labels canoniques (accentués) ─────────────────────────────────────────

EMOTIONS: list[str] = [
    "Colère",
    "Dégoût",
    "Joie",
    "Peur",
    "Surprise",
    "Tristesse",
    "Admiration",
    "Culpabilité",
    "Embarras",
    "Fierté",
    "Jalousie",
]

OPTIONAL_EMOTIONS: list[str] = ["Autre"]

ALL_EMOTIONS: list[str] = EMOTIONS + OPTIONAL_EMOTIONS

MODES: list[str] = ["Comportementale", "Désignée", "Montrée", "Suggérée"]

BASE_EMOTIONS: set[str] = {"Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse"}
COMPLEX_EMOTIONS: set[str] = {"Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie"}


# ── Normalisation (dict direct, pas de strip-accent) ─────────────────────
#
# Les données n'arrivent que sous deux formes attestées :
#   - Glozz : non-accentué (Colere, Degout, Designee, Montree, Suggeree)
#   - XLSX  : accentué    (Colère, Dégoût, Désignée, Montrée, Suggérée)
# Un mapping direct suffit.

# Auto-generate identity mappings from canonical lists.
# Only genuine aliases (Glozz non-accentué) need to be added manually.
_EMOTION_MAP: dict[str, str] = {e: e for e in ALL_EMOTIONS}
_EMOTION_MAP.update({
    # Glozz (non-accentué)
    "Colere": "Colère",
    "Degout": "Dégoût",
    "Culpabilite": "Culpabilité",
    "Fierte": "Fierté",
})

_MODE_MAP: dict[str, str] = {m: m for m in MODES}
_MODE_MAP.update({
    # Glozz (non-accentué)
    "Designee": "Désignée",
    "Montree": "Montrée",
    "Suggeree": "Suggérée",
})


def normalize_emotion(value: object, *, include_autre: bool = True) -> Optional[str]:
    """Normalise une valeur d'émotion vers le label canonique accentué.

    Retourne None pour les valeurs vides, inconnues, ou « Aucune ».
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    key = value.strip()
    if key == "Aucune":
        return None
    canonical = _EMOTION_MAP.get(key)
    if canonical == "Autre" and not include_autre:
        return None
    return canonical


def normalize_mode(value: object) -> Optional[str]:
    """Normalise une valeur de mode vers le label canonique accentué."""
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    return _MODE_MAP.get(value.strip())
