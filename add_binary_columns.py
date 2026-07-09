#!/usr/bin/env python3
"""
add_binary_columns.py
=====================
Prend en entrée un xlsx dépourvu des 19 colonnes binaires
et reconstruit ces colonnes à partir des colonnes Sit_Emo_unit_*.

Les 19 colonnes reconstruites sont :
  - 6 émotions de base   : Colere, Degout, Joie, Peur, Surprise, Tristesse
  - 5 émotions complexes  : Admiration, Culpabilite, Embarras, Fierte, Jalousie
  - 1 catégorie autre     : Autre
  - 4 modes d'expression  : Comportementale, Designee, Montree, Suggeree
  - 3 méta-labels         : Emo, Base, Complexe

Méthodologie des méta-labels
-----------------------------
Soit E(S) = (e_1, ..., e_12) le vecteur des 12 labels émotionnels booléens.

  y_emo(S)      = 1  si  Σ e_i > 0   (au moins une émotion parmi les 12)
  y_base(S)     = 1  si  Σ e_i > 0   pour e_i ∈ {Colere..Tristesse}
  y_complexe(S) = 1  si  Σ e_i > 0   pour e_i ∈ {Admiration..Jalousie}

Usage
-----
  python add_binary_columns.py <input.xlsx> [output.xlsx]

Si output.xlsx n'est pas spécifié, le fichier de sortie est
  <input_sans_extension>_with_binary.xlsx
"""

import sys
import pandas as pd

# ── Mappings : nom de colonne binaire → valeur attendue dans Sit_Emo_unit ──

EMOTION_MAP = {
    "Colere":      "Colère",
    "Degout":      "Dégoût",
    "Joie":        "Joie",
    "Peur":        "Peur",
    "Surprise":    "Surprise",
    "Tristesse":   "Tristesse",
    "Admiration":  "Admiration",
    "Culpabilite": "Culpabilité",
    "Embarras":    "Embarras",
    "Fierte":      "Fierté",
    "Jalousie":    "Jalousie",
    "Autre":       "Autre",
}

MODE_MAP = {
    "Comportementale": "Comportementale",
    "Designee":        "Désignée",
    "Montree":         "Montrée",
    "Suggeree":        "Suggérée",
}

EMOTIONS_BASE     = ["Colere", "Degout", "Joie", "Peur", "Surprise", "Tristesse"]
EMOTIONS_COMPLEXE = ["Admiration", "Culpabilite", "Embarras", "Fierte", "Jalousie"]
ALL_EMOTIONS      = EMOTIONS_BASE + EMOTIONS_COMPLEXE + ["Autre"]


def _collect_unit_columns(df, suffix):
    """Retourne la liste des colonnes Sit_Emo_unit_*_<suffix> présentes."""
    return [c for c in df.columns
            if c.startswith("Sit_Emo_unit_") and c.endswith(f"_{suffix}")
            and "cas_limite" not in c]


def add_binary_columns(df):
    """Ajoute les 19 colonnes binaires au DataFrame *en place* et le retourne."""

    # ── Colonnes sources ──
    emotion_src_cols = (
        _collect_unit_columns(df, "emotion1")
        + _collect_unit_columns(df, "emotion2")
        + _collect_unit_columns(df, "emotion3")
    )
    mode_src_cols = _collect_unit_columns(df, "mode")

    # ── 12 colonnes émotionnelles ──
    for col_name, label_value in EMOTION_MAP.items():
        df[col_name] = df[emotion_src_cols].apply(
            lambda row: int(any(v == label_value for v in row)), axis=1
        )

    # ── 4 colonnes de modes d'expression ──
    for col_name, label_value in MODE_MAP.items():
        df[col_name] = df[mode_src_cols].apply(
            lambda row: int(any(v == label_value for v in row)), axis=1
        )

    # ── 3 méta-labels ──
    df["Emo"]      = (df[ALL_EMOTIONS].sum(axis=1) > 0).astype(int)
    df["Base"]     = (df[EMOTIONS_BASE].sum(axis=1) > 0).astype(int)
    df["Complexe"] = (df[EMOTIONS_COMPLEXE].sum(axis=1) > 0).astype(int)

    return df


# ── Point d'entrée ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        stem = input_path.rsplit(".", 1)[0]
        output_path = f"{stem}_with_binary.xlsx"

    print(f"Lecture de {input_path} …")
    df = pd.read_excel(input_path)
    print(f"  → {df.shape[0]} lignes × {df.shape[1]} colonnes")

    df = add_binary_columns(df)
    print(f"  → 19 colonnes binaires ajoutées ({df.shape[1]} colonnes au total)")

    df.to_excel(output_path, index=False)
    print(f"Fichier sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
