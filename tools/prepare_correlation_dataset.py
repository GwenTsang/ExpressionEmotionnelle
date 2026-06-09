"""Prépare un dataset binaire normalisé pour l'analyse de corrélation.

Transforme les colonnes catégorielles (ROLE, HATE, INTENTION, VERBAL_ABUSE)
en colonnes one-hot binaires, tout en conservant les colonnes catégorielles
originales pour permettre le chi-square global.

Usage :
    python tools/prepare_correlation_dataset.py
"""

import os

import numpy as np
import pandas as pd


CATEGORICAL_COLS = ["ROLE", "HATE", "INTENTION", "VERBAL_ABUSE"]

EMOTION_COLS = [
    "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
    "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
    "Surprise", "Tristesse",
]

MODE_COLS = ["Suggeree", "Montree", "Comportementale", "Designee"]

# Seules ces valeurs de VERBAL_ABUSE sont conservées
VERBAL_ABUSE_VALUES = ["BLM", "NCG", "THR", "DNG", "OTH"]


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(
        base_dir, "data", "raw", "xlsx",
        "CyberAdoAgg_gold_global_total_latest.xlsx",
    )
    output_path = os.path.join(
        base_dir, "data", "processed",
        "correlation_binary_dataset.xlsx",
    )

    # Lecture du fichier source
    df = pd.read_excel(input_path)

    # ------------------------------------------------------------------
    # 1. Nettoyage des colonnes catégorielles (strip des espaces)
    # ------------------------------------------------------------------
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            print(f"Colonne catégorielle ignorée : {col}")
            continue
        df[col] = df[col].astype(str).str.strip()
        # Remettre les NaN (converties en 'nan' par astype(str))
        df.loc[df[col] == "nan", col] = np.nan
        df.loc[df[col] == "", col] = np.nan

    # ------------------------------------------------------------------
    # 2. Filtrage de VERBAL_ABUSE (ne garder que les valeurs valides)
    # ------------------------------------------------------------------
    if "VERBAL_ABUSE" in df.columns:
        mask_invalid = ~df["VERBAL_ABUSE"].isin(VERBAL_ABUSE_VALUES) & df["VERBAL_ABUSE"].notna()
        n_filtered = mask_invalid.sum()
        df.loc[mask_invalid, "VERBAL_ABUSE"] = np.nan
        if n_filtered > 0:
            print(f"[+] VERBAL_ABUSE : {n_filtered} valeurs invalides remplacées par NaN")

    # ------------------------------------------------------------------
    # 3. One-hot encoding des colonnes catégorielles
    # ------------------------------------------------------------------
    onehot_frames = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        dummies.loc[df[col].isna(), :] = np.nan
        onehot_frames.append(dummies)
        print(f"[+] {col} : {dummies.shape[1]} colonnes one-hot créées → {list(dummies.columns)}")

    # ------------------------------------------------------------------
    # 4. Colonnes binaires (émotions et modes) → conversion en int 0/1
    # ------------------------------------------------------------------
    binary_cols = EMOTION_COLS + MODE_COLS
    existing_binary = [c for c in binary_cols if c in df.columns]
    missing_binary = [c for c in binary_cols if c not in df.columns]
    if missing_binary:
        print(f"Colonnes ignorées : {missing_binary}")

    for col in existing_binary:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] > 0).astype(int)

    # ------------------------------------------------------------------
    # 5. Assemblage du fichier de sortie
    # ------------------------------------------------------------------
    # Colonnes catégorielles originales
    existing_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    output_df = df[existing_cat].copy()

    # Colonnes one-hot
    for oh in onehot_frames:
        output_df = pd.concat([output_df, oh], axis=1)

    # Colonnes binaires (émotions + modes)
    for col in existing_binary:
        output_df[col] = df[col]

    # ------------------------------------------------------------------
    # 6. Sauvegarde
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_excel(output_path, index=False)

    print(f"\n[+] Dataset préparé sauvegardé : {output_path}")
    print(f"    - {len(output_df)} lignes")
    print(f"    - {len(output_df.columns)} colonnes")
    print(f"    - Colonnes catégorielles : {existing_cat}")
    print(f"    - Colonnes one-hot : {sum(oh.shape[1] for oh in onehot_frames)}")
    print(f"    - Colonnes binaires : {len(existing_binary)}")


if __name__ == "__main__":
    main()
