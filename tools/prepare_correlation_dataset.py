"""Prépare un dataset binaire normalisé pour l'analyse de corrélation.

Transforme les colonnes catégorielles (ROLE, TARGET, HATE, INTENTION,
VERBAL_ABUSE) en colonnes binaires, tout en conservant les colonnes
catégorielles originales pour permettre le chi-square global.
"""

import os
import re

import numpy as np
import pandas as pd


ROLE_VALUES = ["victim", "bully", "victim_support", "bully_support", "conciliator"]

CATEGORY_VALUES = {
    "ROLE": ROLE_VALUES,
    "TARGET": ROLE_VALUES,
    "HATE": ["OAG", "CAG", "NAG"],
    "INTENTION": ["ATK", "DFN", "CNS", "AIN", "GSL", "EMP", "CR", "OTH"],
    "VERBAL_ABUSE": ["BLM", "NCG", "THR", "DNG", "OTH"],
}
CATEGORICAL_COLS = list(CATEGORY_VALUES)

EMOTION_COLS = [
    "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
    "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
    "Surprise", "Tristesse",
]

MODE_COLS = ["Suggeree", "Montree", "Comportementale", "Designee"]

MISSING_CATEGORY_VALUES = {"", "nan", "null", "none", "<na>"}
TARGET_SEPARATOR_RE = re.compile(
    r"\s*(?:/|;|,|\||\+|&|\bet\b|\band\b)\s*",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def clean_categorical_column(series, col, valid_values):
    cleaned = series.astype("string").str.strip()
    missing_mask = (
        cleaned.isna() | cleaned.str.casefold().isin(MISSING_CATEGORY_VALUES)
    )
    cleaned = cleaned.mask(missing_mask, np.nan)

    canonical_by_case = {value.casefold(): value for value in valid_values}
    canonical = cleaned.str.casefold().map(canonical_by_case)
    canonicalized_mask = (
        cleaned.notna() & canonical.notna() & (cleaned != canonical)
    ).fillna(False)

    if canonicalized_mask.any():
        cleaned = cleaned.mask(canonical.notna(), canonical)
        print(
            f"[+] {col} : {int(canonicalized_mask.sum())} valeurs "
            "normalisées vers la casse canonique"
        )

    invalid_mask = cleaned.notna() & ~cleaned.isin(valid_values)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        examples = [
            str(value)[:80]
            for value in cleaned.loc[invalid_mask].drop_duplicates().head(3)
        ]
        cleaned = cleaned.mask(invalid_mask, np.nan)
        print(
            f"[+] {col} : {n_invalid} valeurs invalides remplacées par NaN "
            f"(exemples : {examples})"
        )

    return cleaned


def is_target_disagreement(text):
    lowered = text.casefold()
    return lowered.startswith("file:") and lowered.endswith("null")


def parse_target_cell(value, valid_values):
    if pd.isna(value):
        return np.nan, tuple(), "missing"

    text = str(value).strip()
    if text.casefold() in MISSING_CATEGORY_VALUES:
        return np.nan, tuple(), "missing"
    if is_target_disagreement(text):
        return np.nan, tuple(), "disagreement"

    canonical_by_case = {valid.casefold(): valid for valid in valid_values}
    pieces = [piece for piece in TARGET_SEPARATOR_RE.split(text) if piece.strip()]

    roles = []
    invalid = []
    for piece in pieces:
        canonical = canonical_by_case.get(piece.casefold())
        if canonical is None:
            invalid.append(piece)
        elif canonical not in roles:
            roles.append(canonical)

    if invalid or not roles:
        return np.nan, tuple(), "invalid"

    status = "valid" if len(roles) == 1 else "multi"
    return "/".join(roles), tuple(roles), status


def clean_target_column(series, valid_values):
    parsed = series.map(lambda value: parse_target_cell(value, valid_values))
    cleaned = pd.Series(
        [item[0] for item in parsed],
        index=series.index,
        name=series.name,
        dtype="string",
    )
    role_sets = pd.Series(
        [item[1] for item in parsed],
        index=series.index,
        name=series.name,
        dtype=object,
    )
    statuses = pd.Series(
        [item[2] for item in parsed],
        index=series.index,
        name=series.name,
    )

    n_valid = int(statuses.eq("valid").sum())
    n_multi = int(statuses.eq("multi").sum())
    n_disagreement = int(statuses.eq("disagreement").sum())
    n_missing = int(statuses.eq("missing").sum())
    n_invalid = int(statuses.eq("invalid").sum())

    print(
        "[+] TARGET : "
        f"{n_valid} valeurs simples valides, "
        f"{n_multi} valeurs multi-cibles, "
        f"{n_disagreement} désaccords ignorés, "
        f"{n_missing} valeurs manquantes"
    )

    if n_invalid > 0:
        examples = [
            str(value)[:80]
            for value in series.loc[statuses.eq("invalid")]
            .drop_duplicates()
            .head(3)
        ]
        print(
            f"[+] TARGET : {n_invalid} valeurs invalides remplacées par NaN "
            f"(exemples : {examples})"
        )

    return cleaned, role_sets


def make_target_dummies(role_sets, valid_values):
    return pd.DataFrame(
        {
            f"TARGET_{role}": role_sets.map(
                lambda roles, role=role: np.nan
                if not roles
                else float(role in roles)
            )
            for role in valid_values
        },
        index=role_sets.index,
    )


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
    # 1. Nettoyage et validation des colonnes catégorielles
    # ------------------------------------------------------------------
    target_role_sets = None
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            print(f"Colonne catégorielle ignorée : {col}")
            continue
        if col == "TARGET":
            df[col], target_role_sets = clean_target_column(
                df[col], CATEGORY_VALUES[col]
            )
        else:
            df[col] = clean_categorical_column(df[col], col, CATEGORY_VALUES[col])

    # ------------------------------------------------------------------
    # 2. Encodage binaire des colonnes catégorielles
    # ------------------------------------------------------------------
    onehot_frames = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if col == "TARGET":
            dummies = make_target_dummies(target_role_sets, CATEGORY_VALUES[col])
        else:
            category_series = pd.Series(
                pd.Categorical(df[col], categories=CATEGORY_VALUES[col]),
                index=df.index,
                name=col,
            )
            dummies = pd.get_dummies(category_series, prefix=col, dtype=float)
            dummies.loc[df[col].isna(), :] = np.nan
        onehot_frames.append(dummies)
        print(
            f"[+] {col} : {dummies.shape[1]} colonnes binaires créées "
            f"→ {list(dummies.columns)}"
        )

    # ------------------------------------------------------------------
    # 3. Colonnes binaires (émotions et modes) → conversion en int 0/1
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
    # 4. Assemblage du fichier de sortie
    # ------------------------------------------------------------------
    # Colonnes catégorielles originales
    existing_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    output_df = df[existing_cat].copy()

    # Colonnes catégorielles encodées en binaire
    for oh in onehot_frames:
        output_df = pd.concat([output_df, oh], axis=1)

    # Colonnes binaires (émotions + modes)
    for col in existing_binary:
        output_df[col] = df[col]

    # ------------------------------------------------------------------
    # 5. Sauvegarde
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_excel(output_path, index=False)

    print(f"\n[+] Dataset préparé sauvegardé : {output_path}")
    print(f"    - {len(output_df)} lignes")
    print(f"    - {len(output_df.columns)} colonnes")
    print(f"    - Colonnes catégorielles : {existing_cat}")
    print(
        "    - Colonnes catégorielles binaires : "
        f"{sum(oh.shape[1] for oh in onehot_frames)}"
    )
    print(f"    - Colonnes binaires : {len(existing_binary)}")


if __name__ == "__main__":
    main()
