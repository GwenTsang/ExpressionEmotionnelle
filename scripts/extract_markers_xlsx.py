#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marker_extraction.py — Extraction des marqueurs linguistiques (format gold_flat XLSX)

Ce script prend en entrée un fichier XLSX « gold_flat » (ou CSV) et génère un
dataframe « marqueur × annotation » où chaque ligne correspond à un marqueur
(mot, lemme ou ponctuation) extrait du texte d'une phrase annotée.

Format d'entrée attendu (XLSX gold_flat, 56 colonnes) :
    - Identité : idx, ID, NAME
    - Texte    : TEXT
    - 12 émotions binaires : Colère, Dégoût, Joie, Peur, Surprise, Tristesse,
                             Admiration, Culpabilité, Embarras, Fierté, Jalousie, Autre
    - 4 modes d'expression binaires : Comportementale, Désignée, Montrée, Suggérée
    - 3 méta-catégories binaires    : Emo, Base, Complexe

Format de sortie (CSV) :
    idx, ID, NAME, TEXT,
    <12 émotions binaires>, <4 modes binaires>, Base, Complexe,
    <colonnes linguistiques optionnelles>, <colonnes contextuelles optionnelles>,
    marker_type, marker_value

Deux backends de lemmatisation sont disponibles :
  - spaCy  (CPU, batch via nlp.pipe())     — défaut
  - stanza (GPU CUDA uniquement, batch via bulk_process())
"""

import os
import re
import sys
import argparse
import logging
import time
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from nlp_utils import (
    FR_STOPWORDS,
    extract_words,
    extract_punctuations,
    get_lemmatizer,
)


# ---------------------------------------------------------------------------
# Lecture auto-détectée (XLSX / CSV)
# ---------------------------------------------------------------------------


def read_input(path: str) -> pd.DataFrame:
    """Lit un fichier d'annotations en détectant automatiquement le format.

    Supporte les extensions .xlsx, .xls et .csv.

    Parameters
    ----------
    path : str
        Chemin vers le fichier d'entrée.

    Returns
    -------
    pd.DataFrame
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        logger.info("Format détecté : Excel (%s)", ext)
        return pd.read_excel(path)
    elif ext == ".csv":
        logger.info("Format détecté : CSV")
        return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    else:
        # Tenter Excel par défaut
        logger.warning(
            "Extension '%s' non reconnue, tentative de lecture Excel...", ext
        )
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


# ---------------------------------------------------------------------------
# Construction du base_record (propagation des colonnes)
# ---------------------------------------------------------------------------


def _build_base_record(row: pd.Series, propagate_columns: list[str]) -> dict:
    """Construit le dictionnaire de métadonnées à propager pour chaque marqueur.

    Parameters
    ----------
    row : pd.Series
        Une ligne du dataframe d'annotations.
    propagate_columns : list[str]
        Liste des noms de colonnes à inclure dans le record.

    Returns
    -------
    dict
    """
    return {col: row[col] for col in propagate_columns}


# ---------------------------------------------------------------------------
# Fonction principale : build_marker_dataframe (adapté gold_flat)
# ---------------------------------------------------------------------------


def build_marker_dataframe(
    annotations_df: pd.DataFrame,
    use_lemma: bool = True,
    include_empty: bool = False,
    lemmatizer_backend: str = "spacy",
    batch_size: int = 256,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Construit le dataframe marqueur × annotation à partir d'un XLSX gold_flat.

    Pour chaque phrase annotée, extrait les mots, ponctuations, et
    optionnellement les lemmes de la colonne TEXT. Chaque marqueur
    produit une ligne avec les métadonnées de l'annotation source.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Le dataframe d'annotations gold_flat (XLSX, 56 colonnes).
    use_lemma : bool
        Si True, extrait aussi les lemmes.
    include_empty : bool
        Si True, inclut les lignes sans émotion active (Emo != 1).
        Si False, seules les lignes avec Emo == 1 sont traitées.
    lemmatizer_backend : str
        "spacy" ou "stanza".
    batch_size : int
        Taille de batch pour la lemmatisation.
    remove_stopwords : bool
        Si True, filtre les mots vides français.

    Returns
    -------
    pd.DataFrame
        Un dataframe avec les colonnes :
        idx, ID, NAME, TEXT,
        <12 émotions binaires>, <4 modes binaires>, Base, Complexe,
        <colonnes linguistiques/contextuelles optionnelles>,
        marker_type, marker_value
    """
    df = annotations_df.copy()

    # --- Vérification des colonnes requises ---
    required_cols = {"idx", "ID", "NAME", "TEXT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes requises manquantes dans le fichier d'entrée : {missing}. "
            f"Colonnes disponibles : {list(df.columns)[:20]}..."
        )

    # --- Filtrage des lignes sans émotion active ---
    if not include_empty:
        before = len(df)
        if "Emo" in df.columns:
            df = df[df["Emo"] == 1].copy()
        else:
            # Fallback : vérifier si au moins une émotion binaire est active
            emo_cols_present = [c for c in EMOTION_COLUMNS if c in df.columns]
            if emo_cols_present:
                df = df[df[emo_cols_present].sum(axis=1) > 0].copy()
            else:
                logger.warning(
                    "Aucune colonne d'émotion trouvée, aucun filtrage appliqué."
                )
        dropped = before - len(df)
        if dropped > 0:
            logger.info(
                "Filtrage : %d lignes sans émotion active exclues "
                "(--include-empty pour les garder)",
                dropped,
            )

    # --- Filtrage des TEXT vides ou NaN ---
    before = len(df)
    df = df[df["TEXT"].notna() & (df["TEXT"].astype(str).str.strip() != "")].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Filtrage : %d lignes avec TEXT vide/NaN exclues", dropped)

    df = df.reset_index(drop=True)

    total = len(df)
    if total == 0:
        logger.warning("Aucune ligne à traiter après filtrage.")
        return pd.DataFrame()

    # --- Déterminer les colonnes à propager ---
    # Colonnes obligatoires
    propagate = ["idx", "ID", "NAME", "TEXT"]

    # Colonnes binaires d'annotation (émotions + modes + méta)
    for col in BINARY_ANNOTATION_COLUMNS:
        if col in df.columns:
            propagate.append(col)

    # Colonnes linguistiques optionnelles
    for col in LINGUISTIC_COLUMNS:
        if col in df.columns:
            propagate.append(col)

    # Colonnes contextuelles optionnelles
    for col in CONTEXT_COLUMNS:
        if col in df.columns:
            propagate.append(col)

    logger.info(
        "Colonnes propagées dans la sortie : %d colonnes + marker_type/marker_value",
        len(propagate),
    )

    # Convertir TEXT en str pour éviter les erreurs sur des valeurs numériques
    df["TEXT"] = df["TEXT"].astype(str)
    texts = df["TEXT"].tolist()

    # --- Extraction des mots et ponctuations (rapide, pas de NLP) ---
    logger.info("Extraction des mots et ponctuations (%d annotations)...", total)
    t0 = time.time()

    all_marker_rows = []
    for i, row in df.iterrows():
        text = row["TEXT"]
        base_record = _build_base_record(row, propagate)

        # Mots
        for word in extract_words(text):
            if remove_stopwords and word in FR_STOPWORDS:
                continue
            r = base_record.copy()
            r["marker_type"] = "word"
            r["marker_value"] = word
            all_marker_rows.append(r)

        # Ponctuations
        for punct in extract_punctuations(text):
            r = base_record.copy()
            r["marker_type"] = "punctuation"
            r["marker_value"] = punct
            all_marker_rows.append(r)

    t_words = time.time() - t0
    n_words_punct = len(all_marker_rows)
    logger.info(
        "Mots/ponctuations extraits : %d marqueurs en %.1fs", n_words_punct, t_words
    )

    # --- Lemmatisation en batch ---
    if use_lemma:
        logger.info(
            "Lemmatisation (%s, batch_size=%d) de %d textes...",
            lemmatizer_backend,
            batch_size,
            total,
        )
        t0 = time.time()

        lemmatizer = get_lemmatizer(lemmatizer_backend, batch_size=batch_size)
        all_lemmas = lemmatizer.lemmatize_batch(texts)

        t_lemma = time.time() - t0
        n_lemmas = sum(len(l) for l in all_lemmas)
        logger.info(
            "Lemmatisation terminée : %d lemmes en %.1fs (%.0f textes/s)",
            n_lemmas,
            t_lemma,
            total / t_lemma if t_lemma > 0 else 0,
        )

        # Ajouter les lemmes au résultat
        for i, row in df.iterrows():
            base_record = _build_base_record(row, propagate)
            for lemma in all_lemmas[i]:
                if remove_stopwords and lemma in FR_STOPWORDS:
                    continue
                r = base_record.copy()
                r["marker_type"] = "lemma"
                r["marker_value"] = lemma
                all_marker_rows.append(r)

    result = pd.DataFrame(all_marker_rows)
    logger.info(
        "Extraction terminée : %d marqueurs extraits de %d annotations",
        len(result),
        total,
    )

    # Résumé par type de marqueur
    if not result.empty:
        logger.info(
            "Par type de marqueur : %s",
            result["marker_type"].value_counts().to_dict(),
        )

    return result


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrait les marqueurs linguistiques (mots, lemmes, ponctuations) "
            "à partir d'un fichier gold_flat XLSX ou CSV."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Chemin du fichier d'annotations (XLSX ou CSV)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "xlsx", "markers.csv"),
        help="Chemin du CSV de sortie (défaut: ../results/xlsx/markers.csv)",
    )
    parser.add_argument(
        "--no-lemma",
        action="store_true",
        help="Désactive la lemmatisation",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Inclut les lignes sans émotion active (Emo != 1)",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Filtre les mots vides (stopwords) français ultra-fréquents",
    )
    parser.add_argument(
        "--lemmatizer",
        choices=["spacy", "stanza"],
        default="spacy",
        help="Backend de lemmatisation : spacy (CPU) ou stanza (GPU CUDA requis) (défaut: spacy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille de batch pour la lemmatisation (défaut: 256)",
    )
    args = parser.parse_args()

    logger.info("=== Extraction des marqueurs linguistiques (gold_flat) ===")
    logger.info("Entrée  : %s", args.input)
    logger.info("Sortie  : %s", args.output)
    logger.info(
        "Lemmatiseur : %s",
        args.lemmatizer if not args.no_lemma else "désactivé",
    )

    # Lecture des annotations
    if not os.path.isfile(args.input):
        logger.error("Fichier d'annotations introuvable : %s", args.input)
        sys.exit(1)

    df = read_input(args.input)
    logger.info("Annotations chargées : %d lignes, %d colonnes", len(df), len(df.columns))

    # Extraction
    markers_df = build_marker_dataframe(
        df,
        use_lemma=not args.no_lemma,
        include_empty=args.include_empty,
        lemmatizer_backend=args.lemmatizer,
        batch_size=args.batch_size,
        remove_stopwords=args.remove_stopwords,
    )

    if markers_df.empty:
        logger.error("Aucun marqueur extrait. Arrêt.")
        sys.exit(1)

    # Export CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    markers_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info("Marqueurs exportés : %s (%d lignes)", args.output, len(markers_df))
    logger.info("=== Extraction terminée ===")


if __name__ == "__main__":
    main()
