#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marker_extraction.py — Extraction des marqueurs linguistiques

Ce script prend en entrée le CSV d'annotations produit par glozz_parser.py
et génère un dataframe « marqueur × annotation » où chaque ligne correspond
à un marqueur (mot, lemme ou ponctuation) extrait d'un segment annoté.

Deux backends de lemmatisation sont disponibles :
  - spaCy  (CPU, batch via nlp.pipe())     — défaut
  - stanza (GPU CUDA uniquement, batch via bulk_process())

Usage :
    python marker_extraction.py                                  # spaCy par défaut
    python marker_extraction.py --lemmatizer stanza               # Stanza GPU
    python marker_extraction.py --no-lemma                        # sans lemmatisation
    python marker_extraction.py --include-empty                   # inclut features vides
    python marker_extraction.py --lemmatizer spacy --batch-size 512
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


def build_marker_dataframe(
    annotations_df: pd.DataFrame,
    use_lemma: bool = True,
    include_empty: bool = False,
    lemmatizer_backend: str = "spacy",
    batch_size: int = 256,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Construit le dataframe marqueur × annotation.

    Pour chaque annotation, extrait les mots, ponctuations, et
    optionnellement les lemmes du text_span. Chaque marqueur
    produit une ligne avec les métadonnées de l'annotation source.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Le dataframe d'annotations (sortie de glozz_parser.py).
    use_lemma : bool
        Si True, extrait aussi les lemmes.
    include_empty : bool
        Si True, inclut les unités avec Mode/Remarque vide.
    lemmatizer_backend : str
        "spacy" ou "stanza".
    batch_size : int
        Taille de batch pour la lemmatisation.

    Returns
    -------
    pd.DataFrame
        Un dataframe avec les colonnes :
        corpus, file_id, unit_id, type, start_idx, end_idx, text_span,
        mode, categorie1, categorie2, remarque,
        marker_type, marker_value
    """
    df = annotations_df.copy()

    # --- Filtrage des unités avec features vides ---
    if not include_empty:
        before = len(df)
        # Pour SitEmo : exclure si mode est vide
        mask_sitemo_valid = (df["type"] == "SitEmo") & df["mode"].notna()
        # Pour Autre : on les garde toutes (Remarque vide est acceptable)
        mask_autre = df["type"] == "Autre"
        df = df[mask_sitemo_valid | mask_autre].copy()
        dropped = before - len(df)
        if dropped > 0:
            logger.info(
                "Filtrage : %d unités SitEmo sans mode exclues (--include-empty pour les garder)",
                dropped,
            )

    # --- Filtrage des text_span vides ---
    df = df[df["text_span"].notna() & (df["text_span"].str.strip() != "")].copy()
    df = df.reset_index(drop=True)

    total = len(df)
    texts = df["text_span"].tolist()

    # --- Extraction des mots et ponctuations (rapide, pas de NLP) ---
    logger.info("Extraction des mots et ponctuations (%d annotations)...", total)
    t0 = time.time()

    all_marker_rows = []
    for i, row in df.iterrows():
        text = row["text_span"]
        base_record = {
            "corpus": row["corpus"],
            "file_id": row["file_id"],
            "unit_id": row["unit_id"],
            "type": row["type"],
            "start_idx": row["start_idx"],
            "end_idx": row["end_idx"],
            "text_span": text,
            "mode": row["mode"],
            "categorie1": row["categorie1"],
            "categorie2": row["categorie2"],
            "remarque": row["remarque"],
        }

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
            base_record = {
                "corpus": row["corpus"],
                "file_id": row["file_id"],
                "unit_id": row["unit_id"],
                "type": row["type"],
                "start_idx": row["start_idx"],
                "end_idx": row["end_idx"],
                "text_span": row["text_span"],
                "mode": row["mode"],
                "categorie1": row["categorie1"],
                "categorie2": row["categorie2"],
                "remarque": row["remarque"],
            }
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
        description="Extrait les marqueurs linguistiques (mots, lemmes, ponctuations) des annotations."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "glozz", "annotations.csv"),
        help="Chemin du CSV d'annotations (défaut: ../results/glozz/annotations.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "glozz", "markers.csv"),
        help="Chemin du CSV de sortie (défaut: ../results/glozz/markers.csv)",
    )
    parser.add_argument(
        "--no-lemma",
        action="store_true",
        help="Désactive la lemmatisation",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Inclut les unités SitEmo/Autre avec Mode/Remarque vide",
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

    logger.info("=== Extraction des marqueurs linguistiques ===")
    logger.info("Lemmatiseur : %s", args.lemmatizer if not args.no_lemma else "désactivé")

    # Lecture des annotations
    if not os.path.isfile(args.input):
        logger.error("Fichier d'annotations introuvable : %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    logger.info("Annotations chargées : %d lignes", len(df))

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

    # Export
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    markers_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info("Marqueurs exportés : %s (%d lignes)", args.output, len(markers_df))
    logger.info("=== Extraction terminée ===")


if __name__ == "__main__":
    main()
