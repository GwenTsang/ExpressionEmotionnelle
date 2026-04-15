#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
marker_extraction.py — Extraction des marqueurs linguistiques

Ce script prend en entrée le CSV d'annotations produit par glozz_parser.py
et génère un dataframe « marqueur × annotation » où chaque ligne correspond
à un marqueur (mot, lemme ou ponctuation) extrait d'un segment annoté.

Usage :
    python marker_extraction.py                             # défaut
    python marker_extraction.py -i output/annotations.csv   # entrée personnalisée
    python marker_extraction.py --no-lemma                  # sans lemmatisation
    python marker_extraction.py --include-empty              # inclut les unités avec Mode/Remarque vide
"""

import os
import re
import sys
import argparse
import logging
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

# Regex pour la tokenisation
RE_WORD = re.compile(r"\b[a-zA-ZÀ-ÿœŒæÆ]+(?:['-][a-zA-ZÀ-ÿœŒæÆ]+)*\b", re.UNICODE)
RE_PUNCT = re.compile(r"[!?.,;:…\-—–\"'«»()\[\]]+")


# ---------------------------------------------------------------------------
# Chargement du modèle spaCy (lazy)
# ---------------------------------------------------------------------------

_nlp = None


def _get_spacy_nlp():
    """Charge le modèle spaCy fr_core_news_sm une seule fois (lazy loading)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        _nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        logger.info("spaCy fr_core_news_sm chargé (lemmatisation activée)")
        return _nlp
    except ImportError:
        logger.warning("spaCy non installé — lemmatisation désactivée")
        return None
    except OSError:
        logger.warning(
            "Modèle fr_core_news_sm introuvable — lemmatisation désactivée"
        )
        return None


# ---------------------------------------------------------------------------
# Extraction des marqueurs
# ---------------------------------------------------------------------------


def extract_words(text: str) -> list[str]:
    """Extrait les mots (tokens alphabétiques) en minuscules."""
    if not text or not isinstance(text, str):
        return []
    return [m.lower() for m in RE_WORD.findall(text)]


def extract_punctuations(text: str) -> list[str]:
    """Extrait les signes de ponctuation individuels."""
    if not text or not isinstance(text, str):
        return []
    # On sépare chaque caractère de ponctuation individuellement
    puncts = []
    for match in RE_PUNCT.finditer(text):
        group = match.group()
        # Traiter les points de suspension comme un seul marqueur
        if "…" in group:
            puncts.append("…")
            group = group.replace("…", "")
        # Séquences de points (... → …)
        while "..." in group:
            puncts.append("…")
            group = group.replace("...", "", 1)
        # Chaque caractère de ponctuation restant
        for ch in group:
            if ch in "!?.,;:—–\"'«»()[]!-":
                puncts.append(ch)
    return puncts


def extract_lemmas(text: str, nlp) -> list[str]:
    """Extrait les lemmes via spaCy, en ne conservant que les tokens alphabétiques."""
    if nlp is None or not text or not isinstance(text, str):
        return []
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_space
    ]


def build_marker_dataframe(
    annotations_df: pd.DataFrame,
    use_lemma: bool = True,
    include_empty: bool = False,
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
        Si True, extrait aussi les lemmes via spaCy.
    include_empty : bool
        Si True, inclut les unités avec Mode/Remarque vide.

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

    # --- Chargement de spaCy si nécessaire ---
    nlp = _get_spacy_nlp() if use_lemma else None

    # --- Extraction des marqueurs ---
    all_marker_rows = []
    total = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 500 == 0:
            logger.info("Extraction des marqueurs : %d/%d", i + 1, total)

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

        # Lemmes
        if nlp is not None:
            for lemma in extract_lemmas(text, nlp):
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
        default="output/annotations.csv",
        help="Chemin du CSV d'annotations (défaut: output/annotations.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/markers.csv",
        help="Chemin du CSV de sortie (défaut: output/markers.csv)",
    )
    parser.add_argument(
        "--no-lemma",
        action="store_true",
        help="Désactive la lemmatisation (pas besoin de spaCy)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Inclut les unités SitEmo/Autre avec Mode/Remarque vide",
    )
    args = parser.parse_args()

    logger.info("=== Extraction des marqueurs linguistiques ===")

    # Lecture des annotations
    if not os.path.isfile(args.input):
        logger.error("Fichier d'annotations introuvable : %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    logger.info("Annotations chargées : %d lignes", len(df))

    # Extraction
    markers_df = build_marker_dataframe(
        df, use_lemma=not args.no_lemma, include_empty=args.include_empty
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
