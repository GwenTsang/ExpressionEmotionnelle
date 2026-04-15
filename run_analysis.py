#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_analysis.py — Orchestrateur pour le pipeline d'analyse des émotions

Enchaîne les 3 étapes :
  1. Parsing Glozz (glozz_parser.py)
  2. Extraction des marqueurs (marker_extraction.py)
  3. Calcul de spécificité (marker_specificity.py)

Usage :
    python run_analysis.py                           # pipeline complet, 4 corpus
    python run_analysis.py --step parse              # parsing seul
    python run_analysis.py --step markers             # extraction des marqueurs seule
    python run_analysis.py --step specificity          # spécificité seule
    python run_analysis.py --corpus CorpusCovid       # un seul corpus
    python run_analysis.py --include-empty             # inclut features vides
    python run_analysis.py --min-freq 5               # seuil de fréquence
"""

import os
import sys
import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# On importe les modules locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glozz_parser import process_all_corpora, process_corpus, export_to_csv, CORPUS_DIRS
from marker_extraction import build_marker_dataframe
from marker_specificity import (
    compute_conditional_entropy,
    compute_entropy_by_mode,
    test_hypothesis,
    EMOTIONS,
    MODES,
)

import pandas as pd


def run_parse(
    corpus_filter: str | None = None,
    output_dir: str = "output",
) -> pd.DataFrame:
    """Étape 1 : parsing des corpus Glozz."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 1 : Parsing des corpus Glozz")
    logger.info("=" * 60)

    if corpus_filter:
        # Un seul corpus
        if corpus_filter not in CORPUS_DIRS:
            logger.error(
                "Corpus '%s' inconnu. Choix : %s",
                corpus_filter,
                list(CORPUS_DIRS.keys()),
            )
            sys.exit(1)
        df = process_corpus(CORPUS_DIRS[corpus_filter], corpus_filter)
    else:
        df = process_all_corpora()

    if df.empty:
        logger.error("Aucune annotation extraite. Arrêt.")
        sys.exit(1)

    csv_path = os.path.join(output_dir, "annotations.csv")
    export_to_csv(df, csv_path)
    return df


def run_markers(
    annotations_df: pd.DataFrame,
    output_dir: str = "output",
    use_lemma: bool = True,
    include_empty: bool = False,
    lemmatizer_backend: str = "spacy",
    batch_size: int = 256,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Étape 2 : extraction des marqueurs linguistiques."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 2 : Extraction des marqueurs linguistiques")
    logger.info("=" * 60)

    markers_df = build_marker_dataframe(
        annotations_df,
        use_lemma=use_lemma,
        include_empty=include_empty,
        lemmatizer_backend=lemmatizer_backend,
        batch_size=batch_size,
        remove_stopwords=remove_stopwords,
    )

    if markers_df.empty:
        logger.error("Aucun marqueur extrait. Arrêt.")
        sys.exit(1)

    csv_path = os.path.join(output_dir, "markers.csv")
    os.makedirs(output_dir, exist_ok=True)
    markers_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Marqueurs exportés : %s (%d lignes)", csv_path, len(markers_df))
    return markers_df


def run_specificity(
    markers_df: pd.DataFrame,
    output_dir: str = "output/specificity_results",
    min_freq: int = 3,
) -> None:
    """Étape 3 : calcul de spécificité et test d'hypothèse."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 3 : Calcul de spécificité des marqueurs")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Entropie sur les émotions
    entropy_emotion = compute_conditional_entropy(
        markers_df, "categorie1", EMOTIONS, min_freq=min_freq
    )
    if not entropy_emotion.empty:
        path = os.path.join(output_dir, "entropy_per_marker_emotion.csv")
        entropy_emotion.to_csv(path, index=False, encoding="utf-8-sig")

    # Entropie sur les modes
    entropy_mode = compute_conditional_entropy(
        markers_df, "mode", MODES, min_freq=min_freq
    )
    if not entropy_mode.empty:
        path = os.path.join(output_dir, "entropy_per_marker_mode.csv")
        entropy_mode.to_csv(path, index=False, encoding="utf-8-sig")

    # Résumé par mode
    entropy_by_mode = compute_entropy_by_mode(markers_df, entropy_emotion)
    if not entropy_by_mode.empty:
        path = os.path.join(output_dir, "entropy_by_mode_summary.csv")
        entropy_by_mode.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("Résumé entropie par mode :")
        for _, row in entropy_by_mode.iterrows():
            logger.info(
                "  %-20s : n=%-5d moy=%.4f  méd=%.4f  σ=%.4f",
                row["mode"],
                row["n_markers"],
                row["mean_entropy"],
                row["median_entropy"],
                row["std_entropy"],
            )

    # Test d'hypothèse
    report = test_hypothesis(markers_df, entropy_emotion)
    report_path = os.path.join(output_dir, "hypothesis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Rapport d'hypothèse : %s", report_path)
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse des marqueurs émotionnels (corpus Glozz)."
    )
    parser.add_argument(
        "--step",
        choices=["parse", "markers", "specificity", "all"],
        default="all",
        help="Étape à exécuter (défaut: all)",
    )
    parser.add_argument(
        "--corpus",
        choices=list(CORPUS_DIRS.keys()),
        default=None,
        help="Nom d'un seul corpus à traiter (défaut: tous)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Dossier de sortie principal (défaut: output/)",
    )
    parser.add_argument(
        "--no-lemma",
        action="store_true",
        help="Désactive la lemmatisation",
    )
    parser.add_argument(
        "--lemmatizer",
        choices=["spacy", "stanza"],
        default="spacy",
        help="Backend de lemmatisation : spacy (CPU batch) ou stanza (GPU CUDA requis) (défaut: spacy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille de batch pour la lemmatisation (défaut: 256)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Inclut les unités SitEmo/Autre avec Mode/Remarque vide dans l'analyse",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Filtre les mots vides (stopwords) français ultra-fréquents",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Fréquence minimale d'un marqueur pour les calculs d'entropie (défaut: 3)",
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=== PIPELINE D'ANALYSE DES MARQUEURS ÉMOTIONNELS ===")
    if args.corpus:
        logger.info("Corpus sélectionné : %s", args.corpus)
    lemmatizer_label = args.lemmatizer if not args.no_lemma else "désactivé"
    logger.info("Options : include-empty=%s, lemmatizer=%s, batch-size=%d, min-freq=%d, remove-stopwords=%s",
                args.include_empty, lemmatizer_label, args.batch_size, args.min_freq, args.remove_stopwords)

    annotations_csv = os.path.join(args.output_dir, "annotations.csv")
    markers_csv = os.path.join(args.output_dir, "markers.csv")
    specificity_dir = os.path.join(args.output_dir, "specificity_results")

    if args.step in ("parse", "all"):
        annotations_df = run_parse(args.corpus, args.output_dir)
    else:
        if not os.path.isfile(annotations_csv):
            logger.error("Fichier d'annotations requis : %s", annotations_csv)
            sys.exit(1)
        annotations_df = pd.read_csv(annotations_csv, encoding="utf-8-sig")

    if args.step in ("markers", "all"):
        markers_df = run_markers(
            annotations_df,
            args.output_dir,
            use_lemma=not args.no_lemma,
            include_empty=args.include_empty,
            lemmatizer_backend=args.lemmatizer,
            batch_size=args.batch_size,
            remove_stopwords=args.remove_stopwords,
        )
    else:
        if not os.path.isfile(markers_csv):
            logger.error("Fichier de marqueurs requis : %s", markers_csv)
            sys.exit(1)
        markers_df = pd.read_csv(markers_csv, encoding="utf-8-sig")

    if args.step in ("specificity", "all"):
        run_specificity(markers_df, specificity_dir, min_freq=args.min_freq)

    elapsed = time.time() - start_time
    logger.info("=== PIPELINE TERMINÉ en %.1f secondes ===", elapsed)


if __name__ == "__main__":
    main()
