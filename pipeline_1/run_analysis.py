#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrateur de l'analyse SimpleSitEmo.

Enchaîne les étapes :
  1. Extraction des marqueurs → ``results/simplesitemo/markers.csv``
  2. Calcul de spécificité    → ``results/simplesitemo/specificity_results/``

Présuppose l'existence du fichier ``SimpleSitEmo.parquet``
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .extract_markers import build_marker_dataframe
from .marker_contract import validate_normalized_markers
from .marker_specificity import (
    EMOTIONS,
    MODES,
    compute_conditional_entropy,
    compute_entropy_by_mode,
    test_hypothesis,
)

# ── Chemins par défaut ────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, "data", "SimpleSitEmo.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "simplesitemo")


# ── Étape 1 : extraction des marqueurs ───────────────────────────────────


def step_extract_markers(
    input_path: str,
    output_dir: str,
    *,
    use_lemma: bool = True,
    lemmatizer_backend: str = "spacy",
    batch_size: int = 256,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Extrait les marqueurs de SimpleSitEmo.parquet et exporte le CSV.

    Returns
    -------
    pd.DataFrame
        Le dataframe de marqueurs (prêt pour l'étape 2).
    """
    print("ÉTAPE 1 — Extraction des marqueurs")

    if not os.path.isfile(input_path):
        print(f"Fichier introuvable : {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(input_path)
    print(f"Parquet chargé : {len(df)} lignes")

    markers_df = build_marker_dataframe(
        df,
        use_lemma=use_lemma,
        lemmatizer_backend=lemmatizer_backend,
        batch_size=batch_size,
        remove_stopwords=remove_stopwords,
    )

    if markers_df.empty:
        print("Aucun marqueur extrait. Arrêt.", file=sys.stderr)
        sys.exit(1)

    markers_path = os.path.join(output_dir, "markers.csv")
    os.makedirs(output_dir, exist_ok=True)
    markers_df.to_csv(markers_path, index=False, encoding="utf-8-sig")
    print(f"Marqueurs exportés : {markers_path} ({len(markers_df)} lignes)")

    return markers_df


# ── Étape 2 : calcul de spécificité ──────────────────────────────────────


def step_specificity(
    markers_df: pd.DataFrame,
    output_dir: str,
    *,
    min_freq: int = 3,
) -> None:
    """Calcule les entropies et le test d'hypothèse.

    Parameters
    ----------
    markers_df : pd.DataFrame
        Marqueurs normalisés (colonnes : marker_value, marker_type, type,
        emotion, mode).
    output_dir : str
        Dossier de sortie pour les résultats de spécificité.
    min_freq : int
        Fréquence minimale d'un marqueur pour les calculs.
    """
    print()
    print("ÉTAPE 2 — Calcul de spécificité")

    validate_normalized_markers(markers_df, table_name="markers for specificity")

    spec_dir = os.path.join(output_dir, "specificity_results")
    os.makedirs(spec_dir, exist_ok=True)

    # 2a. H(Emotion | Marqueur)
    print("--- Calcul H(Emotion | Marqueur) ---")
    entropy_emotion = compute_conditional_entropy(
        markers_df, "emotion", EMOTIONS, min_freq=min_freq
    )
    if not entropy_emotion.empty:
        path = os.path.join(spec_dir, "entropy_per_marker_emotion.csv")
        entropy_emotion.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Exporté : {path}")

    # 2b. H(Mode | Marqueur)
    print("--- Calcul H(Mode | Marqueur) ---")
    entropy_mode = compute_conditional_entropy(
        markers_df, "mode", MODES, min_freq=min_freq
    )
    if not entropy_mode.empty:
        path = os.path.join(spec_dir, "entropy_per_marker_mode.csv")
        entropy_mode.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Exporté : {path}")

    # 2c. Entropie moyenne par mode
    print("--- Entropie moyenne par mode ---")
    entropy_by_mode_df = compute_entropy_by_mode(markers_df, entropy_emotion)
    if not entropy_by_mode_df.empty:
        path = os.path.join(spec_dir, "entropy_by_mode_summary.csv")
        entropy_by_mode_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Exporté : {path}")

    # 2d. Test d'hypothèse
    print("--- Test de l'hypothèse ---")
    report = test_hypothesis(markers_df, entropy_emotion)
    report_path = os.path.join(spec_dir, "hypothesis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport exporté : {report_path}")
    print("\n" + report)

    print("=== Calcul de spécificité terminé ===")


# ── Point d'entrée CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrateur d'analyse SimpleSitEmo (marqueurs → spécificité).",
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--step",
        choices=["markers", "specificity", "all"],
        default="all",
        help="Étape à exécuter (défaut: all)",
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
        help="Backend de lemmatisation (défaut: spacy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille de batch pour la lemmatisation (défaut: 256)",
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
        help="Fréquence minimale d'un marqueur pour la spécificité (défaut: 3)",
    )
    args = parser.parse_args()

    print("ANALYSE SIMPLESITEMO — PIPELINE COMPLET")
    markers_df: pd.DataFrame | None = None

    # ── Étape 1 ───────────────────────────────────────────────────────
    if args.step in ("markers", "all"):
        markers_df = step_extract_markers(
            args.input,
            args.output_dir,
            use_lemma=not args.no_lemma,
            lemmatizer_backend=args.lemmatizer,
            batch_size=args.batch_size,
            remove_stopwords=args.remove_stopwords,
        )

    # ── Étape 2 ───────────────────────────────────────────────────────
    if args.step in ("specificity", "all"):
        if markers_df is None:
            # Charger depuis le CSV produit à l'étape 1
            markers_path = os.path.join(args.output_dir, "markers.csv")
            if not os.path.isfile(markers_path):
                print(
                    f"Fichier de marqueurs introuvable : {markers_path}\n"
                    "Exécutez d'abord --step markers.",
                    file=sys.stderr,
                )
                sys.exit(1)
            markers_df = pd.read_csv(markers_path, encoding="utf-8-sig")
            print(f"Marqueurs chargés depuis : {markers_path} ({len(markers_df)} lignes)")

        step_specificity(
            markers_df,
            args.output_dir,
            min_freq=args.min_freq,
        )

if __name__ == "__main__":
    main()
