#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extraction des marqueurs linguistiques depuis SimpleSitEmo.parquet.

Lit le fichier Parquet unifié, extrait mots / ponctuations / lemmes de
chaque ``text_span``, puis « explose » les combinaisons
(marqueur × émotion × mode) en un CSV plat prêt pour l'analyse de
spécificité.

Usage :
    python -m simplesitemo_pipeline.extract_markers
    python -m simplesitemo_pipeline.extract_markers --no-lemma
    python -m simplesitemo_pipeline.extract_markers -i data/SimpleSitEmo.parquet -o results/simplesitemo/markers.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

from .emotion_taxonomy import normalize_emotion, normalize_mode
from .nlp_utils import (
    FR_STOPWORDS,
    extract_punctuations,
    extract_words,
    get_lemmatizer,
)


# ── Chemin par défaut (relatif à ExpressionEmotionnelle/) ─────────────────

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, "data", "SimpleSitEmo.parquet")
DEFAULT_OUTPUT = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo", "markers.csv"
)


# ── Construction du dataframe de marqueurs ────────────────────────────────


def build_marker_dataframe(
    parquet_df: pd.DataFrame,
    *,
    use_lemma: bool = True,
    lemmatizer_backend: str = "spacy",
    batch_size: int = 256,
    remove_stopwords: bool = False,
) -> pd.DataFrame:
    """Construit le dataframe marqueur × émotion × mode.

    Pour chaque ligne du Parquet, extrait les marqueurs (mots, ponctuations,
    et éventuellement lemmes) puis les « explose » sur les émotions actives
    (emotion1 / emotion2 / emotion3 après normalisation et filtrage de
    ``Autre``).

    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame issu de ``SimpleSitEmo.parquet`` avec au minimum les
        colonnes ``text_span``, ``mode``, ``emotion1``, ``emotion2``,
        ``emotion3``, ``nature_linguistique``.
    use_lemma : bool
        Si True, extrait aussi les lemmes via le backend choisi.
    lemmatizer_backend : str
        ``"spacy"`` ou ``"stanza"``.
    batch_size : int
        Taille de batch pour la lemmatisation.
    remove_stopwords : bool
        Si True, filtre les stopwords français ultra-fréquents.

    Returns
    -------
    pd.DataFrame
        Colonnes : ``text_span``, ``mode``, ``emotion``, ``marker_type``,
        ``marker_value``, ``type``.
    """
    df = parquet_df.copy()

    # ── Filtrage des text_span vides ──────────────────────────────────
    before = len(df)
    df = df[df["text_span"].notna() & (df["text_span"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Filtrage : {dropped} lignes avec text_span vide exclues")

    total = len(df)
    if total == 0:
        print("Aucune ligne avec text_span valide.", file=sys.stderr)
        return pd.DataFrame()

    texts: list[str] = df["text_span"].astype(str).tolist()

    # ── Normalisation mode ────────────────────────────────────────────
    df["mode"] = df["mode"].map(normalize_mode)

    # ── Collecte des émotions par ligne (filtrage Autre — ToDo 07) ────
    emotion_cols = ["emotion1", "emotion2", "emotion3"]
    df["_emotions"] = df[emotion_cols].apply(
        lambda row: [
            e
            for val in row
            if (e := normalize_emotion(val, include_autre=False)) is not None
        ],
        axis=1,
    )

    # ── Extraction mots & ponctuations ────────────────────────────────
    print(f"Extraction des mots et ponctuations ({total} segments)…")

    all_rows: list[dict] = []

    for idx, row in df.iterrows():
        text = str(row["text_span"])
        emotions = row["_emotions"]
        mode = row["mode"]

        if not emotions:
            # Aucune émotion valide → on ignore la ligne
            continue

        # Mots et ponctuations
        for extractor, mtype in (
            (extract_words, "word"),
            (extract_punctuations, "punctuation"),
        ):
            for value in extractor(text):
                if remove_stopwords and mtype == "word" and value in FR_STOPWORDS:
                    continue
                for emo in emotions:
                    all_rows.append(
                        {
                            "text_span": text,
                            "mode": mode,
                            "emotion": emo,
                            "marker_type": mtype,
                            "marker_value": value,
                            "type": "SitEmo",
                        }
                    )

    n_words_punct = len(all_rows)
    print(f"Mots/ponctuations extraits : {n_words_punct} marqueurs")

    # ── Lemmatisation en batch ────────────────────────────────────────
    if use_lemma:
        print(
            f"Lemmatisation ({lemmatizer_backend}, batch_size={batch_size}) "
            f"de {total} textes…"
        )
        lemmatizer = get_lemmatizer(lemmatizer_backend, batch_size=batch_size)
        all_lemmas = lemmatizer.lemmatize_batch(texts)
        n_lemmas = sum(len(lst) for lst in all_lemmas)
        print(f"Lemmatisation terminée : {n_lemmas} lemmes")

        for idx, row in df.iterrows():
            emotions = row["_emotions"]
            mode = row["mode"]
            text = str(row["text_span"])

            if not emotions:
                continue

            for lemma in all_lemmas[idx]:
                if remove_stopwords and lemma in FR_STOPWORDS:
                    continue
                for emo in emotions:
                    all_rows.append(
                        {
                            "text_span": text,
                            "mode": mode,
                            "emotion": emo,
                            "marker_type": "lemma",
                            "marker_value": lemma,
                            "type": "SitEmo",
                        }
                    )

    result = pd.DataFrame(all_rows)
    print(
        f"Extraction terminée : {len(result)} marqueurs extraits de "
        f"{total} segments"
    )

    if not result.empty:
        print(
            f"Par type de marqueur : "
            f"{result['marker_type'].value_counts().to_dict()}"
        )

    return result


# ── Point d'entrée CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction des marqueurs linguistiques depuis SimpleSitEmo.parquet.",
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Chemin du Parquet d'entrée (défaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Chemin du CSV de sortie (défaut: {DEFAULT_OUTPUT})",
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
    args = parser.parse_args()

    print("=== Extraction des marqueurs SimpleSitEmo ===")
    print(f"Lemmatiseur : {args.lemmatizer if not args.no_lemma else 'désactivé'}")

    # Lecture du Parquet
    if not os.path.isfile(args.input):
        print(f"Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.input)
    print(f"Parquet chargé : {len(df)} lignes")

    # Extraction
    markers_df = build_marker_dataframe(
        df,
        use_lemma=not args.no_lemma,
        lemmatizer_backend=args.lemmatizer,
        batch_size=args.batch_size,
        remove_stopwords=args.remove_stopwords,
    )

    if markers_df.empty:
        print("Aucun marqueur extrait. Arrêt.", file=sys.stderr)
        sys.exit(1)

    # Export
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    markers_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Marqueurs exportés : {args.output} ({len(markers_df)} lignes)")

if __name__ == "__main__":
    main()
