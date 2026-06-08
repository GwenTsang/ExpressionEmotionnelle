#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construction de la table token-level forme/lemme.

Produit une table CSV qui conserve la relation entre chaque forme de
surface et ses lemmes (spaCy + Stanza), avec POS tags, statut stopword
et raison d'exclusion éventuelle.

Usage
-----
::

    python -m pipeline_2.build_token_lemma_table \\
        --input data/pipeline_2/SimpleSitEmo.parquet \\
        --output results/pipeline_2_granularity/token_lemmas.csv \\
        --batch-size 256

Colonnes produites
------------------
source_file, text_span, mode, emotion,
token_index, char_start, char_end,
surface, surface_lower,
lemma_spacy, pos_spacy,
lemma_stanza, pos_stanza,
is_alpha, surface_is_stopword, lemma_spacy_is_stopword, lemma_stanza_is_stopword,
kept_for_analysis, drop_reason
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

from .emotion_taxonomy import normalize_emotion, normalize_mode
from .nlp_utils import FR_STOPWORDS


# ── Chemin par défaut ─────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, "data", "pipeline_2", "SimpleSitEmo.parquet")
DEFAULT_OUTPUT = os.path.join(
    _PROJECT_ROOT, "results", "pipeline_2_granularity", "token_lemmas.csv"
)


# ── Traitement spaCy ─────────────────────────────────────────────────────


def _process_spacy(texts: list[str], batch_size: int) -> list[list[dict]]:
    """Traite les textes avec spaCy et retourne les tokens avec lemme + POS.

    Returns
    -------
    list[list[dict]]
        Pour chaque texte, liste de dicts avec clés :
        surface, lemma_spacy, pos_spacy, token_index, char_start, char_end
    """
    import spacy

    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    print(f"  spaCy chargé (fr_core_news_sm, batch_size={batch_size})")

    all_results: list[list[dict]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            tokens.append(
                {
                    "surface": token.text,
                    "surface_lower": token.text.lower(),
                    "lemma_spacy": token.lemma_.lower(),
                    "pos_spacy": token.pos_,
                    "token_index": token.i,
                    "char_start": token.idx,
                    "char_end": token.idx + len(token.text),
                }
            )
        all_results.append(tokens)

    return all_results


# ── Traitement Stanza ─────────────────────────────────────────────────────


def _process_stanza(texts: list[str], batch_size: int) -> list[list[dict]]:
    """Traite les textes avec Stanza et retourne les tokens avec lemme + POS.

    Returns
    -------
    list[list[dict]]
        Pour chaque texte, liste de dicts avec clés :
        surface, lemma_stanza, pos_stanza, start_char, end_char
    """
    import stanza

    # Configurer les threads CPU avant chargement
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    nlp = stanza.Pipeline(
        "fr",
        processors="tokenize,pos,lemma",
        use_gpu=False,
        verbose=False,
    )
    print(f"  Stanza chargé (fr, tokenize+pos+lemma, batch_size={batch_size})")

    all_results: list[list[dict]] = []

    # Traitement par batch
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        docs = [stanza.Document([], text=t) for t in batch]

        if hasattr(nlp, "bulk_process"):
            processed = nlp.bulk_process(docs)
        else:
            processed = [nlp(d) for d in docs]

        for doc in processed:
            tokens = []
            for sent in doc.sentences:
                for word in sent.words:
                    if not word.text.isalpha():
                        continue
                    tokens.append(
                        {
                            "surface_stanza": word.text,
                            "lemma_stanza": word.lemma.lower() if word.lemma else None,
                            "pos_stanza": word.upos,
                            "start_char": word.start_char,
                            "end_char": word.end_char,
                        }
                    )
            all_results.append(tokens)

    return all_results


# ── Alignement spaCy ↔ Stanza ────────────────────────────────────────────


def _align_stanza_to_spacy(
    spacy_tokens: list[dict],
    stanza_tokens: list[dict],
) -> list[dict]:
    """Aligne les tokens Stanza sur les tokens spaCy par chevauchement de positions.

    Pour chaque token spaCy, cherche le token Stanza dont la plage de
    caractères chevauche. En cas de correspondance 1-à-1, remplit
    ``lemma_stanza`` et ``pos_stanza``. Sinon, laisse NaN.

    Returns
    -------
    list[dict]
        Tokens spaCy enrichis avec ``lemma_stanza`` et ``pos_stanza``.
    """
    enriched = []
    for sp in spacy_tokens:
        sp_start = sp["char_start"]
        sp_end = sp["char_end"]

        # Chercher les tokens Stanza qui chevauchent cette plage
        matches = []
        for st in stanza_tokens:
            st_start = st.get("start_char")
            st_end = st.get("end_char")
            if st_start is None or st_end is None:
                continue
            # Chevauchement : les intervalles [sp_start, sp_end) et
            # [st_start, st_end) se recoupent
            if sp_start < st_end and st_start < sp_end:
                matches.append(st)

        row = dict(sp)  # copie les champs spaCy

        if len(matches) == 1:
            # Alignement 1-à-1
            row["lemma_stanza"] = matches[0]["lemma_stanza"]
            row["pos_stanza"] = matches[0]["pos_stanza"]
        else:
            # Pas de match ou multi-match → NaN
            row["lemma_stanza"] = None
            row["pos_stanza"] = None

        enriched.append(row)

    return enriched


# ── Détermination du statut de filtrage ───────────────────────────────────


def _compute_filter_status(token: dict) -> dict:
    """Ajoute les colonnes de statut de filtrage à un token.

    Colonnes ajoutées :
    - is_alpha (toujours True car pré-filtré)
    - surface_is_stopword
    - lemma_spacy_is_stopword
    - lemma_stanza_is_stopword
    - kept_for_analysis
    - drop_reason
    """
    surface_lower = token["surface_lower"]
    lemma_spacy = token.get("lemma_spacy")
    lemma_stanza = token.get("lemma_stanza")

    surface_stop = surface_lower in FR_STOPWORDS
    lemma_sp_stop = lemma_spacy in FR_STOPWORDS if lemma_spacy else False
    lemma_st_stop = lemma_stanza in FR_STOPWORDS if lemma_stanza else False

    # Conservé si la surface ET le lemme spaCy ne sont pas des stopwords
    kept = not surface_stop and not lemma_sp_stop

    drop_reason = ""
    if surface_stop:
        drop_reason = "stopword_surface"
    elif lemma_sp_stop:
        drop_reason = "stopword_lemma_spacy"

    token["is_alpha"] = True
    token["surface_is_stopword"] = surface_stop
    token["lemma_spacy_is_stopword"] = lemma_sp_stop
    token["lemma_stanza_is_stopword"] = lemma_st_stop
    token["kept_for_analysis"] = kept
    token["drop_reason"] = drop_reason

    return token


# ── Construction de la table complète ─────────────────────────────────────


def build_token_lemma_table(
    parquet_df: pd.DataFrame,
    *,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Construit la table token-level avec relations forme/lemme.

    Parameters
    ----------
    parquet_df : pd.DataFrame
        DataFrame issu de ``SimpleSitEmo.parquet``.
    batch_size : int
        Taille de batch pour les backends NLP.

    Returns
    -------
    pd.DataFrame
        Table token-level avec colonnes documentées dans le docstring du module.
    """
    df = parquet_df.copy()

    # ── Filtrage des text_span vides ──────────────────────────────────
    before = len(df)
    df = df[df["text_span"].notna() & (df["text_span"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Filtrage : {dropped} lignes avec text_span vide exclues")

    total = len(df)
    if total == 0:
        print("  Aucune ligne avec text_span valide.", file=sys.stderr)
        return pd.DataFrame()

    # ── Normalisation mode ────────────────────────────────────────────
    df["mode"] = df["mode"].map(normalize_mode)

    # ── Collecte des émotions par ligne ───────────────────────────────
    emotion_cols = ["emotion1", "emotion2", "emotion3"]
    df["_emotions"] = df[emotion_cols].apply(
        lambda row: [
            e
            for val in row
            if (e := normalize_emotion(val, include_autre=False)) is not None
        ],
        axis=1,
    )

    texts: list[str] = df["text_span"].astype(str).tolist()

    # ── Traitement spaCy ──────────────────────────────────────────────
    print(f"Étape 1/4 : Traitement spaCy ({total} segments)…")
    spacy_results = _process_spacy(texts, batch_size)
    n_spacy = sum(len(toks) for toks in spacy_results)
    print(f"  → {n_spacy} tokens alphabétiques extraits")

    # ── Traitement Stanza ─────────────────────────────────────────────
    print(f"Étape 2/4 : Traitement Stanza ({total} segments)…")
    stanza_results = _process_stanza(texts, batch_size)
    n_stanza = sum(len(toks) for toks in stanza_results)
    print(f"  → {n_stanza} tokens alphabétiques extraits")

    # ── Alignement et construction des lignes ─────────────────────────
    print("Étape 3/4 : Alignement spaCy ↔ Stanza et calcul des statuts…")
    all_rows: list[dict] = []
    n_aligned = 0
    n_unaligned = 0

    for idx in range(total):
        row = df.iloc[idx]
        emotions = row["_emotions"]
        mode = row["mode"]
        text = str(row["text_span"])
        source_file = str(row.get("source_file", "unknown"))

        if not emotions:
            continue

        # Alignement
        spacy_toks = spacy_results[idx]
        stanza_toks = stanza_results[idx]
        aligned_toks = _align_stanza_to_spacy(spacy_toks, stanza_toks)

        for tok in aligned_toks:
            # Calculer statut de filtrage
            tok = _compute_filter_status(tok)

            if tok.get("lemma_stanza") is not None:
                n_aligned += 1
            else:
                n_unaligned += 1

            # Exploser les émotions
            for emo in emotions:
                all_rows.append(
                    {
                        "source_file": source_file,
                        "text_span": text,
                        "mode": mode,
                        "emotion": emo,
                        "token_index": tok["token_index"],
                        "char_start": tok["char_start"],
                        "char_end": tok["char_end"],
                        "surface": tok["surface"],
                        "surface_lower": tok["surface_lower"],
                        "lemma_spacy": tok["lemma_spacy"],
                        "pos_spacy": tok["pos_spacy"],
                        "lemma_stanza": tok.get("lemma_stanza"),
                        "pos_stanza": tok.get("pos_stanza"),
                        "is_alpha": tok["is_alpha"],
                        "surface_is_stopword": tok["surface_is_stopword"],
                        "lemma_spacy_is_stopword": tok["lemma_spacy_is_stopword"],
                        "lemma_stanza_is_stopword": tok["lemma_stanza_is_stopword"],
                        "kept_for_analysis": tok["kept_for_analysis"],
                        "drop_reason": tok["drop_reason"],
                    }
                )

        if (idx + 1) % 1000 == 0:
            print(f"  … {idx + 1}/{total} segments traités")

    print(f"  Alignement : {n_aligned} tokens alignés, {n_unaligned} non alignés")

    # ── Construction du DataFrame ─────────────────────────────────────
    print("Étape 4/4 : Construction du DataFrame final…")
    result = pd.DataFrame(all_rows)

    if result.empty:
        print("  Aucun token extrait.", file=sys.stderr)
        return result

    # Statistiques
    n_kept = result["kept_for_analysis"].sum()
    n_dropped = len(result) - n_kept
    print(f"  Total : {len(result)} lignes (tokens × émotions)")
    print(f"  Conservés pour analyse : {n_kept}")
    print(f"  Exclus : {n_dropped}")

    if n_dropped > 0:
        reasons = result[~result["kept_for_analysis"]]["drop_reason"].value_counts()
        for reason, count in reasons.items():
            print(f"    {reason} : {count}")

    return result


# ── Point d'entrée CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construction de la table token-level forme/lemme "
            "(spaCy + Stanza) avec statut de filtrage."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help=f"Fichier Parquet d'entrée (défaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Fichier CSV de sortie (défaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille de batch pour les backends NLP (défaut: 256)",
    )
    args = parser.parse_args()

    print("")
    print("Construction de la table token-level forme/lemme")
    print("")

    # Lecture du Parquet
    if not os.path.isfile(args.input):
        print(f"Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.input)
    print(f"Parquet chargé : {len(df)} lignes")

    # Construction
    result = build_token_lemma_table(df, batch_size=args.batch_size)

    if result.empty:
        print("Aucun résultat. Arrêt.", file=sys.stderr)
        sys.exit(1)

    # Export
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nTable exportée : {args.output} ({len(result)} lignes)")
    print("")


if __name__ == "__main__":
    main()
