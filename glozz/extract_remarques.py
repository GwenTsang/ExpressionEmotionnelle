#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_remarques.py — Extraction des phrases associées à une ou plusieurs remarques

Ce script utilise le parsing ciblé de remarque_parser (qui ne traite que les unités
de type "Autre") et extrait les phrases complètes via spaCy pour les catégories
de remarques demandées.

Usage :
    python -m glozz.extract_remarques haine mépris
    python -m glozz.extract_remarques amour --use-spans
    python -m glozz.extract_remarques --list

Chaque catégorie demandée produit un fichier textes_{catégorie}.txt dans le
dossier de sortie.
"""

import os
import re
import sys
import argparse
import unicodedata
from typing import List, Dict

import spacy

# Configuration des chemins
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

# Insertion du dossier racine dans le path pour importer le parser glozz
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from glozz.remarque_parser import parse_all_remarques, get_unique_remarques, CORPUS_DIRS


def _strip_accents(text: str) -> str:
    """Supprime les accents d'une chaîne pour permettre la comparaison insensible aux accents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def clean_sentence(text: str) -> str:
    """Normalise les espaces d'une phrase extraite sans altérer le contenu textuel.

    Seuls les espaces multiples, retours ligne et tabulations sont normalisés.
    Aucune modification du contenu (ponctuation, casse) n'est effectuée :
    la détection correcte des frontières de phrase est entièrement gérée
    en amont par SentenceExtractor.extract_sentence().
    """
    if not text:
        return text
    return " ".join(text.strip().split())


class SentenceExtractor:
    """Classe pour extraire les phrases complètes à partir des offsets d'annotation."""

    def __init__(self, model_name: str = "fr_core_news_sm"):
        print(f"Chargement du modèle spaCy '{model_name}'...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Erreur : Le modèle spaCy '{model_name}' n'est pas installé.", file=sys.stderr)
            print("Veuillez exécuter './setup.sh' pour l'installer.", file=sys.stderr)
            sys.exit(1)
        self._doc_cache: Dict[str, spacy.tokens.Doc] = {}

    def _get_doc(self, filepath: str) -> spacy.tokens.Doc:
        """Charge et met en cache le document spaCy pour un fichier donné."""
        if filepath not in self._doc_cache:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Fichier introuvable : {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            self._doc_cache[filepath] = self.nlp(text)
        return self._doc_cache[filepath]

    # Regex partagée : ponctuation de fin de phrase, éventuellement suivie de guillemets fermants
    _SENTENCE_END_RE = re.compile(r'[.!?…]["»\u201d\u2019]*\s*$')

    def _build_repaired_sentences(self, doc) -> list:
        """Construit les frontières de phrases réparées à partir d'un doc spaCy.

        Fusionne les segments mal découpés par spaCy (guillemets collés, etc.)
        en se basant sur la présence effective d'une ponctuation de fin de phrase.
        Le résultat est mis en cache par fichier.
        """
        cache_key = id(doc)
        if hasattr(self, '_sents_cache') and cache_key in self._sents_cache:
            return self._sents_cache[cache_key]

        raw_text = doc.text
        repaired = []
        current_start = -1
        current_end = -1

        for sent in doc.sents:
            if current_start == -1:
                current_start = sent.start_char
                current_end = sent.end_char
                continue

            prev_text = raw_text[current_start:current_end].strip()
            ends_properly = self._SENTENCE_END_RE.search(prev_text) is not None

            if not ends_properly or sent.start_char == current_end:
                current_end = sent.end_char
            else:
                repaired.append((current_start, current_end))
                current_start = sent.start_char
                current_end = sent.end_char

        if current_start != -1:
            repaired.append((current_start, current_end))

        if not hasattr(self, '_sents_cache'):
            self._sents_cache = {}
        self._sents_cache[cache_key] = repaired
        return repaired

    def extract_sentence(self, filepath: str, start_idx: int, end_idx: int) -> str:
        """Extrait la phrase complète englobant la plage d'indices [start_idx, end_idx].

        Les frontières de phrase sont déterminées rigoureusement :
        - Vers l'arrière : on recule jusqu'à trouver un début de phrase valide
          (début de texte, ou précédé d'une ponctuation de fin de phrase).
        - Vers l'avant : on avance jusqu'à trouver une ponctuation de fin de
          phrase (. ! ? …), éventuellement suivie de guillemets fermants.

        Aucune altération du texte original n'est effectuée.
        """
        doc = self._get_doc(filepath)
        raw_text = doc.text

        # 1. Construire (ou récupérer du cache) les phrases réparées
        repaired_sents = self._build_repaired_sentences(doc)

        # 2. Trouver les indices des phrases qui chevauchent l'annotation
        first_overlap = None
        last_overlap = None
        for i, (s_start, s_end) in enumerate(repaired_sents):
            if s_end > start_idx and s_start < end_idx:
                if first_overlap is None:
                    first_overlap = i
                last_overlap = i

        if first_overlap is None:
            # Fallback sur le span brut
            return " ".join(raw_text[start_idx:end_idx].strip().split())

        max_extend = 5  # limite de sécurité (dans chaque direction)

        # 3. Étendre vers l'arrière si le début ne correspond pas à une frontière de phrase
        start_i = first_overlap
        for _ in range(max_extend):
            sent_start = repaired_sents[start_i][0]
            if sent_start == 0:
                break  # début du texte = frontière naturelle
            # Vérifier que le texte précédent se termine par une ponctuation de fin
            preceding = raw_text[:sent_start].rstrip()
            if self._SENTENCE_END_RE.search(preceding):
                break  # frontière valide
            if start_i > 0:
                start_i -= 1
            else:
                break

        # 4. Étendre vers l'avant si le texte ne se termine pas par une ponctuation de fin
        end_i = last_overlap
        for _ in range(max_extend):
            candidate = raw_text[repaired_sents[start_i][0]:repaired_sents[end_i][1]].rstrip()
            if self._SENTENCE_END_RE.search(candidate):
                break
            if end_i + 1 < len(repaired_sents):
                end_i += 1
            else:
                break  # fin du texte atteinte

        # 5. Construire le résultat
        result_start = repaired_sents[start_i][0]
        result_end = repaired_sents[end_i][1]
        result = raw_text[result_start:result_end]
        return " ".join(result.strip().split())


def _sanitize_filename(category: str) -> str:
    """Convertit une catégorie en nom de fichier sûr (sans accents, minuscule)."""
    return _strip_accents(category).lower().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Extrait les phrases associées à une ou plusieurs catégories de remarques dans le corpus Glozz.",
        epilog="Exemples :\n"
               "  python -m glozz.extract_remarques haine mépris\n"
               "  python -m glozz.extract_remarques amour --use-spans\n"
               "  python -m glozz.extract_remarques --list\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "categories",
        nargs="*",
        metavar="CATÉGORIE",
        help="Catégorie(s) de remarque à extraire (ex: haine, mépris, amour). "
             "La comparaison est insensible aux accents.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_remarques",
        help="Affiche les catégories de remarques disponibles dans le corpus et quitte.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(_PROJECT_ROOT, "results", "glozz"),
        help="Dossier de sortie pour les fichiers .txt (par défaut: results/glozz/)",
    )
    parser.add_argument(
        "--keep-duplicates",
        "-k",
        action="store_true",
        help="Si spécifié, conserve les doublons (par défaut, les phrases sont dédupliquées).",
    )
    parser.add_argument(
        "--use-spans",
        "-s",
        action="store_true",
        help="Si spécifié, extrait uniquement les segments annotés (spans) au lieu des phrases entières.",
    )
    args = parser.parse_args()

    # 1. Parsing ciblé : uniquement les unités "Autre" (remarques déjà normalisées)
    df = parse_all_remarques()
    if df.empty:
        print("Erreur : Aucune donnée extraite par le parser.", file=sys.stderr)
        sys.exit(1)

    # Mode --list : afficher les remarques disponibles et quitter
    if args.list_remarques:
        unique = get_unique_remarques(df)
        print(f"Catégories de remarques disponibles ({len(unique)}) :")
        for val in unique:
            count = df["remarque"].str.contains(val, case=False, na=False).sum()
            print(f"  - {val}  ({count} occurrence{'s' if count > 1 else ''})")
        sys.exit(0)

    # Validation : au moins une catégorie requise
    if not args.categories:
        parser.error(
            "Veuillez spécifier au moins une catégorie de remarque, "
            "ou utilisez --list pour voir les catégories disponibles."
        )

    categories = [c.strip().lower() for c in args.categories]

    print("=== Démarrage de l'extraction des textes ===")
    print(f"Catégories demandées : {', '.join(categories)}")
    print(f"Nombre total d'unités 'Autre' : {len(df)}")

    # Table de correspondance sans accents pour chaque catégorie demandée
    categories_stripped = {cat: _strip_accents(cat) for cat in categories}

    # 2. Initialisation de l'extracteur de phrases (si on n'utilise pas uniquement les spans)
    extractor = None if args.use_spans else SentenceExtractor()

    # Dictionnaire {catégorie: [phrases]}
    results: Dict[str, List[str]] = {cat: [] for cat in categories}

    # 3. Traitement de chaque annotation
    for _, row in df.iterrows():
        remarque = row["remarque"]
        if not remarque or not isinstance(remarque, str):
            continue

        # Valeurs de remarque de cette annotation (déjà normalisées par remarque_parser)
        row_remarques = [r.strip().lower() for r in remarque.split(",")]
        row_remarques_stripped = [_strip_accents(r) for r in row_remarques]

        # Vérifier quelles catégories demandées sont présentes dans cette annotation
        matched_categories = [
            cat for cat, cat_stripped in categories_stripped.items()
            if cat_stripped in row_remarques_stripped
        ]

        if not matched_categories:
            continue

        # Extraction du texte (span brut ou phrase complète via spaCy)
        if args.use_spans:
            text_extracted = " ".join(row["text_span"].strip().split())
        else:
            corpus = row["corpus"]
            file_id = row["file_id"]
            start_idx = int(row["start_idx"])
            end_idx = int(row["end_idx"])

            ac_filepath = os.path.join(CORPUS_DIRS[corpus], "ac", f"{file_id}.ac")
            
            try:
                text_extracted = extractor.extract_sentence(ac_filepath, start_idx, end_idx)
            except Exception as e:
                print(f"Avertissement : Impossible d'extraire la phrase pour {file_id} [{start_idx}:{end_idx}] : {e}", file=sys.stderr)
                # Fallback sur le span
                text_extracted = " ".join(row["text_span"].strip().split())

        text_extracted = clean_sentence(text_extracted)

        # Ajout aux catégories correspondantes
        for cat in matched_categories:
            results[cat].append(text_extracted)

    # 4. Déduplication par défaut
    if not args.keep_duplicates:
        for cat in results:
            results[cat] = list(dict.fromkeys(results[cat]))

    # 5. Sauvegarde des fichiers
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Rapport d'extraction ===")
    print(f"Dossier de destination : {args.output_dir}")

    for cat in categories:
        phrases = results[cat]
        filename = f"textes_{_sanitize_filename(cat)}.txt"
        filepath = os.path.join(args.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for phrase in phrases:
                f.write(phrase + "\n")

        print(f"  {cat:<20s} -> {filename}  ({len(phrases)} phrase{'s' if len(phrases) != 1 else ''})")


if __name__ == "__main__":
    main()

