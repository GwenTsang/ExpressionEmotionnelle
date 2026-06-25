#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_remarques.py — Extraction des phrases associées aux remarques "haine" et "mépris"

Ce script parcourt les annotations Glozz, identifie les unités d'annotation de type
"Autre" pour lesquelles la feature "Remarque" est "haine" ou "mépris", et extrait
les phrases complètes correspondantes à partir des fichiers de texte brut (.ac).

Les résultats sont sauvegardés dans deux fichiers texte :
    - textes_haine.txt
    - textes_mepris.txt
"""

import os
import sys
import argparse
from typing import List, Dict

import pandas as pd
import spacy

# Configuration des chemins
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

# Insertion du dossier racine dans le path pour importer le parser glozz
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from glozz.glozz_parser import process_all_corpora, CORPUS_DIRS


def clean_sentence(text: str) -> str:
    """Nettoie et formate correctement une phrase."""
    text = text.strip()
    if not text:
        return text

    # Mettre en majuscule la première lettre
    text = text[0].upper() + text[1:]

    # Corriger la fin avec deux-points
    if text.endswith(":"):
        text = text[:-1].rstrip() + "."

    # Nettoyer les tirets à la fin
    if text.endswith(("-", "–", "—")):
        text = text.rstrip("-–—").rstrip()

    # Tolérer si la ponctuation finale est avant un guillemet
    if text.endswith(("»", '"', "”", "'")) and len(text) > 1:
        if text[-2] in (".", "!", "?", "…"):
            return text
        if len(text) > 2 and text[-2] == " " and text[-3] in (".", "!", "?", "…"):
            return text

    # Ajouter un point final si ce n'est pas une ponctuation de fin
    if not text.endswith((".", "!", "?", "…")):
        text += "."

    return text


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

    def extract_sentence(self, filepath: str, start_idx: int, end_idx: int) -> str:
        """Extrait la phrase complète englobant la plage d'indices [start_idx, end_idx]."""
        doc = self._get_doc(filepath)
        
        # Parcourt les phrases du document pour trouver celle qui contient l'annotation
        for sent in doc.sents:
            if sent.start_char <= start_idx < sent.end_char or sent.start_char < end_idx <= sent.end_char:
                # Normalisation des espaces et retours à la ligne
                return " ".join(sent.text.strip().split())
        
        # En cas de problème de détection, fallback sur le span textuel brut
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        fallback_span = text[start_idx:end_idx]
        return " ".join(fallback_span.strip().split())


def main():
    parser = argparse.ArgumentParser(
        description="Extrait les phrases associées aux remarques 'haine' et 'mépris' dans le corpus Glozz."
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

    print("=== Démarrage de l'extraction des textes ===")
    
    # 1. Récupération de toutes les annotations
    df = process_all_corpora()
    if df.empty:
        print("Erreur : Aucune donnée extraite par le parser.", file=sys.stderr)
        sys.exit(1)

    # 2. Filtrage pour ne garder que le type "Autre"
    autre_df = df[df["type"] == "Autre"].copy()
    print(f"Nombre total d'unités 'Autre' : {len(autre_df)}")

    # 3. Initialisation de l'extracteur de phrases (si on n'utilise pas uniquement les spans)
    extractor = None if args.use_spans else SentenceExtractor()

    # Listes pour stocker les résultats
    phrases_haine: List[str] = []
    phrases_mepris: List[str] = []

    # 4. Traitement de chaque annotation
    for _, row in autre_df.iterrows():
        remarque = row["remarque"]
        if not remarque or not isinstance(remarque, str):
            continue

        # Séparation au cas où il y aurait plusieurs valeurs normalisées séparées par des virgules
        remarques_list = [r.strip().lower() for r in remarque.split(",")]
        
        is_haine = "haine" in remarques_list
        is_mepris = "mépris" in remarques_list or "mepris" in remarques_list

        if not (is_haine or is_mepris):
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

        # Ajout aux listes correspondantes
        if is_haine:
            phrases_haine.append(text_extracted)
        if is_mepris:
            phrases_mepris.append(text_extracted)

    # 5. Déduplication par défaut
    if not args.keep_duplicates:
        # On conserve l'ordre d'apparition
        phrases_haine = list(dict.fromkeys(phrases_haine))
        phrases_mepris = list(dict.fromkeys(phrases_mepris))

    # 6. Sauvegarde des fichiers
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_haine = os.path.join(args.output_dir, "textes_haine.txt")
    file_mepris = os.path.join(args.output_dir, "textes_mepris.txt")

    with open(file_haine, "w", encoding="utf-8") as f:
        for phrase in phrases_haine:
            f.write(phrase + "\n")

    with open(file_mepris, "w", encoding="utf-8") as f:
        for phrase in phrases_mepris:
            f.write(phrase + "\n")

    print("\n=== Rapport d'extraction ===")
    print(f"Dossier de destination : {args.output_dir}")
    print(f"Fichier haine   : {os.path.basename(file_haine)} -> {len(phrases_haine)} phrases écrites")
    print(f"Fichier mépris  : {os.path.basename(file_mepris)} -> {len(phrases_mepris)} phrases écrites")


if __name__ == "__main__":
    main()
