#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_analysis_xlsx.py — Orchestrateur pour le pipeline d'analyse des données XLSX

Ce script :
  1. Lit tous les fichiers XLSX du dossier data/raw/xlsx/
  2. Extrait les marqueurs avec extract_markers_xlsx.py
  3. Formate le résultat pour le rendre compatible avec le module de spécificité
  4. Calcule les scores d'entropie avec marker_specificity.py

Usage :
    python run_analysis_xlsx.py
    python run_analysis_xlsx.py --min-freq 5
"""

import os
import sys
import argparse
import logging
import glob

import pandas as pd
import numpy as np

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import des modules locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_markers_xlsx import read_input, build_marker_dataframe
from marker_specificity import (
    compute_conditional_entropy,
    compute_entropy_by_mode,
    test_hypothesis,
    EMOTIONS,
    MODES,
)

# Mappings pour convertir les colonnes XLSX en format Glozz
EMOTION_MAP = {
    "Colère": "Colere", "Dégoût": "Degout", "Joie": "Joie", "Peur": "Peur", "Surprise": "Surprise", "Tristesse": "Tristesse",
    "Admiration": "Admiration", "Culpabilité": "Culpabilite", "Embarras": "Embarras", "Fierté": "Fierte", "Jalousie": "Jalousie",
    "Autre": "Autre"
}

MODE_MAP = {
    "Comportementale": "Comportementale",
    "Désignée": "Designee",
    "Montrée": "Montree",
    "Suggérée": "Suggeree"
}


def process_all_xlsx(
    input_dir: str,
    output_dir: str,
    use_lemma: bool = True,
    lemmatizer_backend: str = "spacy",
) -> pd.DataFrame:
    """Étape 1 : Lit et extrait les marqueurs de tous les XLSX."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 1 : Extraction des marqueurs (XLSX)")
    logger.info("=" * 60)

    xlsx_files = glob.glob(os.path.join(input_dir, "*.xlsx"))
    if not xlsx_files:
        logger.error(f"Aucun fichier XLSX trouvé dans {input_dir}")
        sys.exit(1)

    all_markers = []
    
    for file in xlsx_files:
        logger.info(f"Traitement de {os.path.basename(file)}...")
        df_in = read_input(file)
        # On inclut les lignes sans émotion pour éviter de les perdre, 
        # mais la conversion Glozz les filtrera si besoin.
        df_markers = build_marker_dataframe(
            df_in,
            use_lemma=use_lemma,
            include_empty=True,
            lemmatizer_backend=lemmatizer_backend,
        )
        if not df_markers.empty:
            df_markers["source_file"] = os.path.basename(file)
            all_markers.append(df_markers)
            
    if not all_markers:
        logger.error("Aucun marqueur extrait de l'ensemble des fichiers.")
        sys.exit(1)
        
    combined_markers = pd.concat(all_markers, ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "markers.csv")
    combined_markers.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Total des marqueurs extraits : {len(combined_markers)} -> exportés dans {out_csv}")
    
    return combined_markers


def adapt_to_glozz_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit le format binaire XLSX (Colonnes d'émotions) en format catégorie Glozz."""
    logger.info("Adaptation du format pour compatibilité avec le calcul d'entropie...")
    
    # Identifier les émotions et modes actifs pour chaque ligne
    def get_active_emotions(row):
        active = [EMOTION_MAP[col] for col in EMOTION_MAP.keys() if col in row and row[col] == 1]
        return active if active else ["Autre"]
        
    def get_active_modes(row):
        active = [MODE_MAP[col] for col in MODE_MAP.keys() if col in row and row[col] == 1]
        return active if active else [np.nan]

    # Appliquer sur le dataframe
    df["categorie1"] = df.apply(get_active_emotions, axis=1)
    df["mode"] = df.apply(get_active_modes, axis=1)
    
    # Explode pour avoir une ligne par combinaison Emotion x Mode
    df_exploded = df.explode("categorie1").explode("mode")
    df_exploded["type"] = "SitEmo"
    
    return df_exploded


def run_specificity(
    markers_df: pd.DataFrame,
    output_dir: str,
    min_freq: int = 5,
):
    """Étape 2 : Calcule la spécificité."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 2 : Calcul de la spécificité (Entropie)")
    logger.info("=" * 60)

    # Convertir au format attendu par marker_specificity.py
    df_glozz_format = adapt_to_glozz_format(markers_df)

    out_spec = os.path.join(output_dir, "specificity_results")
    os.makedirs(out_spec, exist_ok=True)

    # 1. Spécificité par Émotion
    df_emo = compute_conditional_entropy(
        df_glozz_format, "categorie1", EMOTIONS, min_freq=min_freq
    )
    if not df_emo.empty:
        p1 = os.path.join(out_spec, "entropy_per_marker_emotion.csv")
        df_emo.to_csv(p1, index=False, encoding="utf-8-sig")
        logger.info("-> %s", p1)

    # 2. Spécificité par Mode
    df_mode = compute_conditional_entropy(
        df_glozz_format, "mode", MODES, min_freq=min_freq
    )
    if not df_mode.empty:
        p2 = os.path.join(out_spec, "entropy_per_marker_mode.csv")
        df_mode.to_csv(p2, index=False, encoding="utf-8-sig")
        logger.info("-> %s", p2)

    # 3. Entropie moyenne par Mode
    df_mean_mode = compute_entropy_by_mode(df_glozz_format, df_emo)
    if not df_mean_mode.empty:
        p3 = os.path.join(out_spec, "mean_entropy_by_mode.csv")
        df_mean_mode.to_csv(p3, index=False, encoding="utf-8-sig")
        logger.info("-> %s", p3)

    # 4. Test d'hypothèse
    if not df_emo.empty:
        test_hypothesis(df_glozz_format, df_emo)


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet pour les fichiers XLSX")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "xlsx"),
        help="Dossier contenant les fichiers XLSX",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "xlsx"),
        help="Dossier pour exporter les résultats",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Fréquence minimale pour le calcul d'entropie (défaut: 5)",
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
        help="Backend NLP (spacy ou stanza)",
    )
    
    args = parser.parse_args()
    
    # 1. Extraction
    combined_markers = process_all_xlsx(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_lemma=not args.no_lemma,
        lemmatizer_backend=args.lemmatizer,
    )
    
    # 2. Spécificité
    run_specificity(
        markers_df=combined_markers,
        output_dir=args.output_dir,
        min_freq=args.min_freq,
    )
    
    logger.info("=" * 60)
    logger.info("Analyse complète terminée avec succès !")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
