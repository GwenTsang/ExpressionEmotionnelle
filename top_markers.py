#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
top_markers.py — Extraction des meilleurs marqueurs par catégorie / mode

Ce script lit les fichiers d'entropie (par émotion et par mode) générés par 
marker_specificity.py et extrait les 20 meilleurs marqueurs pour chaque classe.
Un "meilleur" marqueur est défini comme ayant la plus forte probabilité 
conditionnelle P(Classe | Marqueur), en favorisant ceux qui apparaissent le plus souvent.

Usage :
    python top_markers.py
    python top_markers.py -e output/specificity_results/entropy_per_marker_emotion.csv -m output/specificity_results/entropy_per_marker_mode.csv --top 20
"""

import os
import argparse
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def print_top_markers_for_file(filepath: str, title: str, top_n: int = 20, export_path: str = None):
    """Affiche (et exporte optionnellement) les top N marqueurs pour chaque classe détectée dans le fichier."""
    if not os.path.isfile(filepath):
        logger.error(f"Fichier introuvable : {filepath}")
        return

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    
    # Identifier les colonnes de probabilités P(...)
    prob_cols = [c for c in df.columns if c.startswith("P(") and c.endswith(")")]
    if not prob_cols:
        logger.error(f"Aucune colonne de probabilité P(...) trouvée dans {filepath}")
        return
    
    print("\n" + "="*80)
    print(f"Meilleurs marqueurs pour : {title.upper()}")
    print("="*80)

    # Récupérer les noms des classes (ex: P(Colere) -> Colere)
    classes = [c[2:-1] for c in prob_cols]
    
    all_top_records = []

    for cls, p_col in zip(classes, prob_cols):
        # On ne s'intéresse qu'aux marqueurs qui ont une probabilité non nulle pour cette classe
        df_cls = df[df[p_col] > 0].copy()
        
        if df_cls.empty:
            print(f"\n--- {cls} : Aucun marqueur trouvé ---")
            continue
            
        # On trie d'abord par probabilité P(Classe|Marqueur) décroissante,
        # puis par la fréquence totale du marqueur (total_count) décroissante,
        # puis par l'entropie la plus faible (pour les départager)
        df_sorted = df_cls.sort_values(by=[p_col, "total_count", "entropy"], 
                                       ascending=[False, False, True])
        
        top_n_df = df_sorted.head(top_n)
        
        print(f"\n--- Top {top_n} marqueurs pour : {cls} ---")
        print(f"{'Marqueur':<25} {'Type':<12} {'P(Classe)':<10} {'Freq_Globale':<15} {'Entropie':<10}")
        print("-" * 75)
        
        for _, row in top_n_df.iterrows():
            marker = row['marker_value']
            m_type = row['marker_type']
            prob = row[p_col]
            tot = row['total_count']
            ent = row['entropy']
            
            print(f"{marker:<25} {m_type:<12} {prob:<10.4f} {tot:<15} {ent:<10.4f}")
            
            all_top_records.append({
                "Classe": cls,
                "Marqueur": marker,
                "Type": m_type,
                "P(Classe)": prob,
                "Freq_Globale": tot,
                "Entropie": ent
            })

    if export_path:
        out_df = pd.DataFrame(all_top_records)
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        out_df.to_csv(export_path, index=False, encoding="utf-8-sig")
        logger.info(f"\nExporté : {export_path}")

def main():
    parser = argparse.ArgumentParser(description="Affiche les meilleurs marqueurs par catégorie/mode.")
    parser.add_argument(
        "--emotion", "-e",
        default="output/specificity_results/entropy_per_marker_emotion.csv",
        help="Chemin du CSV d'entropie par émotion"
    )
    parser.add_argument(
        "--mode", "-m",
        default="output/specificity_results/entropy_per_marker_mode.csv",
        help="Chemin du CSV d'entropie par mode"
    )
    parser.add_argument(
        "--outdir", "-o",
        default="output/specificity_results",
        help="Dossier pour exporter les synthèses (défaut: output/specificity_results)"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=20,
        help="Nombre de marqueurs à afficher par classe (défaut: 20)"
    )
    args = parser.parse_args()

    # Emotions
    if os.path.exists(args.emotion):
        out_emo = os.path.join(args.outdir, "top_markers_emotion.csv")
        print_top_markers_for_file(args.emotion, "ÉMOTIONS", top_n=args.top, export_path=out_emo)
    else:
        logger.warning(f"Fichier d'émotions manquant : {args.emotion}")

    # Modes
    if os.path.exists(args.mode):
        out_mode = os.path.join(args.outdir, "top_markers_mode.csv")
        print_top_markers_for_file(args.mode, "MODES D'EXPRESSION", top_n=args.top, export_path=out_mode)
    else:
        logger.warning(f"Fichier de modes manquant : {args.mode}")

if __name__ == "__main__":
    main()
