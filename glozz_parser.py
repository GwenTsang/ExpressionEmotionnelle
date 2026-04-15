#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glozz_parser.py — Parsing de corpus annotés au format Glozz (.aa/.ac)

Ce script parcourt les quatre sous-corpus, associe chaque fichier
d'annotation XML (.aa) à son fichier texte brut (.ac), et extrait
toutes les unités d'annotation de type SitEmo et Autre.

Colonnes produites :
    corpus, file_id, unit_id, type, start_idx, end_idx, text_span,
    mode, categorie1, categorie2, remarque

Usage :
    python glozz_parser.py                  # traite les 4 corpus, exporte output/annotations.csv
    python glozz_parser.py --output mon.csv # chemin de sortie personnalisé
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import logging
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Chemins absolus des quatre corpus
CORPUS_DIRS = {
    "Albert_dataset": "/home/gwen/Documents/Emotions/Albert_dataset",
    "CorpusCovid": "/home/gwen/Documents/Emotions/CorpusCovid",
    "LitteratureJeunesse": "/home/gwen/Documents/Emotions/LitteratureJeunesse",
    "PtitLibe": "/home/gwen/Documents/Emotions/PtitLibe",
}

# Types d'annotation à extraire
TARGET_TYPES = {"SitEmo", "Autre"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fonctions de parsing
# ---------------------------------------------------------------------------


def _get_feature_value(feature_node) -> Optional[str]:
    """Extrait la valeur d'un nœud <feature>.

    Le format Glozz encode les valeurs comme texte direct :
        <feature name="Mode">Suggeree</feature>
    ou vide :
        <feature name="Mode"/>

    Retourne None si le nœud est vide ou le texte est absent.
    """
    if feature_node is None:
        return None
    text = feature_node.text
    if text is None or text.strip() == "":
        return None
    return text.strip()


def parse_aa_ac_pair(
    aa_filepath: str,
    ac_filepath: str,
    corpus_name: str = "",
) -> list[dict]:
    """Parse une paire de fichiers .aa (annotations) et .ac (texte brut).

    Extrait toutes les unités de type SitEmo et Autre avec leurs
    métadonnées et le segment textuel correspondant.

    Parameters
    ----------
    aa_filepath : str
        Chemin vers le fichier d'annotation XML (.aa).
    ac_filepath : str
        Chemin vers le fichier texte brut correspondant (.ac).
    corpus_name : str
        Nom du corpus source (pour la colonne 'corpus').

    Returns
    -------
    list[dict]
        Liste de dictionnaires, un par unité extraite.
    """
    # --- Lecture du texte brut ---
    try:
        with open(ac_filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        logger.warning("Fichier .ac manquant : %s", ac_filepath)
        return []
    except Exception as e:
        logger.error("Erreur lecture .ac %s : %s", ac_filepath, e)
        return []

    # --- Parsing XML ---
    try:
        tree = ET.parse(aa_filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning("Erreur XML dans %s : %s", aa_filepath, e)
        return []
    except Exception as e:
        logger.error("Erreur lecture .aa %s : %s", aa_filepath, e)
        return []

    file_id = os.path.basename(aa_filepath).replace(".aa", "")
    extracted = []

    for unit in root.findall(".//unit"):
        # --- Identification du type ---
        type_node = unit.find("./characterisation/type")
        if type_node is None:
            continue
        unit_type = (type_node.text or "").strip()
        if unit_type not in TARGET_TYPES:
            continue

        # --- Identifiant de l'unité ---
        unit_id = unit.get("id", "")

        # --- Positions (offsets) ---
        start_node = unit.find("./positioning/start/singlePosition")
        end_node = unit.find("./positioning/end/singlePosition")
        if start_node is None or end_node is None:
            logger.debug(
                "Unité %s sans position dans %s — ignorée", unit_id, file_id
            )
            continue

        try:
            start_idx = int(start_node.get("index", "-1"))
            end_idx = int(end_node.get("index", "-1"))
        except (ValueError, TypeError):
            logger.debug(
                "Index invalide pour l'unité %s dans %s", unit_id, file_id
            )
            continue

        # --- Extraction du segment textuel ---
        if 0 <= start_idx <= end_idx <= len(raw_text):
            text_span = raw_text[start_idx:end_idx]
        else:
            logger.debug(
                "Offsets hors limites [%d:%d] pour l'unité %s dans %s "
                "(longueur texte=%d)",
                start_idx,
                end_idx,
                unit_id,
                file_id,
                len(raw_text),
            )
            text_span = ""

        # Nettoyage : remplacer les retours à la ligne par des espaces
        text_span_clean = text_span.replace("\n", " ").replace("\r", " ")

        # --- Extraction des features ---
        features = {}
        feature_set = unit.find("./characterisation/featureSet")
        if feature_set is not None:
            for feature in feature_set.findall("feature"):
                feat_name = feature.get("name", "")
                feat_value = _get_feature_value(feature)
                features[feat_name] = feat_value

        # --- Construction de l'enregistrement ---
        record = {
            "corpus": corpus_name,
            "file_id": file_id,
            "unit_id": unit_id,
            "type": unit_type,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text_span": text_span_clean,
            "mode": None,
            "categorie1": None,
            "categorie2": None,
            "remarque": None,
        }

        if unit_type == "SitEmo":
            record["mode"] = features.get("Mode")
            record["categorie1"] = features.get("Categorie")
            record["categorie2"] = features.get("Categorie2")
        elif unit_type == "Autre":
            record["remarque"] = features.get("Remarque")

        extracted.append(record)

    return extracted


def process_corpus(corpus_dir: str, corpus_name: str) -> pd.DataFrame:
    """Parcourt un corpus (dossiers aa/ et ac/) et extrait les annotations.

    Parameters
    ----------
    corpus_dir : str
        Chemin vers le dossier racine du corpus (contenant aa/ et ac/).
    corpus_name : str
        Nom du corpus pour la colonne 'corpus'.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant toutes les annotations extraites.
    """
    aa_dir = os.path.join(corpus_dir, "aa")
    ac_dir = os.path.join(corpus_dir, "ac")

    if not os.path.isdir(aa_dir):
        logger.error("Dossier aa/ introuvable : %s", aa_dir)
        return pd.DataFrame()
    if not os.path.isdir(ac_dir):
        logger.error("Dossier ac/ introuvable : %s", ac_dir)
        return pd.DataFrame()

    all_records = []
    aa_files = sorted(
        f for f in os.listdir(aa_dir) if f.endswith(".aa")
    )
    n_files = len(aa_files)
    n_missing = 0
    n_errors = 0

    logger.info("Corpus '%s' : %d fichiers .aa trouvés", corpus_name, n_files)

    for aa_filename in aa_files:
        aa_path = os.path.join(aa_dir, aa_filename)
        ac_filename = aa_filename.replace(".aa", ".ac")
        ac_path = os.path.join(ac_dir, ac_filename)

        if not os.path.isfile(ac_path):
            logger.warning("Fichier .ac manquant pour %s", aa_filename)
            n_missing += 1
            continue

        records = parse_aa_ac_pair(aa_path, ac_path, corpus_name)
        if records is None:
            n_errors += 1
            continue
        all_records.extend(records)

    logger.info(
        "Corpus '%s' : %d annotations extraites (%d fichiers manquants, "
        "%d erreurs)",
        corpus_name,
        len(all_records),
        n_missing,
        n_errors,
    )

    return pd.DataFrame(all_records)


def process_all_corpora(
    corpus_dirs: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Agrège les annotations de tous les corpus.

    Parameters
    ----------
    corpus_dirs : dict[str, str], optional
        Dictionnaire {nom_corpus: chemin}. Par défaut : CORPUS_DIRS.

    Returns
    -------
    pd.DataFrame
        DataFrame consolidé de toutes les annotations.
    """
    if corpus_dirs is None:
        corpus_dirs = CORPUS_DIRS

    all_dfs = []
    for corpus_name, corpus_dir in corpus_dirs.items():
        if not os.path.isdir(corpus_dir):
            logger.error(
                "Dossier corpus introuvable : %s (%s)",
                corpus_dir,
                corpus_name,
            )
            continue
        df = process_corpus(corpus_dir, corpus_name)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("Aucune annotation extraite de tous les corpus.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "Total : %d annotations extraites de %d corpus",
        len(result),
        len(all_dfs),
    )
    return result


def export_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Exporte le DataFrame dans un fichier CSV encodé en UTF-8.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à exporter.
    output_path : str
        Chemin du fichier de sortie.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Résultats exportés : %s (%d lignes)", output_path, len(df))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse les corpus Glozz et extrait les annotations émotionnelles."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/annotations.csv",
        help="Chemin du fichier CSV de sortie (défaut: output/annotations.csv)",
    )
    args = parser.parse_args()

    logger.info("=== Démarrage du parsing des corpus Glozz ===")
    df = process_all_corpora()

    if df.empty:
        logger.error("Aucune annotation extraite. Arrêt.")
        sys.exit(1)

    # Résumé statistique rapide
    logger.info("--- Résumé ---")
    logger.info("Lignes totales      : %d", len(df))
    logger.info("Par corpus          : %s", df["corpus"].value_counts().to_dict())
    logger.info("Par type            : %s", df["type"].value_counts().to_dict())
    if "mode" in df.columns:
        logger.info(
            "Par mode (SitEmo)   : %s",
            df.loc[df["type"] == "SitEmo", "mode"].value_counts(dropna=False).to_dict(),
        )
    if "categorie1" in df.columns:
        logger.info(
            "Par catégorie1      : %s",
            df.loc[df["type"] == "SitEmo", "categorie1"]
            .value_counts(dropna=False)
            .to_dict(),
        )

    export_to_csv(df, args.output)
    logger.info("=== Parsing terminé ===")


if __name__ == "__main__":
    main()
