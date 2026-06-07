#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcul de spécificité des marqueurs linguistiques (pipeline SimpleSitEmo).

Adapté de ``analysis_pipeline.marker_specificity`` :
- Colonne condition : « emotion » (au lieu de « categorie1 »).
- Modes accentués (Désignée, Montrée, Suggérée).
- Pas de ``normalize_existing_marker_table()`` — les marqueurs sont déjà
  normalisés par ``extract_markers.py``.

Ce script calcule :
  1. P(Emotion = e | Marqueur = x) et P(Mode = m | Marqueur = x)
  2. L'entropie de Shannon H(Emotion|x) et H(Mode|x) pour chaque marqueur
  3. Le test de l'hypothèse : les marqueurs Montrée/Suggérée sont-ils plus
     dispersés (entropie plus élevée) que Désignée/Comportementale ?

Usage :
    python -m simplesitemo_pipeline.marker_specificity
    python -m simplesitemo_pipeline.marker_specificity -i results/simplesitemo/markers.csv --min-freq 5
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .emotion_taxonomy import EMOTIONS, MODES
from .marker_contract import validate_normalized_markers

# ── Chemin par défaut (relatif à ExpressionEmotionnelle/) ─────────────────

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_INPUT = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo", "markers.csv"
)
DEFAULT_OUTDIR = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo", "specificity_results"
)


# ---------------------------------------------------------------------------
# Calcul d'entropie de Shannon
# ---------------------------------------------------------------------------


def shannon_entropy(prob_dist: np.ndarray) -> float:
    """Calcule l'entropie de Shannon H = -Σ p(x) log2(p(x)).

    Les probabilités nulles sont ignorées (0 * log(0) = 0 par convention).
    """
    p = prob_dist[prob_dist > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Calcul des probabilités conditionnelles et de l'entropie
# ---------------------------------------------------------------------------


def compute_conditional_entropy(
    markers_df: pd.DataFrame,
    condition_col: str,
    valid_values: list[str],
    marker_col: str = "marker_value",
    min_freq: int = 3,
) -> pd.DataFrame:
    """Calcule P(condition|marqueur) et H(condition|marqueur) pour chaque marqueur.

    Parameters
    ----------
    markers_df : pd.DataFrame
        Le dataframe de marqueurs (sortie de extract_markers.py).
    condition_col : str
        Colonne cible (ex: 'emotion' ou 'mode').
    valid_values : list[str]
        Liste des valeurs valides pour la condition.
    marker_col : str
        Colonne contenant les valeurs des marqueurs.
    min_freq : int
        Fréquence minimale d'un marqueur pour être inclus.

    Returns
    -------
    pd.DataFrame
        Un dataframe avec pour chaque marqueur :
        - marker_value : le marqueur
        - marker_type : type du marqueur (word/lemma/punctuation)
        - total_count : nombre total d'occurrences
        - P(val1), P(val2), ... : probabilités conditionnelles
        - entropy : entropie de Shannon
    """
    validate_normalized_markers(
        markers_df,
        table_name=f"marker table for {condition_col}",
        require_condition_columns=False,
    )
    if condition_col not in markers_df.columns:
        raise ValueError(
            f"Colonne de condition absente: {condition_col}. "
            f"Colonnes disponibles: {list(markers_df.columns)}"
        )

    # Filtrer les lignes avec des valeurs valides dans la colonne condition
    df = markers_df[markers_df[condition_col].isin(valid_values)].copy()

    if df.empty:
        print(
            f"Aucune donnée pour la condition '{condition_col}'. Vérifiez les données.",
            file=sys.stderr,
        )
        return pd.DataFrame()

    # Compter les occurrences par marqueur × type × valeur
    grouped = (
        df.groupby([marker_col, "marker_type", condition_col])
        .size()
        .reset_index(name="count")
    )

    # Pivoter pour avoir une colonne par valeur de la condition
    pivot = grouped.pivot_table(
        index=[marker_col, "marker_type"],
        columns=condition_col,
        values="count",
        fill_value=0,
    )

    # S'assurer que toutes les valeurs valides sont présentes
    for val in valid_values:
        if val not in pivot.columns:
            pivot[val] = 0

    # Fréquence totale
    pivot[valid_values] = pivot[valid_values].astype(int)
    pivot["total_count"] = pivot.sum(axis=1).astype(int)

    # Filtrer par fréquence minimale
    pivot = pivot[pivot["total_count"] >= min_freq]

    if pivot.empty:
        print(
            f"Aucun marqueur avec freq >= {min_freq} pour '{condition_col}'",
            file=sys.stderr,
        )
        return pd.DataFrame()

    # Probabilités conditionnelles
    prob_cols = []
    for val in valid_values:
        prob_col = f"P({val})"
        pivot[prob_col] = pivot[val] / pivot["total_count"]
        prob_cols.append(prob_col)

    # Entropie de Shannon
    pivot["entropy"] = pivot[prob_cols].apply(
        lambda row: shannon_entropy(row.values), axis=1
    )

    # Entropie maximale (pour normalisation)
    max_entropy = np.log2(len(valid_values))
    pivot["normalized_entropy"] = (
        pivot["entropy"] / max_entropy if max_entropy > 0 else 0.0
    )

    # Reset index
    result = pivot.reset_index()
    result = result.rename(columns={marker_col: "marker_value"})

    # Tri par entropie décroissante
    result = result.sort_values("entropy", ascending=False)

    print(
        f"Entropie calculée pour '{condition_col}' : "
        f"{len(result)} marqueurs (freq >= {min_freq})"
    )
    print(
        f"  Entropie moyenne : {result['entropy'].mean():.4f} / "
        f"max théorique : {max_entropy:.4f}"
    )

    return result


# ---------------------------------------------------------------------------
# Entropie moyenne par mode (test d'hypothèse)
# ---------------------------------------------------------------------------


def compute_entropy_by_mode(
    markers_df: pd.DataFrame,
    entropy_emotion_df: pd.DataFrame,
    marker_col: str = "marker_value",
) -> pd.DataFrame:
    """Calcule l'entropie moyenne des marqueurs pour chaque mode d'expression.

    Pour chaque mode, on identifie les marqueurs qui apparaissent dans ce mode,
    puis on calcule la moyenne de leur entropie sur les émotions.

    Parameters
    ----------
    markers_df : pd.DataFrame
        Le dataframe de marqueurs.
    entropy_emotion_df : pd.DataFrame
        Le dataframe d'entropie par marqueur (sorti de compute_conditional_entropy).

    Returns
    -------
    pd.DataFrame
        Résumé : mode, n_markers, mean_entropy, median_entropy, std_entropy
    """
    validate_normalized_markers(markers_df, table_name="markers for entropy by mode")

    # On ne garde que les SitEmo avec un mode valide
    sitemo = markers_df[
        (markers_df["type"] == "SitEmo") & markers_df["mode"].isin(MODES)
    ].copy()

    if sitemo.empty or entropy_emotion_df.empty:
        return pd.DataFrame()

    results = []
    for mode in MODES:
        # Marqueurs présents dans ce mode
        mode_markers = sitemo[sitemo["mode"] == mode][
            [marker_col, "marker_type"]
        ].drop_duplicates()

        # Jointure avec les entropies
        merged = mode_markers.merge(
            entropy_emotion_df[["marker_value", "marker_type", "entropy"]],
            left_on=[marker_col, "marker_type"],
            right_on=["marker_value", "marker_type"],
            how="inner",
        )

        if merged.empty:
            results.append({
                "mode": mode,
                "n_markers": 0,
                "mean_entropy": np.nan,
                "median_entropy": np.nan,
                "std_entropy": np.nan,
            })
            continue

        results.append({
            "mode": mode,
            "n_markers": len(merged),
            "mean_entropy": merged["entropy"].mean(),
            "median_entropy": merged["entropy"].median(),
            "std_entropy": merged["entropy"].std(),
        })

    return pd.DataFrame(results)


def test_hypothesis(
    markers_df: pd.DataFrame,
    entropy_emotion_df: pd.DataFrame,
    marker_col: str = "marker_value",
) -> str:
    """Teste l'hypothèse : les marqueurs Montrée/Suggérée sont-ils plus dispersés ?

    Renvoie un rapport textuel avec les résultats.
    """
    validate_normalized_markers(markers_df, table_name="markers for hypothesis test")

    sitemo = markers_df[
        (markers_df["type"] == "SitEmo") & markers_df["mode"].isin(MODES)
    ].copy()

    if sitemo.empty or entropy_emotion_df.empty:
        return "ERREUR : pas de données suffisantes pour tester l'hypothèse.\n"

    # Collecter les entropies par mode
    entropy_by_mode: dict[str, np.ndarray] = {}
    for mode in MODES:
        mode_markers = sitemo[sitemo["mode"] == mode][
            [marker_col, "marker_type"]
        ].drop_duplicates()
        merged = mode_markers.merge(
            entropy_emotion_df[["marker_value", "marker_type", "entropy"]],
            left_on=[marker_col, "marker_type"],
            right_on=["marker_value", "marker_type"],
            how="inner",
        )
        entropy_by_mode[mode] = merged["entropy"].values

    report_lines: list[str] = []
    report_lines.append("=" * 70)
    report_lines.append("TEST D'HYPOTHÈSE SUR LA DISPERSION DES MARQUEURS")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(
        "Hypothèse : les marqueurs des émotions Montrée et Suggérée"
    )
    report_lines.append(
        "présentent une plus grande dispersion (entropie plus élevée)"
    )
    report_lines.append(
        "que les marqueurs des émotions Désignée et Comportementale."
    )
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("STATISTIQUES DESCRIPTIVES PAR MODE")
    report_lines.append("-" * 70)

    for mode in MODES:
        vals = entropy_by_mode.get(mode, np.array([]))
        if len(vals) > 0:
            report_lines.append(
                f"  {mode:20s} : n={len(vals):5d}, "
                f"moy={np.mean(vals):.4f}, "
                f"méd={np.median(vals):.4f}, "
                f"σ={np.std(vals):.4f}"
            )
        else:
            report_lines.append(f"  {mode:20s} : aucun marqueur")

    report_lines.append("")

    # --- Test de Kruskal-Wallis (comparaison globale des 4 modes) ---
    groups = [
        entropy_by_mode[m]
        for m in MODES
        if len(entropy_by_mode.get(m, [])) > 0
    ]
    if len(groups) >= 2:
        stat_kw, p_kw = stats.kruskal(*groups)
        report_lines.append("-" * 70)
        report_lines.append("TEST DE KRUSKAL-WALLIS (4 modes)")
        report_lines.append("-" * 70)
        report_lines.append(f"  H = {stat_kw:.4f}, p = {p_kw:.6f}")
        if p_kw < 0.05:
            report_lines.append(
                "  → Différence significative entre les modes (p < 0.05)"
            )
        else:
            report_lines.append(
                "  → Pas de différence significative entre les modes (p >= 0.05)"
            )
        report_lines.append("")

    # --- Mann-Whitney U (Montrée+Suggérée vs Désignée+Comportementale) ---
    group_high = np.concatenate([
        entropy_by_mode.get("Montrée", np.array([])),
        entropy_by_mode.get("Suggérée", np.array([])),
    ])
    group_low = np.concatenate([
        entropy_by_mode.get("Désignée", np.array([])),
        entropy_by_mode.get("Comportementale", np.array([])),
    ])

    if len(group_high) > 0 and len(group_low) > 0:
        stat_mw, p_mw = stats.mannwhitneyu(
            group_high, group_low, alternative="greater"
        )
        report_lines.append("-" * 70)
        report_lines.append(
            "TEST DE MANN-WHITNEY U (Montrée+Suggérée vs Désignée+Comportementale)"
        )
        report_lines.append("-" * 70)
        report_lines.append(
            f"  Hypothèse H1 : entropie(Montrée+Suggérée) > entropie(Désignée+Comportementale)"
        )
        report_lines.append(f"  U = {stat_mw:.4f}, p (unilatéral) = {p_mw:.6f}")
        report_lines.append(
            f"  Moyenne Montrée+Suggérée     : {np.mean(group_high):.4f} (n={len(group_high)})"
        )
        report_lines.append(
            f"  Moyenne Désignée+Comportementale : {np.mean(group_low):.4f} (n={len(group_low)})"
        )
        report_lines.append(
            f"  Différence de moyennes       : {np.mean(group_high) - np.mean(group_low):.4f}"
        )

        # Taille d'effet (rank-biserial correlation)
        n1, n2 = len(group_high), len(group_low)
        r = 1 - (2 * stat_mw) / (n1 * n2)
        report_lines.append(f"  Taille d'effet (r)           : {r:.4f}")

        if p_mw < 0.001:
            conclusion = "TRÈS FORTEMENT supportée (p < 0.001)"
        elif p_mw < 0.01:
            conclusion = "FORTEMENT supportée (p < 0.01)"
        elif p_mw < 0.05:
            conclusion = "SUPPORTÉE (p < 0.05)"
        else:
            conclusion = "NON SUPPORTÉE (p >= 0.05)"

        report_lines.append(f"  → Hypothèse : {conclusion}")
        report_lines.append("")

    # --- Tests par paires (Mann-Whitney U) ---
    report_lines.append("-" * 70)
    report_lines.append("COMPARAISONS PAR PAIRES (Mann-Whitney U, bilatéral)")
    report_lines.append("-" * 70)
    for i, m1 in enumerate(MODES):
        for m2 in MODES[i + 1 :]:
            v1 = entropy_by_mode.get(m1, np.array([]))
            v2 = entropy_by_mode.get(m2, np.array([]))
            if len(v1) > 0 and len(v2) > 0:
                stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                sig = (
                    "***" if p < 0.001
                    else "**" if p < 0.01
                    else "*" if p < 0.05
                    else "ns"
                )
                report_lines.append(
                    f"  {m1} vs {m2} : U={stat:.1f}, p={p:.6f} [{sig}]"
                )

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("FIN DU RAPPORT")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcule la spécificité des marqueurs linguistiques (SimpleSitEmo).",
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Chemin du CSV de marqueurs (défaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--outdir", "-o",
        default=DEFAULT_OUTDIR,
        help=f"Dossier de sortie (défaut: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Fréquence minimale d'un marqueur pour les calculs d'entropie (défaut: 3)",
    )
    parser.add_argument(
        "--marker-type",
        choices=["word", "lemma", "punctuation", "all"],
        default="all",
        help="Type de marqueur à analyser (défaut: all)",
    )
    args = parser.parse_args()

    print("=== Calcul de spécificité des marqueurs (SimpleSitEmo) ===")

    # Lecture des marqueurs
    if not os.path.isfile(args.input):
        print(f"Fichier de marqueurs introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    validate_normalized_markers(df, table_name="input markers")
    print(f"Marqueurs chargés : {len(df)} lignes")

    # Filtrage par type de marqueur
    if args.marker_type != "all":
        df = df[df["marker_type"] == args.marker_type].copy()
        print(
            f"Filtrage par type '{args.marker_type}' : "
            f"{len(df)} marqueurs restants"
        )

    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Entropie sur les émotions H(Emotion | Marqueur) ---
    print("--- Calcul H(Emotion | Marqueur) ---")
    entropy_emotion = compute_conditional_entropy(
        df, "emotion", EMOTIONS, min_freq=args.min_freq
    )
    if not entropy_emotion.empty:
        path = os.path.join(args.outdir, "entropy_per_marker_emotion.csv")
        entropy_emotion.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.2f")
        print(f"Exporté : {path}")

        # Top 10 marqueurs les plus déterminés (entropie la plus basse)
        print("Top 10 marqueurs les plus DÉTERMINÉS (entropie basse) :")
        top_determined = entropy_emotion.nsmallest(10, "entropy")
        for _, row in top_determined.iterrows():
            print(
                f"  {row['marker_value']:<20} ({row['marker_type']}) : "
                f"H={row['entropy']:.4f}, n={row['total_count']}"
            )

        # Top 10 marqueurs les plus indéterminés
        print("Top 10 marqueurs les plus INDÉTERMINÉS (entropie haute) :")
        top_indetermined = entropy_emotion.nlargest(10, "entropy")
        for _, row in top_indetermined.iterrows():
            print(
                f"  {row['marker_value']:<20} ({row['marker_type']}) : "
                f"H={row['entropy']:.4f}, n={row['total_count']}"
            )

    # --- 2. Entropie sur les modes H(Mode | Marqueur) ---
    print("--- Calcul H(Mode | Marqueur) ---")
    entropy_mode = compute_conditional_entropy(
        df, "mode", MODES, min_freq=args.min_freq
    )
    if not entropy_mode.empty:
        path = os.path.join(args.outdir, "entropy_per_marker_mode.csv")
        entropy_mode.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.2f")
        print(f"Exporté : {path}")

    # --- 3. Entropie moyenne par mode ---
    print("--- Entropie moyenne par mode ---")
    entropy_by_mode = compute_entropy_by_mode(df, entropy_emotion)
    if not entropy_by_mode.empty:
        path = os.path.join(args.outdir, "entropy_by_mode_summary.csv")
        entropy_by_mode.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.2f")
        print(f"Exporté : {path}")
        print("Résumé par mode :")
        for _, row in entropy_by_mode.iterrows():
            print(
                f"  {row['mode']:<20} : n={row['n_markers']}, "
                f"moy={row['mean_entropy']:.4f}, méd={row['median_entropy']:.4f}"
            )

    # --- 4. Test de l'hypothèse ---
    print("--- Test de l'hypothèse ---")
    report = test_hypothesis(df, entropy_emotion)
    report_path = os.path.join(args.outdir, "hypothesis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport exporté : {report_path}")
    print("\n" + report)

    print("Calcul de spécificité terminé")


if __name__ == "__main__":
    main()