#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse de granularité forme/lemme et carte de dilution.

Lit la table token-level produite par ``build_token_lemma_table.py`` et
calcule :

1. L'entropie émotionnelle par niveau de granularité (surface vs lemme)
2. Les liens forme → lemme avec delta d'entropie
3. Un résumé par famille lemmatique
4. Une carte de dilution interactive (Plotly HTML)

Usage
-----
::

    python -m pipeline.compute_granularity \\
        --input results/simplesitemo_granularity/token_lemmas.csv \\
        --output-dir results/simplesitemo_granularity \\
        --min-freq 3
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from .emotion_taxonomy import EMOTIONS
from .nlp_utils import FR_STOPWORDS


# ── Chemin par défaut ─────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_INPUT = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo_granularity", "token_lemmas.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo_granularity"
)


# ── Palette de couleurs par émotion ───────────────────────────────────────

EMOTION_COLORS = {
    "Colère": "#E74C3C",
    "Dégoût": "#8E44AD",
    "Joie": "#F39C12",
    "Peur": "#2C3E50",
    "Surprise": "#1ABC9C",
    "Tristesse": "#3498DB",
    "Admiration": "#E67E22",
    "Culpabilité": "#95A5A6",
    "Embarras": "#D35400",
    "Fierté": "#27AE60",
    "Jalousie": "#C0392B",
}

# Couleur par défaut pour les émotions inconnues
_DEFAULT_COLOR = "#7F8C8D"

# Seuils de classification
DELTA_THRESHOLD = 0.1


# ── Calcul d'entropie ────────────────────────────────────────────────────


def _shannon_entropy(counts: np.ndarray) -> float:
    """Entropie de Shannon en bits."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_marker_entropy(
    df: pd.DataFrame,
    marker_col: str,
    emotion_col: str = "emotion",
    min_freq: int = 3,
) -> pd.DataFrame:
    """Calcule l'entropie émotionnelle pour chaque valeur de marqueur.

    Parameters
    ----------
    df : pd.DataFrame
        Table token-level filtrée (kept_for_analysis == True).
    marker_col : str
        Nom de la colonne servant de marqueur (ex: ``surface_lower``,
        ``lemma_spacy``, ``lemma_stanza``).
    emotion_col : str
        Colonne contenant l'émotion normalisée.
    min_freq : int
        Fréquence minimale pour conserver un marqueur.

    Returns
    -------
    pd.DataFrame
        Colonnes : marker_value, total_count, dominant_emotion,
        dominant_emotion_share, entropy, max_entropy, normalized_entropy
    """
    # Supprimer les NaN dans la colonne marqueur
    valid = df[df[marker_col].notna()].copy()

    # Compter par (marqueur, émotion)
    counts = (
        valid.groupby([marker_col, emotion_col])
        .size()
        .reset_index(name="count")
    )

    # Agréger par marqueur
    results = []
    for marker_val, grp in counts.groupby(marker_col):
        total = grp["count"].sum()
        if total < min_freq:
            continue

        # Distribution
        emotion_counts = grp.set_index(emotion_col)["count"]
        counts_arr = emotion_counts.values.astype(float)

        # Entropie
        entropy = _shannon_entropy(counts_arr)
        n_emotions = len(counts_arr)
        max_entropy = math.log2(n_emotions) if n_emotions > 1 else 0.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Émotion dominante
        dom_idx = emotion_counts.idxmax()
        dom_share = emotion_counts.max() / total

        results.append(
            {
                "marker_value": marker_val,
                "total_count": int(total),
                "dominant_emotion": dom_idx,
                "dominant_emotion_share": round(dom_share, 4),
                "entropy": round(entropy, 4),
                "max_entropy": round(max_entropy, 4),
                "normalized_entropy": round(norm_entropy, 4),
            }
        )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("total_count", ascending=False)
    return result_df


# ── Liens surface → lemme ────────────────────────────────────────────────


def build_surface_lemma_links(
    df: pd.DataFrame,
    backend: str,
    surface_entropy: pd.DataFrame,
    lemma_entropy: pd.DataFrame,
    min_freq: int = 3,
) -> pd.DataFrame:
    """Construit la table des liens forme → lemme avec delta d'entropie.

    Parameters
    ----------
    df : pd.DataFrame
        Table token-level filtrée.
    backend : str
        ``"spacy"`` ou ``"stanza"``.
    surface_entropy : pd.DataFrame
        Entropie par forme de surface (de ``compute_marker_entropy``).
    lemma_entropy : pd.DataFrame
        Entropie par lemme (de ``compute_marker_entropy``).
    min_freq : int
        Fréquence minimale.

    Returns
    -------
    pd.DataFrame
        Table des liens avec classification de granularité.
    """
    lemma_col = f"lemma_{backend}"

    # Supprimer les NaN
    valid = df[df[lemma_col].notna()].copy()

    # Compter les liens (surface_lower, lemma)
    link_counts = (
        valid.groupby(["surface_lower", lemma_col])
        .size()
        .reset_index(name="surface_lemma_count")
    )

    # Mode dominant par paire
    mode_counts = (
        valid.groupby(["surface_lower", lemma_col, "mode"])
        .size()
        .reset_index(name="mode_count")
    )
    dominant_mode = (
        mode_counts.sort_values("mode_count", ascending=False)
        .drop_duplicates(["surface_lower", lemma_col])
        [["surface_lower", lemma_col, "mode"]]
        .rename(columns={"mode": "dominant_mode"})
    )

    links = link_counts.merge(dominant_mode, on=["surface_lower", lemma_col], how="left")

    # Fusionner avec l'entropie surface
    surf_cols = surface_entropy.rename(
        columns={
            "marker_value": "surface_lower",
            "total_count": "surface_count",
            "dominant_emotion": "surface_dominant_emotion",
            "dominant_emotion_share": "surface_dominant_emotion_share",
            "entropy": "surface_entropy",
            "normalized_entropy": "surface_normalized_entropy",
        }
    )[
        [
            "surface_lower",
            "surface_count",
            "surface_dominant_emotion",
            "surface_entropy",
            "surface_normalized_entropy",
        ]
    ]
    links = links.merge(surf_cols, on="surface_lower", how="inner")

    # Fusionner avec l'entropie lemme
    lem_cols = lemma_entropy.rename(
        columns={
            "marker_value": lemma_col,
            "total_count": "lemma_count",
            "dominant_emotion": "lemma_dominant_emotion",
            "dominant_emotion_share": "lemma_dominant_emotion_share",
            "entropy": "lemma_entropy",
            "normalized_entropy": "lemma_normalized_entropy",
        }
    )[
        [
            lemma_col,
            "lemma_count",
            "lemma_dominant_emotion",
            "lemma_entropy",
            "lemma_normalized_entropy",
        ]
    ]
    links = links.merge(lem_cols, on=lemma_col, how="inner")

    if links.empty:
        return links

    # Delta d'entropie
    links["delta_entropy"] = (
        links["lemma_entropy"] - links["surface_entropy"]
    ).round(4)

    # Changement d'émotion dominante
    links["dominant_emotion_changed"] = (
        links["surface_dominant_emotion"] != links["lemma_dominant_emotion"]
    )

    # Part de la surface dans le lemme
    links["surface_share_in_lemma"] = (
        links["surface_lemma_count"] / links["lemma_count"]
    ).round(4)

    # ── Classification de granularité ─────────────────────────────────
    def _classify(row: pd.Series) -> str:
        lemma_val = row[lemma_col]

        # Stopword leakage
        if lemma_val in FR_STOPWORDS:
            return "stopword_leakage"

        # Backend artifact : lemme très court et différent de la surface
        if (
            isinstance(lemma_val, str)
            and len(lemma_val) < 3
            and lemma_val != row["surface_lower"]
        ):
            return "backend_artifact"

        delta = row["delta_entropy"]
        emotion_changed = row["dominant_emotion_changed"]

        if abs(delta) <= DELTA_THRESHOLD and not emotion_changed:
            return "specificity_preserved"
        elif delta > DELTA_THRESHOLD:
            return "lemma_dilutes_surface"
        elif delta < -DELTA_THRESHOLD:
            return "lemma_reveals_family"
        elif emotion_changed:
            return "dominant_emotion_shift"
        else:
            return "specificity_preserved"

    links["granularity_class"] = links.apply(_classify, axis=1)

    # Renommer la colonne lemme pour cohérence
    links = links.rename(columns={lemma_col: "lemma"})
    links.insert(0, "backend", backend)

    # Trier par |delta_entropy| décroissant
    links = links.sort_values(
        "delta_entropy", key=lambda s: s.abs(), ascending=False
    )

    return links


# ── Résumé par famille lemmatique ─────────────────────────────────────────


def build_lemma_family_summary(
    links: pd.DataFrame,
    lemma_entropy: pd.DataFrame,
    backend: str,
) -> pd.DataFrame:
    """Résumé par famille lemmatique : nombre de formes, distribution, cohérence.

    Parameters
    ----------
    links : pd.DataFrame
        Table des liens surface→lemme (sortie de ``build_surface_lemma_links``).
    lemma_entropy : pd.DataFrame
        Entropie par lemme.
    backend : str
        Nom du backend.

    Returns
    -------
    pd.DataFrame
    """
    if links.empty:
        return pd.DataFrame()

    backend_links = links[links["backend"] == backend].copy()
    if backend_links.empty:
        return pd.DataFrame()

    summaries = []
    for lemma, grp in backend_links.groupby("lemma"):
        n_forms = grp["surface_lower"].nunique()
        forms_sorted = (
            grp.sort_values("surface_lemma_count", ascending=False)
            ["surface_lower"]
            .unique()
        )
        surface_forms = ", ".join(forms_sorted[:10])

        total_count = grp["surface_lemma_count"].sum()

        # Forme dominante
        dominant_surface = forms_sorted[0] if len(forms_sorted) > 0 else ""
        dominant_surface_share = (
            grp[grp["surface_lower"] == dominant_surface]["surface_lemma_count"].sum()
            / total_count
            if total_count > 0
            else 0
        )

        # Entropie du lemme
        lem_row = lemma_entropy[lemma_entropy["marker_value"] == lemma]
        if not lem_row.empty:
            lem_entropy = lem_row.iloc[0]["entropy"]
            lem_dom_emotion = lem_row.iloc[0]["dominant_emotion"]
            lem_dom_share = lem_row.iloc[0]["dominant_emotion_share"]
        else:
            lem_entropy = None
            lem_dom_emotion = None
            lem_dom_share = None

        # Nombre de formes avec émotion différente du lemme
        n_emotion_diff = grp[grp["dominant_emotion_changed"]].shape[0]

        summaries.append(
            {
                "backend": backend,
                "lemma": lemma,
                "n_surface_forms": n_forms,
                "surface_forms": surface_forms,
                "total_count": int(total_count),
                "dominant_surface": dominant_surface,
                "dominant_surface_share": round(dominant_surface_share, 4),
                "dominant_emotion": lem_dom_emotion,
                "dominant_emotion_share": lem_dom_share,
                "emotion_entropy": lem_entropy,
                "n_forms_with_different_emotion": n_emotion_diff,
            }
        )

    result = pd.DataFrame(summaries)
    if not result.empty:
        result = result.sort_values("total_count", ascending=False)
    return result


# ── Delta par marqueur avec priorité de revue ────────────────────────────


def build_delta_by_marker(links: pd.DataFrame) -> pd.DataFrame:
    """Construit ``granularity_delta_by_marker.csv`` trié par |delta_entropy|.

    Ajoute une colonne ``review_priority`` : high / medium / low.
    """
    if links.empty:
        return links.copy()

    result = links.copy()

    def _priority(row: pd.Series) -> str:
        delta = abs(row["delta_entropy"])
        count = row.get("surface_count", 0)
        changed = row.get("dominant_emotion_changed", False)

        if delta > 0.3 or (changed and count >= 10):
            return "high"
        elif delta > 0.1 and count >= 5:
            return "medium"
        else:
            return "low"

    result["review_priority"] = result.apply(_priority, axis=1)
    result = result.sort_values(
        "delta_entropy", key=lambda s: s.abs(), ascending=False
    )

    return result


# ── Carte de dilution (Vue 3) ────────────────────────────────────────────


def build_dilution_map(
    links: pd.DataFrame,
    output_path: str,
) -> None:
    """Construit la carte de dilution interactive en HTML (Plotly).

    Axes :
    - X : entropie de la forme de surface
    - Y : entropie du lemme
    - Couleur : émotion dominante de la surface
    - Taille : fréquence de la surface
    - Dropdown : backend (spaCy / Stanza / combiné)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(
            "  ⚠ Plotly non disponible, carte de dilution non générée.",
            file=sys.stderr,
        )
        return

    if links.empty:
        print("  ⚠ Aucun lien pour la carte de dilution.", file=sys.stderr)
        return

    backends = sorted(links["backend"].unique())

    # Préparer les données
    fig = go.Figure()

    # ── Référence diagonale y = x ─────────────────────────────────────
    max_val = max(
        links["surface_entropy"].max(),
        links["lemma_entropy"].max(),
        1.0,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val * 1.05],
            y=[0, max_val * 1.05],
            mode="lines",
            line=dict(color="rgba(150, 150, 150, 0.5)", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
            name="y = x",
        )
    )

    # ── Zones de dilution / consolidation (annotations) ───────────────
    fig.add_annotation(
        x=0.2,
        y=max_val * 0.9,
        text="↑ Dilution<br>(lemme plus dispersé)",
        showarrow=False,
        font=dict(size=11, color="rgba(200, 100, 100, 0.7)"),
        xanchor="left",
    )
    fig.add_annotation(
        x=max_val * 0.7,
        y=0.15,
        text="↓ Consolidation<br>(lemme plus spécifique)",
        showarrow=False,
        font=dict(size=11, color="rgba(100, 150, 200, 0.7)"),
        xanchor="left",
    )

    # ── Traces par backend ────────────────────────────────────────────
    # Pour chaque backend, créer des traces par émotion
    all_traces_per_backend: dict[str, list[int]] = {}
    trace_idx = 1  # index 0 = diagonale

    for backend in backends:
        bl = links[links["backend"] == backend].copy()
        trace_indices = []

        # Taille des marqueurs (log scale, capped)
        sizes = np.clip(np.log2(bl["surface_count"].values + 1) * 4, 5, 30)

        for emotion in sorted(bl["surface_dominant_emotion"].unique()):
            mask = bl["surface_dominant_emotion"] == emotion
            subset = bl[mask]
            s = sizes[mask.values]

            color = EMOTION_COLORS.get(emotion, _DEFAULT_COLOR)

            hover_text = [
                (
                    f"<b>{row.surface_lower}</b> → <b>{row.lemma}</b><br>"
                    f"Backend: {row.backend}<br>"
                    f"Δ entropie: {row.delta_entropy:+.3f}<br>"
                    f"Surface: H={row.surface_entropy:.3f}, n={row.surface_count}<br>"
                    f"Lemme: H={row.lemma_entropy:.3f}, n={row.lemma_count}<br>"
                    f"Émotion surface: {row.surface_dominant_emotion}<br>"
                    f"Émotion lemme: {row.lemma_dominant_emotion}<br>"
                    f"Mode: {row.dominant_mode}<br>"
                    f"Classe: {row.granularity_class}"
                )
                for _, row in subset.iterrows()
            ]

            fig.add_trace(
                go.Scatter(
                    x=subset["surface_entropy"],
                    y=subset["lemma_entropy"],
                    mode="markers",
                    marker=dict(
                        size=s,
                        color=color,
                        opacity=0.7,
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                    ),
                    name=f"{emotion}",
                    text=hover_text,
                    hoverinfo="text",
                    legendgroup=emotion,
                    visible=True if backend == backends[0] else False,
                )
            )
            trace_indices.append(trace_idx)
            trace_idx += 1

        all_traces_per_backend[backend] = trace_indices

    # ── Vue combinée ──────────────────────────────────────────────────
    combined_indices = []
    for emotion in sorted(links["surface_dominant_emotion"].unique()):
        mask = links["surface_dominant_emotion"] == emotion
        subset = links[mask]
        sizes = np.clip(
            np.log2(subset["surface_count"].values + 1) * 4, 5, 30
        )
        color = EMOTION_COLORS.get(emotion, _DEFAULT_COLOR)

        # Symboles différents par backend
        symbols = []
        for _, row in subset.iterrows():
            symbols.append("circle" if row.backend == "spacy" else "diamond")

        hover_text = [
            (
                f"<b>{row.surface_lower}</b> → <b>{row.lemma}</b><br>"
                f"Backend: {row.backend}<br>"
                f"Δ entropie: {row.delta_entropy:+.3f}<br>"
                f"Surface: H={row.surface_entropy:.3f}, n={row.surface_count}<br>"
                f"Lemme: H={row.lemma_entropy:.3f}, n={row.lemma_count}<br>"
                f"Émotion surface: {row.surface_dominant_emotion}<br>"
                f"Émotion lemme: {row.lemma_dominant_emotion}<br>"
                f"Mode: {row.dominant_mode}<br>"
                f"Classe: {row.granularity_class}"
            )
            for _, row in subset.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=subset["surface_entropy"],
                y=subset["lemma_entropy"],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.7,
                    symbol=symbols,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                ),
                name=f"{emotion}",
                text=hover_text,
                hoverinfo="text",
                legendgroup=emotion,
                visible=False,
            )
        )
        combined_indices.append(trace_idx)
        trace_idx += 1

    # ── Dropdown pour backend ─────────────────────────────────────────
    total_traces = trace_idx
    buttons = []

    for backend in backends:
        visibility = [False] * total_traces
        visibility[0] = True  # diagonale toujours visible
        for idx in all_traces_per_backend[backend]:
            visibility[idx] = True
        buttons.append(
            dict(
                label=f"Backend: {backend}",
                method="update",
                args=[{"visible": visibility}],
            )
        )

    # Vue combinée
    vis_combined = [False] * total_traces
    vis_combined[0] = True
    for idx in combined_indices:
        vis_combined[idx] = True
    buttons.append(
        dict(
            label="Combiné (● spaCy, ◆ Stanza)",
            method="update",
            args=[{"visible": vis_combined}],
        )
    )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                "Carte de dilution forme/lemme<br>"
                "<sub>H(émotion|surface) vs H(émotion|lemme) — "
                "au-dessus de la diagonale = dilution</sub>"
            ),
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Entropie de la forme de surface H(émotion|surface)",
            gridcolor="rgba(128,128,128,0.15)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Entropie du lemme H(émotion|lemme)",
            gridcolor="rgba(128,128,128,0.15)",
            zeroline=False,
        ),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        legend=dict(
            title="Émotion dominante",
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="rgba(30,30,60,0.9)",
                font=dict(color="#e0e0e0"),
            )
        ],
        width=1000,
        height=800,
        hovermode="closest",
    )

    # Export HTML
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
    )
    print(f"  Carte de dilution exportée : {output_path}")


# ── Orchestration principale ──────────────────────────────────────────────


def run_granularity_analysis(
    input_path: str,
    output_dir: str,
    min_freq: int = 3,
    skip_viz: bool = False,
) -> None:
    """Lance l'analyse de granularité complète.

    Parameters
    ----------
    input_path : str
        Chemin vers ``token_lemmas.csv``.
    output_dir : str
        Répertoire de sortie.
    min_freq : int
        Fréquence minimale pour les calculs d'entropie.
    skip_viz : bool
        Si True, ne génère pas la carte de dilution HTML.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Étape 1 : Chargement ──────────────────────────────────────────
    print("Étape 1 : Chargement de la table token-level…")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"  {len(df)} lignes chargées")

    # Filtrer pour les calculs
    kept = df[df["kept_for_analysis"] == True].copy()  # noqa: E712
    n_kept = len(kept)
    print(f"  {n_kept} lignes conservées pour analyse (kept_for_analysis=True)")

    if n_kept == 0:
        print("  Aucune ligne conservée. Arrêt.", file=sys.stderr)
        return

    # ── Étape 2 : Entropie par niveau de granularité ──────────────────
    print(f"\nÉtape 2 : Calcul de l'entropie émotionnelle (min_freq={min_freq})…")

    backends = ["spacy", "stanza"]
    surface_entropies: dict[str, pd.DataFrame] = {}
    lemma_entropies: dict[str, pd.DataFrame] = {}

    # Entropie des formes de surface (identique pour les deux backends)
    print("  Surface : calcul de H(émotion|surface)…")
    surf_ent = compute_marker_entropy(kept, "surface_lower", min_freq=min_freq)
    print(f"    → {len(surf_ent)} formes de surface avec freq ≥ {min_freq}")

    for backend in backends:
        surface_entropies[backend] = surf_ent  # même pour les deux

        lemma_col = f"lemma_{backend}"
        print(f"  Lemme {backend} : calcul de H(émotion|lemme)…")
        lem_ent = compute_marker_entropy(kept, lemma_col, min_freq=min_freq)
        lemma_entropies[backend] = lem_ent
        print(f"    → {len(lem_ent)} lemmes {backend} avec freq ≥ {min_freq}")

    # ── Étape 3 : Liens surface → lemme ───────────────────────────────
    print("\nÉtape 3 : Construction des liens surface → lemme…")
    all_links = []

    for backend in backends:
        print(f"  Backend {backend}…")
        bl = build_surface_lemma_links(
            kept,
            backend,
            surface_entropies[backend],
            lemma_entropies[backend],
            min_freq=min_freq,
        )
        print(f"    → {len(bl)} liens forme/lemme")
        if not bl.empty:
            # Statistiques par classe
            class_counts = bl["granularity_class"].value_counts()
            for cls, cnt in class_counts.items():
                print(f"      {cls} : {cnt}")
        all_links.append(bl)

    links = pd.concat(all_links, ignore_index=True)

    # Export
    links_path = os.path.join(output_dir, "surface_lemma_links.csv")
    links.to_csv(links_path, index=False, encoding="utf-8-sig")
    print(f"  → Exporté : {links_path} ({len(links)} lignes)")

    # ── Étape 4 : Résumé par famille lemmatique ───────────────────────
    print("\nÉtape 4 : Résumé par famille lemmatique…")
    all_summaries = []
    for backend in backends:
        summary = build_lemma_family_summary(
            links, lemma_entropies[backend], backend
        )
        print(f"  {backend} : {len(summary)} familles lemmatiques")
        all_summaries.append(summary)

    family_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = os.path.join(output_dir, "lemma_family_summary.csv")
    family_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  → Exporté : {summary_path}")

    # ── Étape 5 : Delta par marqueur avec priorité ────────────────────
    print("\nÉtape 5 : Delta par marqueur avec priorité de revue…")
    delta = build_delta_by_marker(links)
    delta_path = os.path.join(output_dir, "granularity_delta_by_marker.csv")
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")
    n_high = (delta["review_priority"] == "high").sum() if not delta.empty else 0
    n_medium = (delta["review_priority"] == "medium").sum() if not delta.empty else 0
    print(f"  → {n_high} cas haute priorité, {n_medium} cas moyenne priorité")
    print(f"  → Exporté : {delta_path}")

    # ── Étape 6 : Carte de dilution ───────────────────────────────────
    if not skip_viz:
        print("\nÉtape 6 : Génération de la carte de dilution…")
        viz_path = os.path.join(output_dir, "dilution_map.html")
        build_dilution_map(links, viz_path)
    else:
        print("\nÉtape 6 : Carte de dilution ignorée (--skip-viz)")


# ── Point d'entrée CLI ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse de granularité forme/lemme : liens, deltas d'entropie, "
            "classification et carte de dilution interactive."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Fréquence minimale pour les calculs (défaut: 3)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Ne pas générer la carte de dilution HTML",
    )
    args = parser.parse_args()

    print("")
    print("Analyse de granularité forme/lemme")
    print("")

    if not os.path.isfile(args.input):
        print(f"Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    run_granularity_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        min_freq=args.min_freq,
        skip_viz=args.skip_viz,
    )

    print("")
    print("Analyse terminée.")
    print("")


if __name__ == "__main__":
    main()
