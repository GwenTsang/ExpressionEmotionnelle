#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualisation des variations flexionnelles par famille lemmatique.

Produit une page HTML interactive montrant, pour chaque famille
lemmatique multi-formes, la distribution émotionnelle de chaque
forme de surface comparée à celle du lemme regroupé.

Usage
-----
::

    python -m pipeline.viz_flexional_families \\
        --input results/simplesitemo_granularity/token_lemmas.csv \\
        --output results/simplesitemo_granularity/flexional_families.html \\
        --min-freq 3 \\
        --backend spacy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import pandas as pd

from .emotion_taxonomy import EMOTIONS


# ── Chemins par défaut ────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DEFAULT_INPUT = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo_granularity", "token_lemmas.csv"
)
DEFAULT_OUTPUT = os.path.join(
    _PROJECT_ROOT, "results", "simplesitemo_granularity", "flexional_families.html"
)


# ── Palette émotionnelle ─────────────────────────────────────────────────

EMOTION_COLORS = {
    "Colère": "#e74c3c",
    "Dégoût": "#8e44ad",
    "Joie": "#f1c40f",
    "Peur": "#34495e",
    "Surprise": "#1abc9c",
    "Tristesse": "#3498db",
    "Admiration": "#e67e22",
    "Culpabilité": "#95a5a6",
    "Embarras": "#d35400",
    "Fierté": "#27ae60",
    "Jalousie": "#c0392b",
}


# ── Construction des données par famille ──────────────────────────────────


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Entropie de Shannon en bits."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def build_family_data(
    df: pd.DataFrame,
    backend: str = "spacy",
    min_freq: int = 3,
) -> list[dict]:
    """Construit les données de visualisation pour chaque famille lemmatique.

    Parameters
    ----------
    df : pd.DataFrame
        Table token-level filtrée (kept_for_analysis == True).
    backend : str
        ``"spacy"`` ou ``"stanza"``.
    min_freq : int
        Fréquence minimale pour inclure une forme.

    Returns
    -------
    list[dict]
        Liste de familles, chaque famille contenant le lemme,
        les formes de surface et leurs distributions émotionnelles.
    """
    lemma_col = f"lemma_{backend}"

    # Ne garder que les lignes avec lemme valide
    valid = df[df[lemma_col].notna()].copy()

    # Identifier les familles multi-formes
    form_per_lemma = (
        valid.groupby(lemma_col)["surface_lower"]
        .nunique()
        .reset_index(name="n_forms")
    )
    multi_lemmas = set(
        form_per_lemma[form_per_lemma["n_forms"] > 1][lemma_col]
    )

    families = []

    for lemma in sorted(multi_lemmas):
        lemma_rows = valid[valid[lemma_col] == lemma]

        # Distribution émotionnelle globale du lemme
        lemma_emo = Counter(lemma_rows["emotion"])
        lemma_total = sum(lemma_emo.values())

        if lemma_total < min_freq:
            continue

        lemma_entropy = _shannon_entropy(lemma_emo)
        lemma_dominant = max(lemma_emo, key=lemma_emo.get)

        # Formes de surface
        forms = []
        for surface, grp in lemma_rows.groupby("surface_lower"):
            emo_counts = Counter(grp["emotion"])
            total = sum(emo_counts.values())
            if total < 1:
                continue

            entropy = _shannon_entropy(emo_counts)
            dominant = max(emo_counts, key=emo_counts.get)

            # POS tag dominant
            pos_col = f"pos_{backend}"
            pos_counts = grp[pos_col].value_counts()
            pos = pos_counts.index[0] if len(pos_counts) > 0 else ""

            forms.append(
                {
                    "surface": surface,
                    "total": total,
                    "emotions": dict(emo_counts),
                    "entropy": round(entropy, 3),
                    "dominant": dominant,
                    "pos": pos,
                }
            )

        if len(forms) < 2:
            continue

        # Trier les formes par fréquence décroissante
        forms.sort(key=lambda f: f["total"], reverse=True)

        # Classification de la famille
        dominants = set(f["dominant"] for f in forms)
        if len(dominants) == 1:
            coherence = "cohérente"
        elif len(dominants) == 2:
            coherence = "mixte"
        else:
            coherence = "hétérogène"

        # Mode dominant
        mode_counts = Counter(lemma_rows["mode"])
        dominant_mode = max(mode_counts, key=mode_counts.get)

        families.append(
            {
                "lemma": lemma,
                "total": lemma_total,
                "n_forms": len(forms),
                "entropy": round(lemma_entropy, 3),
                "dominant": lemma_dominant,
                "dominant_mode": dominant_mode,
                "coherence": coherence,
                "emotions": dict(lemma_emo),
                "forms": forms,
            }
        )

    # Trier par fréquence décroissante
    families.sort(key=lambda f: f["total"], reverse=True)

    return families


# ── Génération HTML ──────────────────────────────────────────────────────


def generate_html(families: list[dict], backend: str) -> str:
    """Génère la page HTML de visualisation des familles flexionnelles."""

    families_json = json.dumps(families, ensure_ascii=False)
    colors_json = json.dumps(EMOTION_COLORS, ensure_ascii=False)

    # Liste des émotions présentes dans les données
    all_emotions = set()
    for fam in families:
        all_emotions.update(fam["emotions"].keys())
        for form in fam["forms"]:
            all_emotions.update(form["emotions"].keys())
    emotions_list = json.dumps(sorted(all_emotions), ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Variations flexionnelles — familles lemmatiques ({backend})</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Inter', sans-serif;
    background: #f8fafc;
    color: #1f2937;
    line-height: 1.5;
}}

.header {{
    background: #ffffff;
    padding: 1.25rem 2rem 1rem;
    border-bottom: 1px solid #e5e7eb;
}}

.header h1 {{
    font-size: 1.6rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.3rem;
}}

.header .subtitle {{
    font-size: 0.85rem;
    color: #5f6b7a;
    font-weight: 300;
}}

.controls {{
    padding: 1rem 2rem;
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
}}

.controls label {{
    font-size: 0.8rem;
    color: #4b5563;
}}

.controls input, .controls select {{
    background: #ffffff;
    color: #111827;
    border: 1px solid #cfd6df;
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.8rem;
    font-family: inherit;
}}

.controls input:focus, .controls select:focus {{
    outline: none;
    border-color: #0f766e;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
}}

.controls input[type="search"] {{
    width: 220px;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem 2rem 3rem;
}}

.family-card {{
    background: #ffffff;
    border: 1px solid #e1e7ef;
    border-radius: 8px;
    margin-bottom: 1.2rem;
    overflow: hidden;
    transition: border-color 0.2s;
}}

.family-card:hover {{
    border-color: #b9c4d0;
}}

.family-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1.2rem;
    background: #ffffff;
    border-bottom: 1px solid #edf1f5;
    cursor: pointer;
    user-select: none;
}}

.family-header:hover {{
    background: #f3f6fa;
}}

.lemma-label {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #111827;
}}

.lemma-meta {{
    display: flex;
    gap: 0.8rem;
    align-items: center;
    font-size: 0.75rem;
    color: #64748b;
}}

.tag {{
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 500;
}}

.tag-coherente {{
    background: #e8f7ef;
    color: #1f8a4c;
    border: 1px solid #b7e4cb;
}}

.tag-mixte {{
    background: #fff4db;
    color: #a86200;
    border: 1px solid #f5d99b;
}}

.tag-hétérogène {{
    background: #fdeceb;
    color: #c0392b;
    border: 1px solid #f3beb8;
}}

.family-body {{
    padding: 1rem 1.2rem;
}}

.forms-grid {{
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}}

.form-row {{
    display: grid;
    grid-template-columns: 140px 40px 1fr 65px;
    align-items: center;
    gap: 0.8rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #eef2f6;
}}

.form-row:last-child {{
    border-bottom: none;
}}

.form-row.is-lemma {{
    background: #f3f6fa;
    border-radius: 6px;
    padding: 0.5rem 0.5rem;
    margin-bottom: 0.4rem;
    border-bottom: 2px solid #d9e1ea;
}}

.form-name {{
    font-size: 0.85rem;
    font-weight: 500;
    color: #263241;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

.form-row.is-lemma .form-name {{
    font-weight: 700;
    color: #111827;
    font-size: 0.9rem;
}}

.form-count {{
    font-size: 0.75rem;
    color: #64748b;
    text-align: right;
    font-variant-numeric: tabular-nums;
}}

.bar-container {{
    height: 22px;
    display: flex;
    border-radius: 4px;
    overflow: hidden;
    background: #eef2f6;
}}

.form-row.is-lemma .bar-container {{
    height: 26px;
}}

.bar-segment {{
    height: 100%;
    transition: opacity 0.2s;
    position: relative;
}}

.bar-segment:hover {{
    opacity: 0.85;
}}

.entropy-badge {{
    font-size: 0.7rem;
    color: #64748b;
    text-align: right;
    font-variant-numeric: tabular-nums;
}}

/* Légende */
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    padding: 0.8rem 1.2rem;
    border-top: 1px solid #edf1f5;
    background: #fbfcfe;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.7rem;
    color: #64748b;
}}

.legend-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
}}

/* Tooltip */
.tooltip {{
    position: fixed;
    background: #ffffff;
    border: 1px solid #cfd6df;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.75rem;
    color: #1f2937;
    pointer-events: none;
    z-index: 1000;
    max-width: 250px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
    display: none;
}}

.tooltip .tt-title {{
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.3rem;
}}

.tooltip .tt-row {{
    display: flex;
    justify-content: space-between;
    gap: 1rem;
}}

.no-results {{
    text-align: center;
    padding: 3rem;
    color: #556;
    font-size: 0.9rem;
}}

.expand-icon {{
    transition: transform 0.2s;
    color: #556;
    font-size: 0.8rem;
}}

.family-card.collapsed .expand-icon {{
    transform: rotate(-90deg);
}}

.family-card.collapsed .family-body,
.family-card.collapsed .legend {{
    display: none;
}}
</style>
</head>
<body>

<div class="header">
    <h1>Variations flexionnelles par famille lemmatique</h1>
    <p class="subtitle">
        Distribution émotionnelle comparée des formes fléchies —
        Backend : <strong>{backend}</strong>
    </p>
</div>

<div class="controls">
    <label>Rechercher&nbsp;:
        <input type="search" id="search" placeholder="lemme ou forme…">
    </label>
    <label>Cohérence&nbsp;:
        <select id="filterCoherence">
            <option value="all">Toutes</option>
            <option value="cohérente">Cohérentes</option>
            <option value="mixte">Mixtes</option>
            <option value="hétérogène">Hétérogènes</option>
        </select>
    </label>
    <label>Émotion&nbsp;:
        <select id="filterEmotion">
            <option value="all">Toutes</option>
        </select>
    </label>
    <label>Mode&nbsp;:
        <select id="filterMode">
            <option value="all">Tous</option>
        </select>
    </label>
    <label>Tri&nbsp;:
        <select id="sortBy">
            <option value="freq">Fréquence ↓</option>
            <option value="entropy">Entropie ↓</option>
            <option value="forms">Nb formes ↓</option>
            <option value="alpha">Alphabétique</option>
        </select>
    </label>
</div>

<div class="container" id="families"></div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = {families_json};
const COLORS = {colors_json};
const EMOTIONS = {emotions_list};
const DEFAULT_COLOR = '#555';

// Populate emotion filter
const selEmo = document.getElementById('filterEmotion');
EMOTIONS.forEach(e => {{
    const opt = document.createElement('option');
    opt.value = e; opt.textContent = e;
    selEmo.appendChild(opt);
}});

// Populate mode filter
const modes = [...new Set(DATA.map(f => f.dominant_mode))].sort();
const selMode = document.getElementById('filterMode');
modes.forEach(m => {{
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m;
    selMode.appendChild(opt);
}});

function getColor(emo) {{
    return COLORS[emo] || DEFAULT_COLOR;
}}

function renderBar(emotions, total) {{
    // Sort by count desc for stable rendering
    const entries = Object.entries(emotions).sort((a,b) => b[1] - a[1]);
    return entries.map(([emo, count]) => {{
        const pct = (count / total * 100).toFixed(1);
        return `<div class="bar-segment" style="width:${{pct}}%;background:${{getColor(emo)}}"
            data-emotion="${{emo}}" data-count="${{count}}" data-pct="${{pct}}"></div>`;
    }}).join('');
}}

function renderFamily(fam) {{
    const tagClass = fam.coherence === 'cohérente' ? 'tag-coherente' :
                     fam.coherence === 'mixte' ? 'tag-mixte' : 'tag-hétérogène';

    let formsHtml = '';

    // Lemma row (aggregate)
    formsHtml += `
        <div class="form-row is-lemma">
            <div class="form-name" title="Lemme : ${{fam.lemma}}">⊕ ${{fam.lemma}}</div>
            <div class="form-count">${{fam.total}}</div>
            <div class="bar-container">${{renderBar(fam.emotions, fam.total)}}</div>
            <div class="entropy-badge">H=${{fam.entropy}}</div>
        </div>`;

    // Individual forms
    fam.forms.forEach(f => {{
        const indent = f.surface === fam.lemma ? '' : '  ';
        formsHtml += `
        <div class="form-row">
            <div class="form-name" title="${{f.surface}} (${{f.pos}})">${{indent}}${{f.surface}} <span style="color:#556;font-size:0.7rem">${{f.pos}}</span></div>
            <div class="form-count">${{f.total}}</div>
            <div class="bar-container">${{renderBar(f.emotions, f.total)}}</div>
            <div class="entropy-badge">H=${{f.entropy}}</div>
        </div>`;
    }});

    // Légende : only emotions present in this family
    const familyEmotions = new Set();
    Object.keys(fam.emotions).forEach(e => familyEmotions.add(e));
    fam.forms.forEach(f => Object.keys(f.emotions).forEach(e => familyEmotions.add(e)));
    const legendHtml = [...familyEmotions].sort().map(e =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${{getColor(e)}}"></span>${{e}}</span>`
    ).join('');

    return `
    <div class="family-card" data-lemma="${{fam.lemma}}" data-coherence="${{fam.coherence}}"
         data-emotion="${{fam.dominant}}" data-mode="${{fam.dominant_mode}}"
         data-forms="${{fam.forms.map(f=>f.surface).join(' ')}}">
        <div class="family-header" onclick="this.parentElement.classList.toggle('collapsed')">
            <div>
                <span class="lemma-label">${{fam.lemma}}</span>
            </div>
            <div class="lemma-meta">
                <span>${{fam.n_forms}} formes</span>
                <span>n=${{fam.total}}</span>
                <span class="tag ${{tagClass}}">${{fam.coherence}}</span>
                <span style="color:${{getColor(fam.dominant)}}">\u25CF ${{fam.dominant}}</span>
                <span>${{fam.dominant_mode}}</span>
                <span class="expand-icon">\u25BC</span>
            </div>
        </div>
        <div class="family-body">
            <div class="forms-grid">${{formsHtml}}</div>
        </div>
        <div class="legend">${{legendHtml}}</div>
    </div>`;
}}

function render() {{
    const search = document.getElementById('search').value.toLowerCase();
    const cohFilter = document.getElementById('filterCoherence').value;
    const emoFilter = document.getElementById('filterEmotion').value;
    const modeFilter = document.getElementById('filterMode').value;
    const sortBy = document.getElementById('sortBy').value;

    let filtered = DATA.filter(fam => {{
        if (cohFilter !== 'all' && fam.coherence !== cohFilter) return false;
        if (emoFilter !== 'all' && fam.dominant !== emoFilter) return false;
        if (modeFilter !== 'all' && fam.dominant_mode !== modeFilter) return false;
        if (search) {{
            const allText = fam.lemma + ' ' + fam.forms.map(f => f.surface).join(' ');
            if (!allText.includes(search)) return false;
        }}
        return true;
    }});

    // Sort
    if (sortBy === 'freq') filtered.sort((a,b) => b.total - a.total);
    else if (sortBy === 'entropy') filtered.sort((a,b) => b.entropy - a.entropy);
    else if (sortBy === 'forms') filtered.sort((a,b) => b.n_forms - a.n_forms);
    else if (sortBy === 'alpha') filtered.sort((a,b) => a.lemma.localeCompare(b.lemma));

    const container = document.getElementById('families');
    if (filtered.length === 0) {{
        container.innerHTML = '<div class="no-results">Aucune famille ne correspond aux filtres.</div>';
    }} else {{
        container.innerHTML = filtered.map(renderFamily).join('');
    }}
}}

// Tooltip on bar segments
document.addEventListener('mouseover', e => {{
    const seg = e.target.closest('.bar-segment');
    if (!seg) return;
    const tt = document.getElementById('tooltip');
    const emo = seg.dataset.emotion;
    const count = seg.dataset.count;
    const pct = seg.dataset.pct;
    tt.innerHTML = `<div class="tt-title" style="color:${{getColor(emo)}}">${{emo}}</div>
        <div class="tt-row"><span>Occurrences</span><span>${{count}}</span></div>
        <div class="tt-row"><span>Proportion</span><span>${{pct}}%</span></div>`;
    tt.style.display = 'block';
}});

document.addEventListener('mousemove', e => {{
    const tt = document.getElementById('tooltip');
    if (tt.style.display === 'block') {{
        tt.style.left = (e.clientX + 12) + 'px';
        tt.style.top = (e.clientY - 10) + 'px';
    }}
}});

document.addEventListener('mouseout', e => {{
    if (e.target.closest('.bar-segment')) {{
        document.getElementById('tooltip').style.display = 'none';
    }}
}});

// Event listeners
document.getElementById('search').addEventListener('input', render);
document.getElementById('filterCoherence').addEventListener('change', render);
document.getElementById('filterEmotion').addEventListener('change', render);
document.getElementById('filterMode').addEventListener('change', render);
document.getElementById('sortBy').addEventListener('change', render);

// Initial render
render();
</script>
</body>
</html>"""

    return html


# ── Orchestration ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualisation des variations flexionnelles par famille lemmatique. "
            "Produit une page HTML interactive."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Table token-level CSV (défaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Fichier HTML de sortie (défaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Fréquence minimale pour inclure une forme (défaut: 3)",
    )
    parser.add_argument(
        "--backend",
        choices=["spacy", "stanza"],
        default="spacy",
        help="Backend de lemmatisation (défaut: spacy)",
    )
    args = parser.parse_args()

    print("")
    print("Visualisation des variations flexionnelles")
    print("")

    if not os.path.isfile(args.input):
        print(f"Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    # Chargement
    print(f"Chargement : {args.input}")
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    kept = df[df["kept_for_analysis"] == True].copy()  # noqa: E712
    print(f"  {len(kept)} lignes conservées pour analyse")

    # Construction des données
    print(f"Construction des familles ({args.backend}, min_freq={args.min_freq})…")
    families = build_family_data(kept, backend=args.backend, min_freq=args.min_freq)
    print(f"  {len(families)} familles multi-formes")

    n_coh = sum(1 for f in families if f["coherence"] == "cohérente")
    n_mix = sum(1 for f in families if f["coherence"] == "mixte")
    n_het = sum(1 for f in families if f["coherence"] == "hétérogène")
    print(f"  Cohérentes: {n_coh}, Mixtes: {n_mix}, Hétérogènes: {n_het}")

    # Génération HTML
    print("Génération HTML…")
    html = generate_html(families, args.backend)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"  → {args.output}")
    print("")


if __name__ == "__main__":
    main()
