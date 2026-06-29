#!/usr/bin/env python3
"""Generate an offline HTML/CSS view of CyberAdoAgg Montree = 1 annotations."""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    BASE_DIR
    / "data"
    / "raw"
    / "xlsx"
    / "CyberAdoAgg_gold_global_total_latest.xlsx"
)
DEFAULT_OUTPUT = BASE_DIR / "results" / "montree_annotations.html"

SPAN_TEXT_RE = re.compile(r"^span(\d+)_text$")
def cell_text(value: Any) -> str:
    """Return a stripped display value, with pandas missing values normalized to ''."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_key(value: Any) -> str:
    """Normalize labels for robust comparisons across accents and spacing."""
    text = cell_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_sheet(sheet: object) -> object:
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def detect_span_numbers(columns: pd.Index) -> list[int]:
    span_numbers = []
    for column in columns:
        match = SPAN_TEXT_RE.match(str(column))
        if match:
            span_numbers.append(int(match.group(1)))
    return sorted(span_numbers)


def is_montree_value(value: Any) -> bool:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0] == 1


def is_shown_mode(value: Any) -> bool:
    return normalize_key(value) == "montree"


def html_text(value: str) -> str:
    return escape(value).replace("\n", "<br>\n")


def find_all(text: str, needle: str) -> list[int]:
    starts = []
    start_at = 0
    while True:
        start = text.find(needle, start_at)
        if start < 0:
            break
        starts.append(start)
        start_at = start + 1
    return starts


def locate_spans(full_text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach start/end offsets to spans when their text is found in TEXT."""
    occurrence_counts: defaultdict[str, int] = defaultdict(int)
    unmatched = []

    for span in spans:
        span_text = span["text"]
        starts = find_all(full_text, span_text)
        if not starts:
            unmatched.append(span)
            span["start"] = None
            span["end"] = None
            continue

        occurrence_index = occurrence_counts[span_text]
        start = starts[min(occurrence_index, len(starts) - 1)]
        occurrence_counts[span_text] += 1
        span["start"] = start
        span["end"] = start + len(span_text)

    return unmatched


def render_nature_label(nature: str) -> str:
    return f'<span class="nature-label">{escape(nature)}</span>'


def render_annotated_text(full_text: str, spans: list[dict[str, Any]]) -> str:
    ranges = [
        (span["start"], span["end"], span)
        for span in spans
        if span.get("start") is not None and span.get("end") is not None
    ]
    if not ranges:
        return html_text(full_text)

    boundaries = {0, len(full_text)}
    for start, end, _span in ranges:
        boundaries.add(start)
        boundaries.add(end)

    pieces = []
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:]):
        if left == right:
            continue
        segment = full_text[left:right]
        active = [
            span
            for start, end, span in ranges
            if start <= left and right <= end
        ]
        if not active:
            pieces.append(html_text(segment))
            continue

        title = " ; ".join(span["nature"] for span in active)
        overlap = " overlap" if len(active) > 1 else ""
        pieces.append(
            '<span class="text-mark'
            f'{overlap}" title="{escape(title, quote=True)}">'
            f"{html_text(segment)}</span>"
        )

        ending_spans = [
            span
            for _start, end, span in ranges
            if end == right and span.get("nature")
        ]
        ending_spans.sort(
            key=lambda span: (span["end"] - span["start"], span["span_no"])
        )
        pieces.extend(render_nature_label(span["nature"]) for span in ending_spans)

    return "".join(pieces)


def build_records(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_columns = {"Montree", "TEXT"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes: {', '.join(missing)}")

    span_numbers = detect_span_numbers(df.columns)
    if not span_numbers:
        raise ValueError("Aucune colonne spanN_text trouvée.")

    records = []
    missing_nature = []
    unmatched_spans = []
    excluded_spans = 0
    montree_rows = 0

    for row_index, row in df.iterrows():
        if not is_montree_value(row["Montree"]):
            continue

        montree_rows += 1
        spans = []
        for span_no in span_numbers:
            span_text = cell_text(row.get(f"span{span_no}_text"))
            if not span_text:
                continue

            if not is_shown_mode(row.get(f"span{span_no}_mode")):
                excluded_spans += 1
                continue

            nature = cell_text(row.get(f"nature_linguistique_span_{span_no}"))
            if not nature:
                missing_nature.append(
                    {
                        "excel_row": row_index + 2,
                        "idx": cell_text(row.get("idx")),
                        "ID": cell_text(row.get("ID")),
                        "span_no": span_no,
                        "span_text": span_text,
                    }
                )

            spans.append(
                {
                    "span_no": span_no,
                    "text": span_text,
                    "nature": nature,
                }
            )

        if not spans:
            continue

        full_text = cell_text(row["TEXT"])
        unmatched_spans.extend(
            {
                "excel_row": row_index + 2,
                "idx": cell_text(row.get("idx")),
                "ID": cell_text(row.get("ID")),
                "span_no": span["span_no"],
                "span_text": span["text"],
            }
            for span in locate_spans(full_text, spans)
        )

        records.append(
            {
                "text": full_text,
                "spans": spans,
            }
        )

    stats = {
        "montree_rows": montree_rows,
        "displayed_rows": len(records),
        "displayed_spans": sum(len(record["spans"]) for record in records),
        "excluded_spans": excluded_spans,
        "missing_nature": missing_nature,
        "unmatched_spans": unmatched_spans,
    }
    return records, stats


def render_record(record: dict[str, Any]) -> str:
    return f"""
      <article class="message">
        <p class="message-text">{render_annotated_text(record['text'], record['spans'])}</p>
      </article>
    """


def render_html(
    records: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    record_html = "\n".join(render_record(record) for record in records)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CyberAdoAgg - annotations Montrée</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #fbfbfa;
      --ink: #222426;
      --muted: #6a7078;
      --line: #e2e4e7;
      --mark: #fff0a6;
      --mark-line: #6b5b12;
      --overlap: #f8d1bd;
      --label-bg: #f4f5f6;
      --label-line: #d2d6db;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 16px;
      line-height: 1.65;
      letter-spacing: 0;
    }}
    main {{
      width: min(920px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    .page-header {{
      margin-bottom: 22px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
      font-weight: 680;
    }}
    .records {{
      display: grid;
      gap: 0;
    }}
    .message {{
      border-bottom: 1px solid var(--line);
      padding: 14px 0;
    }}
    .message-text {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .text-mark {{
      background: var(--mark);
      border-bottom: 2px solid var(--mark-line);
      padding: 0 1px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }}
    .text-mark.overlap {{
      background: var(--overlap);
      border-bottom-style: double;
    }}
    .nature-label {{
      display: inline-block;
      margin: 0 0.35rem 0 0.25rem;
      border: 1px solid var(--label-line);
      border-radius: 999px;
      background: var(--label-bg);
      color: var(--muted);
      padding: 0.05rem 0.38rem;
      font-size: 0.72rem;
      line-height: 1.35;
      vertical-align: 0.12em;
      white-space: nowrap;
    }}
    @media (max-width: 560px) {{
      main {{
        width: min(100vw - 20px, 1320px);
        padding-top: 20px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="page-header">
      <h1>Émotions montrées</h1>
    </header>

    <section class="records" aria-label="Messages et spans Montrée">
      {record_html}
    </section>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère un fichier HTML/CSS autonome pour visualiser les spans "
            "d'émotions montrées dans CyberAdoAgg."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Fichier XLSX source (défaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Nom ou index de la feuille Excel à lire (défaut: 0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Fichier HTML de sortie (défaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--allow-missing-nature",
        action="store_true",
        help="Continuer même si des spans Montrée n'ont pas nature_linguistique.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_excel(args.input, sheet_name=parse_sheet(args.sheet))
    records, stats = build_records(df)

    if stats["missing_nature"] and not args.allow_missing_nature:
        print("ERREUR: nature_linguistique manquante pour des spans Montrée.")
        print(pd.DataFrame(stats["missing_nature"]).to_string(index=False))
        return 1

    if stats["unmatched_spans"]:
        print("ATTENTION: certains spans Montrée n'ont pas été retrouvés dans TEXT.")
        print(pd.DataFrame(stats["unmatched_spans"]).to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(records, stats), encoding="utf-8")

    print(f"HTML écrit: {args.output}")
    print(f"Lignes Montree = 1: {stats['montree_rows']}")
    print(f"Messages affichés: {stats['displayed_rows']}")
    print(f"Spans Montrée affichés: {stats['displayed_spans']}")
    print(f"Spans non Montrée exclus: {stats['excluded_spans']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
