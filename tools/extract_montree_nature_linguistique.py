#!/usr/bin/env python3
"""Extract nature_linguistique distribution for Montree = 1 spans."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    BASE_DIR
    / "data"
    / "raw"
    / "xlsx"
    / "CyberAdoAgg_gold_global_total_latest.xlsx"
)
DEFAULT_OUTPUT = (
    BASE_DIR
    / "results"
    / "montree_nature_linguistique_distribution.csv"
)


def is_non_empty(series: pd.Series) -> pd.Series:
    """Return True for values that are neither NA nor blank after stripping."""
    return series.notna() & series.astype("string").str.strip().ne("")


def detect_span_numbers(columns: pd.Index) -> list[int]:
    span_numbers = []
    prefix = "nature_linguistique_span_"
    for column in columns:
        if column.startswith(prefix):
            suffix = column.removeprefix(prefix)
            if suffix.isdigit():
                span_numbers.append(int(suffix))
    return sorted(span_numbers)


def build_long_nature_table(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Montree", "n_spans"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {', '.join(missing_columns)}")

    span_numbers = detect_span_numbers(df.columns)
    if not span_numbers:
        raise ValueError("Aucune colonne nature_linguistique_span_N trouvee.")

    montree_mask = pd.to_numeric(df["Montree"], errors="coerce").eq(1)
    montree_df = df[montree_mask].copy()
    montree_df["_excel_row"] = montree_df.index + 2
    n_spans = pd.to_numeric(montree_df["n_spans"], errors="coerce")

    records = []
    for span_no in span_numbers:
        nature_col = f"nature_linguistique_span_{span_no}"
        text_col = f"span{span_no}_text"

        required_by_n_spans = n_spans.ge(span_no).fillna(False)
        if text_col in montree_df.columns:
            required_by_text = is_non_empty(montree_df[text_col])
            span_text = montree_df[text_col]
        else:
            required_by_text = pd.Series(False, index=montree_df.index)
            span_text = pd.Series(pd.NA, index=montree_df.index)

        required_span = required_by_n_spans | required_by_text
        span_df = montree_df.loc[required_span, ["_excel_row", "idx", "ID", "n_spans"]].copy()
        span_df["span_no"] = span_no
        span_df["span_text"] = span_text.loc[required_span].astype("string").str.strip()
        span_df["nature_linguistique"] = (
            montree_df.loc[required_span, nature_col].astype("string").str.strip()
        )
        records.append(span_df)

    if not records:
        return pd.DataFrame(
            columns=[
                "_excel_row",
                "idx",
                "ID",
                "n_spans",
                "span_no",
                "span_text",
                "nature_linguistique",
            ]
        )

    return pd.concat(records, ignore_index=True)


def summarize(long_df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    total = len(long_df)
    counts = long_df["nature_linguistique"].value_counts().rename_axis(
        "nature_linguistique"
    )
    summary = counts.reset_index(name="nombre_occurrences")
    summary["proportion_corpus_pct"] = (
        summary["nombre_occurrences"] / total * 100
    ).round(decimals)
    return summary.sort_values(
        ["nombre_occurrences", "nature_linguistique"], ascending=[False, True]
    ).reset_index(drop=True)


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max([len(header), *(len(row[col_idx]) for row in rows)])
        for col_idx, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator, *(fmt_row(row) for row in rows)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filtre Montree = 1, verifie que chaque span attendu a une "
            "nature_linguistique, puis produit la distribution des natures."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Chemin du fichier XLSX source (defaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Nom ou index de la feuille Excel a lire (defaut: 0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Chemin du CSV de sortie (defaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Ne pas ecrire de CSV, afficher seulement le tableau.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Nombre de decimales pour les pourcentages (defaut: 2).",
    )
    return parser.parse_args()


def parse_sheet(sheet: object) -> object:
    if isinstance(sheet, str) and sheet.isdigit():
        return int(sheet)
    return sheet


def main() -> int:
    args = parse_args()
    df = pd.read_excel(args.input, sheet_name=parse_sheet(args.sheet))

    long_df = build_long_nature_table(df)
    missing = long_df[~is_non_empty(long_df["nature_linguistique"])]

    if not missing.empty:
        columns = ["_excel_row", "idx", "ID", "n_spans", "span_no", "span_text"]
        print(
            "ERREUR: nature_linguistique n'est pas renseignee pour tous les "
            "spans Montree = 1."
        )
        print(missing[columns].to_string(index=False))
        return 1

    summary = summarize(long_df, decimals=args.decimals)
    print(to_markdown_table(summary))

    if not args.no_csv:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\nCSV ecrit: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
