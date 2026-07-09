#!/usr/bin/env python3
"""Build a single-sheet CyberAggAdo gold XLSX with SitEmo segment/declencheur fields.

The output keeps message-level columns, including the 19 binary EMOTYC labels
used by Eval-EMOTYC, but replaces historical spanN_* columns with
Sit_Emo_unit_N_* columns.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_XLSX = (
    BASE_DIR
    / "data"
    / "raw"
    / "xlsx"
    / "CyberAdoAgg_gold_global_total_latest.xlsx"
)
DEFAULT_DECLENCHEURS_JSONL = (
    BASE_DIR / "results" / "cyberaggado_declencheur_inference.jsonl"
)
DEFAULT_OUTPUT_XLSX = (
    BASE_DIR
    / "data"
    / "raw"
    / "xlsx"
    / "CyberAdoAgg_gold_global_total_latest_with_declencheurs.xlsx"
)

MAX_UNITS = 4
SPAN_COLUMNS_RE = re.compile(r"^span\d+_(text|cat|mode)$")
NATURE_SPAN_RE = re.compile(r"^nature_linguistique_span_\d+$")
UNIT_PREFIX = "Sit_Emo_unit"


def cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def empty_value(value: Any) -> bool:
    return cell_text(value) == ""


def parse_emotions(value: Any) -> list[str]:
    text = cell_text(value)
    if not text:
        return []
    return [part.strip() for part in text.split(" + ") if part.strip()][:3]


def normalize_mode(value: Any) -> str:
    text = cell_text(value)
    aliases = {
        "Designee": "Désignée",
        "Montree": "Montrée",
        "Suggeree": "Suggérée",
    }
    return aliases.get(text, text)


def record_key(row_index: int, source_idx: Any, unit_no: int) -> str:
    return f"{row_index}:{cell_text(source_idx)}:span{unit_no}"


def legacy_span_id(source_idx: Any, unit_no: int) -> str:
    return f"{cell_text(source_idx)}:span{unit_no}"


def load_declencheur_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    by_key: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        row_index = item.get("row_index")
        span_id = item.get("span_id")
        if row_index is None or not span_id:
            continue
        key = f"{row_index}:{span_id}"
        by_key[key] = item
    return by_key


def find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    spans = []
    start_at = 0
    while True:
        start = text.find(needle, start_at)
        if start < 0:
            break
        end = start + len(needle)
        spans.append((start, end))
        start_at = start + 1
    return spans


def offsets_in_text(
    text: str,
    needle: str,
    *,
    occurrence_index: int | None = None,
) -> list[list[int]] | None:
    occurrences = find_occurrences(text, needle)
    if occurrence_index is None:
        if len(occurrences) != 1:
            return None
        start, end = occurrences[0]
        return [[start, end]]

    if not (0 <= occurrence_index < len(occurrences)):
        return None
    start, end = occurrences[occurrence_index]
    return [[start, end]]


def declencheur_offsets_in_segment(
    *,
    segment_text: str,
    segment_offsets: list[list[int]] | None,
    declencheurs: list[str],
) -> list[list[int]] | None:
    if not declencheurs or not segment_offsets or len(segment_offsets) != 1:
        return None

    segment_start = segment_offsets[0][0]
    relative_offsets: list[list[int]] = []
    used_ranges: list[tuple[int, int]] = []

    for declencheur in declencheurs:
        occurrences = find_occurrences(segment_text, declencheur)
        available = [
            (start, end)
            for start, end in occurrences
            if not any(max(start, used_start) < min(end, used_end) for used_start, used_end in used_ranges)
        ]
        if len(available) != 1:
            return None
        start, end = available[0]
        used_ranges.append((start, end))
        relative_offsets.append([segment_start + start, segment_start + end])

    return relative_offsets


def json_or_na(value: Any) -> Any:
    if value is None:
        return pd.NA
    return json.dumps(value, ensure_ascii=False)


def bool_or_na(value: bool | None) -> Any:
    if value is None:
        return pd.NA
    return bool(value)


def declencheurs_for_unit(
    *,
    row_index: int,
    source_idx: Any,
    unit_no: int,
    mode: str,
    span_text: str,
    results_by_key: dict[str, dict[str, Any]],
) -> list[str]:
    if mode == "Désignée":
        return [span_text] if span_text else []
    if mode == "Montrée":
        return []

    result = results_by_key.get(f"{row_index}:{legacy_span_id(source_idx, unit_no)}")
    if not result:
        return []

    prediction = result.get("prediction") or {}
    declencheurs = prediction.get("declencheurs") or []
    return [cell_text(item) for item in declencheurs if cell_text(item)]


def segment_for_unit(mode: str, span_text: str) -> str:
    if mode == "Désignée":
        return ""
    return span_text


def previous_same_segment_count(row: pd.Series, unit_no: int, segment_text: str) -> int:
    count = 0
    for previous_no in range(1, unit_no):
        previous_span = cell_text(row.get(f"span{previous_no}_text"))
        previous_mode = normalize_mode(row.get(f"span{previous_no}_mode"))
        previous_segment = segment_for_unit(previous_mode, previous_span)
        if previous_segment == segment_text:
            count += 1
    return count


def build_unit_payload(
    *,
    row: pd.Series,
    row_index: int,
    unit_no: int,
    results_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_idx = row.get("idx")
    message = cell_text(row.get("TEXT"))
    span_text = cell_text(row.get(f"span{unit_no}_text"))
    mode = normalize_mode(row.get(f"span{unit_no}_mode"))
    emotions = parse_emotions(row.get(f"span{unit_no}_cat"))
    nature = cell_text(row.get(f"nature_linguistique_span_{unit_no}"))

    if not span_text and not mode and not emotions and not nature:
        return {}

    segment_text = segment_for_unit(mode, span_text)
    segment_offsets = None
    if segment_text:
        segment_offsets = offsets_in_text(message, segment_text)
        if segment_offsets is None:
            # The source XLSX had no offsets. If the same short segment occurs
            # multiple times, align repeated identical unit texts by order.
            segment_offsets = offsets_in_text(
                message,
                segment_text,
                occurrence_index=previous_same_segment_count(row, unit_no, segment_text),
            )
    segment_is_discontinuous = False if segment_text else None

    declencheurs = declencheurs_for_unit(
        row_index=row_index,
        source_idx=source_idx,
        unit_no=unit_no,
        mode=mode,
        span_text=span_text,
        results_by_key=results_by_key,
    )
    declencheur_text = "; ".join(declencheurs)

    if mode == "Désignée":
        declencheur_offsets = offsets_in_text(message, declencheur_text)
    else:
        declencheur_offsets = declencheur_offsets_in_segment(
            segment_text=segment_text,
            segment_offsets=segment_offsets,
            declencheurs=declencheurs,
        )

    declencheur_is_discontinuous = None
    if declencheurs:
        declencheur_is_discontinuous = len(declencheurs) > 1

    prefix = f"{UNIT_PREFIX}_{unit_no}"
    return {
        f"{prefix}_id": f"{cell_text(source_idx)}:unit{unit_no}",
        f"{prefix}_mode": mode or pd.NA,
        f"{prefix}_emotion1": emotions[0] if len(emotions) > 0 else pd.NA,
        f"{prefix}_emotion2": emotions[1] if len(emotions) > 1 else pd.NA,
        f"{prefix}_emotion3": emotions[2] if len(emotions) > 2 else pd.NA,
        f"{prefix}_nature_linguistique": nature or pd.NA,
        f"{prefix}_segment_text": segment_text or pd.NA,
        f"{prefix}_segment_offsets_json": json_or_na(segment_offsets),
        f"{prefix}_segment_is_discontinuous": bool_or_na(segment_is_discontinuous),
        f"{prefix}_declencheur_text": declencheur_text or pd.NA,
        f"{prefix}_declencheur_offsets_json": json_or_na(declencheur_offsets),
        f"{prefix}_declencheur_is_discontinuous": bool_or_na(declencheur_is_discontinuous),
    }


def historical_span_columns(columns: list[str]) -> set[str]:
    old = {"n_spans", "spans_json"}
    for column in columns:
        if SPAN_COLUMNS_RE.match(column) or NATURE_SPAN_RE.match(column):
            old.add(column)
    return old


def build_output_dataframe(
    df: pd.DataFrame,
    *,
    results_by_key: dict[str, dict[str, Any]],
    max_units: int,
) -> pd.DataFrame:
    drop_columns = historical_span_columns(list(df.columns))
    base_columns = [column for column in df.columns if column not in drop_columns]
    output = df[base_columns].copy()

    output["n_Sit_Emo_units"] = 0

    for unit_no in range(1, max_units + 1):
        prefix = f"{UNIT_PREFIX}_{unit_no}"
        for suffix in (
            "id",
            "mode",
            "emotion1",
            "emotion2",
            "emotion3",
            "nature_linguistique",
            "segment_text",
            "segment_offsets_json",
            "segment_is_discontinuous",
            "declencheur_text",
            "declencheur_offsets_json",
            "declencheur_is_discontinuous",
        ):
            output[f"{prefix}_{suffix}"] = pd.NA

    for row_index, row in df.iterrows():
        n_units = 0
        for unit_no in range(1, max_units + 1):
            payload = build_unit_payload(
                row=row,
                row_index=int(row_index),
                unit_no=unit_no,
                results_by_key=results_by_key,
            )
            if not payload:
                continue
            n_units += 1
            for column, value in payload.items():
                output.at[row_index, column] = value
        output.at[row_index, "n_Sit_Emo_units"] = n_units

    return output


def summarize(output: pd.DataFrame, max_units: int) -> dict[str, int]:
    n_units = int(output["n_Sit_Emo_units"].sum())
    n_discontinuous_declencheurs = 0
    n_missing_segment_offsets = 0
    n_missing_declencheur_offsets = 0

    for unit_no in range(1, max_units + 1):
        prefix = f"{UNIT_PREFIX}_{unit_no}"
        decl_col = f"{prefix}_declencheur_text"
        decl_disc_col = f"{prefix}_declencheur_is_discontinuous"
        decl_offsets_col = f"{prefix}_declencheur_offsets_json"
        seg_col = f"{prefix}_segment_text"
        seg_offsets_col = f"{prefix}_segment_offsets_json"

        n_discontinuous_declencheurs += int(output[decl_disc_col].fillna(False).sum())
        n_missing_segment_offsets += int(
            output[seg_col].notna().sum() - output[seg_offsets_col].notna().sum()
        )
        n_missing_declencheur_offsets += int(
            output[decl_col].notna().sum() - output[decl_offsets_col].notna().sum()
        )

    return {
        "n_rows": len(output),
        "n_units": n_units,
        "n_columns": len(output.columns),
        "n_discontinuous_declencheurs": n_discontinuous_declencheurs,
        "n_missing_segment_offsets": n_missing_segment_offsets,
        "n_missing_declencheur_offsets": n_missing_declencheur_offsets,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construit un XLSX CyberAggAdo enrichi avec segment/declencheur."
    )
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--declencheurs-jsonl", type=Path, default=DEFAULT_DECLENCHEURS_JSONL)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--max-units", type=int, default=MAX_UNITS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = pd.read_excel(args.input_xlsx)
    results_by_key = load_declencheur_results(args.declencheurs_jsonl)
    output = build_output_dataframe(df, results_by_key=results_by_key, max_units=args.max_units)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output.to_excel(args.output_xlsx, index=False, engine="openpyxl")

    summary = summarize(output, args.max_units)
    print(f"Écrit : {args.output_xlsx}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
