#!/usr/bin/env python3
"""Build a CAA-like XLSX from the raw Glozz corpus.

The script reads only the Glozz ``.aa``/``.ac`` tree.  It reconstructs textual
rows from Glozz paragraph units and a conservative sentence splitter, then adds
wide ``Sit_Emo_unit_N_*`` columns similar to ``CAA_raw.xlsx``.

``emotexttokids_gold_flat.xlsx`` is not an input.  It can optionally be supplied
with ``--reference-xlsx`` only to audit how close the reconstructed TEXT values
are to the historical flat export.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GLOZZ_DIR = BASE_DIR / "data" / "raw" / "glozz"
DEFAULT_OUTPUT_XLSX = BASE_DIR / "data" / "raw" / "xlsx" / "Glozz_raw.xlsx"
DEFAULT_OUTPUT_PARQUET = BASE_DIR / "data" / "raw" / "xlsx" / "Glozz_raw.parquet"
UNIT_PREFIX = "Sit_Emo_unit"

SENTENCE_CLOSERS = {'"', "»", "”", "’"}
NO_SPLIT_AFTER = {
    "av",
    "apr",
    "dr",
    "m",
    "mme",
    "mlle",
    "mm",
    "n",
    "no",
    "nos",
    "pr",
    "st",
    "ste",
}
APOSTROPHE_ALIASES = {"’": "'", "‘": "'", "`": "'", "ʼ": "'"}


@dataclass(frozen=True)
class TextRow:
    corpus: str
    file_id: str
    paragraph_no: int
    text_start: int
    text_end: int
    text: str


def cell_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_text(text: str) -> str:
    """Normalize only spacing in exported TEXT/segment values."""
    return re.sub(r"\s+", " ", text.strip())


def strip_text(text: str) -> str:
    """Strip row boundaries while preserving internal characters for offsets."""
    return text.strip()


def unit_type(unit: ET.Element) -> str:
    type_node = unit.find("./characterisation/type")
    return (type_node.text or "").strip() if type_node is not None else ""


def single_position(unit: ET.Element) -> tuple[int, int] | None:
    start_node = unit.find("./positioning/start/singlePosition")
    end_node = unit.find("./positioning/end/singlePosition")
    if start_node is None or end_node is None:
        return None
    try:
        return int(start_node.get("index", "-1")), int(end_node.get("index", "-1"))
    except (TypeError, ValueError):
        return None


def paragraph_spans(aa_path: Path, raw_text: str) -> list[tuple[int, int]]:
    root = ET.parse(aa_path).getroot()
    spans: list[tuple[int, int]] = []
    for unit in root.findall(".//unit"):
        if unit_type(unit) != "paragraph":
            continue
        position = single_position(unit)
        if position is None:
            continue
        start, end = position
        if 0 <= start < end <= len(raw_text):
            spans.append((start, end))

    spans = sorted(set(spans))
    if not spans:
        return [(0, len(raw_text))] if raw_text else []

    # Keep non-overlapping paragraph spans, preserving the annotation order.
    filtered: list[tuple[int, int]] = []
    for start, end in spans:
        if filtered and start < filtered[-1][1]:
            previous_start, previous_end = filtered[-1]
            filtered[-1] = (previous_start, max(previous_end, end))
        else:
            filtered.append((start, end))

    return filtered


def is_decimal_or_thousands_dot(text: str, index: int) -> bool:
    return (
        text[index] == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def previous_token(text: str, index: int) -> str:
    left = text[:index].rstrip()
    match = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ-]+)$", left)
    return match.group(1).lower() if match else ""


def is_abbreviation_dot(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    token = previous_token(text, index)
    return token in NO_SPLIT_AFTER


def is_ellipsis(text: str, index: int) -> bool:
    return text.startswith("...", index) or text[index] == "…"


def next_non_space(text: str, index: int) -> str:
    pos = index
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return text[pos] if pos < len(text) else ""


def should_split_at(text: str, punct_start: int, punct_end: int) -> bool:
    if is_ellipsis(text, punct_start):
        # Avoid reproducing old exports that split "président... une pastèque"
        # or "seulement... 78 jours" into fragments.
        return punct_end >= len(text.rstrip())
    if is_decimal_or_thousands_dot(text, punct_start):
        return False
    if is_abbreviation_dot(text, punct_start):
        return False

    after = punct_end
    while after < len(text) and text[after] in SENTENCE_CLOSERS:
        after += 1
    if after >= len(text):
        return True
    if not text[after].isspace():
        return False

    following = next_non_space(text, after)
    if not following:
        return True
    return True


def trim_span(raw_text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(raw_text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Conservatively split one Glozz paragraph into textual rows."""
    start, end = trim_span(raw_text, start, end)
    if start >= end:
        return []

    spans: list[tuple[int, int]] = []
    current_start = start
    index = start
    while index < end:
        char = raw_text[index]
        if raw_text.startswith("...", index):
            punct_start, punct_end = index, min(index + 3, end)
        elif char in ".!?…":
            punct_start = index
            punct_end = index + 1
            while punct_end < end and raw_text[punct_end] in ".!?…":
                punct_end += 1
        else:
            index += 1
            continue

        if should_split_at(raw_text[start:end], punct_start - start, punct_end - start):
            absolute_end = punct_end
            while absolute_end < end and raw_text[absolute_end] in SENTENCE_CLOSERS:
                absolute_end += 1
            row_start, row_end = trim_span(raw_text, current_start, absolute_end)
            if row_start < row_end:
                spans.append((row_start, row_end))
            current_start = absolute_end
            while current_start < end and raw_text[current_start].isspace():
                current_start += 1
            index = current_start
            continue

        index = punct_end

    row_start, row_end = trim_span(raw_text, current_start, end)
    if row_start < row_end:
        spans.append((row_start, row_end))
    return spans


def parse_segments(value: Any) -> list[list[int]]:
    text = cell_text(value)
    if not text:
        return []
    return [[int(start), int(end)] for start, end in json.loads(text)]


def trim_segment(raw_text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1
    return start, end


def record_segments(record: pd.Series, raw_text: str | None = None) -> list[list[int]]:
    segments = parse_segments(record.get("segments"))
    if raw_text is None:
        return segments
    trimmed = []
    for start, end in segments:
        original_start, original_end = start, end
        start, end = trim_segment(raw_text, start, end)
        if start < end:
            trimmed.append([start, end])
            continue

        # A few Glozz units annotate only an exclamation mark but their stored
        # interval is empty or contains the preceding blank. Recover that mark.
        candidates = [original_start, original_end, original_start - 1, original_end - 1]
        for candidate in candidates:
            if 0 <= candidate < len(raw_text) and raw_text[candidate] in "!?":
                trimmed.append([candidate, candidate + 1])
                break
    return trimmed


def record_bounds(record: pd.Series, raw_text: str | None = None) -> tuple[int, int] | None:
    segments = record_segments(record, raw_text)
    if not segments:
        return None
    return min(start for start, _ in segments), max(end for _, end in segments)


def merge_boundaries_crossing_annotations(
    spans: list[tuple[int, int]],
    annotations: list[pd.Series],
    raw_text: str,
) -> list[tuple[int, int]]:
    if len(spans) <= 1:
        return spans

    boundaries = {end for _start, end in spans[:-1]}
    for record in annotations:
        bounds = record_bounds(record, raw_text)
        if bounds is None:
            continue
        ann_start, ann_end = bounds
        for boundary in list(boundaries):
            if ann_start < boundary < ann_end:
                boundaries.remove(boundary)

    merged: list[tuple[int, int]] = []
    current_start, current_end = spans[0]
    for next_start, next_end in spans[1:]:
        if current_end in boundaries:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
        else:
            current_end = next_end
    merged.append((current_start, current_end))
    return merged


def normalize_for_search(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(text):
        char = APOSTROPHE_ALIASES.get(char, char)
        normalized.append(char)
        mapping.append(index)
    return "".join(normalized), mapping


def find_part_offsets(window_text: str, part: str) -> list[tuple[int, int]]:
    occurrences: list[tuple[int, int]] = []
    start_at = 0
    while True:
        start = window_text.find(part, start_at)
        if start < 0:
            break
        occurrences.append((start, start + len(part)))
        start_at = start + 1
    if occurrences:
        return occurrences

    norm_window, window_map = normalize_for_search(window_text)
    norm_part, _part_map = normalize_for_search(part)
    start_at = 0
    while True:
        start = norm_window.find(norm_part, start_at)
        if start < 0:
            break
        end = start + len(norm_part)
        occurrences.append((window_map[start], window_map[end - 1] + 1))
        start_at = start + 1
    return occurrences


def declencheur_offsets(
    *,
    row_text: str,
    segment_offsets: list[list[int]],
    declencheur: str,
) -> list[list[int]] | None:
    parts = [part.strip() for part in re.split(r"[;+]", declencheur) if part.strip()]
    if not parts:
        return None

    windows = [(start, end, row_text[start:end]) for start, end in segment_offsets]
    offsets: list[list[int]] = []
    used: list[tuple[int, int]] = []
    for part in parts:
        found: list[tuple[int, int]] = []
        for window_start, _window_end, window_text in windows:
            for start, end in find_part_offsets(window_text, part):
                absolute = (window_start + start, window_start + end)
                if any(max(absolute[0], left) < min(absolute[1], right) for left, right in used):
                    continue
                found.append(absolute)
        if len(found) != 1:
            return None
        used.append(found[0])
        offsets.append([found[0][0], found[0][1]])
    return offsets


def load_emotion_annotations(glozz_dir: Path) -> pd.DataFrame:
    sys.path.insert(0, str(BASE_DIR))
    from glozz.glozz_parser import process_all_corpora

    corpus_dirs = {
        corpus_dir.name: str(corpus_dir)
        for corpus_dir in sorted(glozz_dir.iterdir())
        if corpus_dir.is_dir()
    }
    raw = process_all_corpora(corpus_dirs)
    if raw.empty:
        return raw
    return raw[raw["type"].isin(["SitEmo", "Autre"])].copy()


def load_raw_texts(glozz_dir: Path) -> dict[tuple[str, str], str]:
    raw_texts: dict[tuple[str, str], str] = {}
    for ac_path in sorted(glozz_dir.glob("*/ac/*.ac")):
        raw_texts[(ac_path.parent.parent.name, ac_path.stem)] = ac_path.read_text(encoding="utf-8")
    return raw_texts


def load_label_normalizers():
    sys.path.insert(0, str(BASE_DIR))
    from pipeline.emotion_taxonomy import normalize_emotion, normalize_mode

    return normalize_emotion, normalize_mode


def build_text_rows(
    glozz_dir: Path,
    annotations: pd.DataFrame,
) -> list[TextRow]:
    by_file: dict[tuple[str, str], list[pd.Series]] = {}
    if not annotations.empty:
        for _, record in annotations.iterrows():
            by_file.setdefault((record["corpus"], record["file_id"]), []).append(record)

    rows: list[TextRow] = []
    for ac_path in sorted(glozz_dir.glob("*/ac/*.ac")):
        corpus = ac_path.parent.parent.name
        file_id = ac_path.stem
        aa_path = ac_path.parent.parent / "aa" / f"{file_id}.aa"
        raw_text = ac_path.read_text(encoding="utf-8")
        paragraphs = paragraph_spans(aa_path, raw_text) if aa_path.exists() else [(0, len(raw_text))]

        for paragraph_no, (para_start, para_end) in enumerate(paragraphs, start=1):
            paragraph_annotations = [
                record
                for record in by_file.get((corpus, file_id), [])
                if (bounds := record_bounds(record, raw_text)) is not None
                and para_start <= bounds[0]
                and bounds[1] <= para_end
            ]
            spans = sentence_spans(raw_text, para_start, para_end)
            spans = merge_boundaries_crossing_annotations(spans, paragraph_annotations, raw_text)
            for text_start, text_end in spans:
                text = strip_text(raw_text[text_start:text_end])
                if text:
                    rows.append(
                        TextRow(
                            corpus=corpus,
                            file_id=file_id,
                            paragraph_no=paragraph_no,
                            text_start=text_start,
                            text_end=text_end,
                            text=text,
                        )
                    )
    return rows


def index_text_rows(rows: list[TextRow]) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    by_file: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for index, row in enumerate(rows):
        by_file.setdefault((row.corpus, row.file_id), []).append((row.text_start, row.text_end, index))
    for file_rows in by_file.values():
        file_rows.sort()
    return by_file


def containing_row(
    rows_by_file: dict[tuple[str, str], list[tuple[int, int, int]]],
    record: pd.Series,
    raw_text: str,
) -> int | None:
    bounds = record_bounds(record, raw_text)
    if bounds is None:
        return None
    ann_start, ann_end = bounds
    file_rows = rows_by_file.get((record["corpus"], record["file_id"]), [])
    starts = [start for start, _end, _index in file_rows]
    position = bisect.bisect_right(starts, ann_start)
    for start, end, index in reversed(file_rows[max(0, position - 3) : position + 1]):
        if start <= ann_start and ann_end <= end:
            return index
    return None


def unit_payload(
    record: pd.Series,
    row: TextRow,
    raw_text: str,
    normalize_emotion,
    normalize_mode,
) -> dict[str, Any]:
    segments = record_segments(record, raw_text)
    segment_offsets = [
        [max(start, row.text_start) - row.text_start, min(end, row.text_end) - row.text_start]
        for start, end in segments
        if max(start, row.text_start) < min(end, row.text_end)
    ]
    segment_text = " ".join(row.text[start:end].strip() for start, end in segment_offsets)

    record_type = cell_text(record.get("type"))
    if record_type == "Autre":
        mode = pd.NA
        emotion1 = "Autre"
        emotion2 = None
    else:
        mode = normalize_mode(record.get("mode")) or pd.NA
        emotion1 = normalize_emotion(record.get("categorie1"))
        emotion2 = normalize_emotion(record.get("categorie2"))
    declencheur = cell_text(record.get("declencheur"))
    decl_offsets = (
        declencheur_offsets(
            row_text=row.text,
            segment_offsets=segment_offsets,
            declencheur=declencheur,
        )
        if declencheur
        else None
    )

    return {
        "id": f"{record.get('file_id')}:{record.get('unit_id')}",
        "type": record_type or pd.NA,
        "mode": mode,
        "emotion1": emotion1 or pd.NA,
        "emotion2": emotion2 or pd.NA,
        "emotion3": pd.NA,
        "nature_linguistique": clean_text(record.get("nature")) if cell_text(record.get("nature")) else pd.NA,
        "segment_text": segment_text or pd.NA,
        "segment_offsets_json": json.dumps(segment_offsets, ensure_ascii=False),
        "segment_is_discontinuous": bool(record.get("is_discontinuous")),
        "declencheur_text": declencheur or pd.NA,
        "declencheur_offsets_json": (
            json.dumps(decl_offsets, ensure_ascii=False) if decl_offsets is not None else pd.NA
        ),
        "declencheur_is_discontinuous": (
            bool(decl_offsets is not None and len(decl_offsets) > 1) if declencheur else pd.NA
        ),
    }


def assign_units(
    rows: list[TextRow],
    annotations: pd.DataFrame,
    raw_texts: dict[tuple[str, str], str],
) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    normalize_emotion, normalize_mode = load_label_normalizers()
    rows_by_file = index_text_rows(rows)
    row_units: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(rows))}
    unassigned: list[str] = []

    for _, record in annotations.iterrows():
        raw_text = raw_texts.get((record["corpus"], record["file_id"]), "")
        row_index = containing_row(rows_by_file, record, raw_text)
        if row_index is None:
            unassigned.append(f"{record.get('file_id')}:{record.get('unit_id')}")
            continue
        payload = unit_payload(record, rows[row_index], raw_text, normalize_emotion, normalize_mode)
        bounds = record_bounds(record, raw_text)
        payload["_sort_key"] = (bounds[0], bounds[1], str(record.get("unit_id"))) if bounds else (0, 0, "")
        row_units[row_index].append(payload)

    for units in row_units.values():
        units.sort(key=lambda item: item["_sort_key"])
        for unit in units:
            unit.pop("_sort_key", None)
    return row_units, unassigned


def unit_columns(max_units: int) -> list[str]:
    suffixes = [
        "id",
        "type",
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
    ]
    return [
        f"{UNIT_PREFIX}_{unit_no}_{suffix}"
        for unit_no in range(1, max_units + 1)
        for suffix in suffixes
    ]


def resolve_max_units(value: str, row_units: dict[int, list[dict[str, Any]]]) -> int:
    if value == "auto":
        return max((len(units) for units in row_units.values()), default=0)
    max_units = int(value)
    if max_units < 0:
        raise ValueError("--max-units must be non-negative or 'auto'")
    return max_units


def build_output_dataframe(
    rows: list[TextRow],
    row_units: dict[int, list[dict[str, Any]]],
    max_units: int,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "idx": list(range(len(rows))),
            "corpus": [row.corpus for row in rows],
            "file_id": [row.file_id for row in rows],
            "paragraph_no": [row.paragraph_no for row in rows],
            "text_start": [row.text_start for row in rows],
            "text_end": [row.text_end for row in rows],
            "TEXT": [row.text for row in rows],
            "n_emotion_units": [len(row_units[index]) for index in range(len(rows))],
            "n_Sit_Emo_units": [
                sum(unit.get("type") == "SitEmo" for unit in row_units[index])
                for index in range(len(rows))
            ],
            "n_Autre_units": [
                sum(unit.get("type") == "Autre" for unit in row_units[index])
                for index in range(len(rows))
            ],
        }
    )

    empty_unit_columns = pd.DataFrame(
        {column: pd.NA for column in unit_columns(max_units)},
        index=output.index,
    )
    output = pd.concat([output, empty_unit_columns], axis=1)

    for row_index, units in row_units.items():
        for unit_no, unit in enumerate(units[:max_units], start=1):
            prefix = f"{UNIT_PREFIX}_{unit_no}"
            for suffix, value in unit.items():
                output.at[row_index, f"{prefix}_{suffix}"] = value
    return output


def compare_with_reference(output: pd.DataFrame, reference_xlsx: Path) -> dict[str, int]:
    reference = pd.read_excel(reference_xlsx, usecols=["TEXT"])
    generated_counter = Counter(output["TEXT"].astype(str))
    reference_counter = Counter(reference["TEXT"].astype(str))
    return {
        "reference_rows": int(len(reference)),
        "generated_rows": int(len(output)),
        "reference_unique_texts": int(len(reference_counter)),
        "generated_unique_texts": int(len(generated_counter)),
        "texts_missing_from_generated": int(sum((reference_counter - generated_counter).values())),
        "texts_extra_in_generated": int(sum((generated_counter - reference_counter).values())),
    }


def write_counter_examples(
    output: pd.DataFrame,
    reference_xlsx: Path,
    report_path: Path,
    *,
    limit: int = 200,
) -> None:
    reference = pd.read_excel(reference_xlsx, usecols=["TEXT"])
    generated_counter = Counter(output["TEXT"].astype(str))
    reference_counter = Counter(reference["TEXT"].astype(str))
    missing = list((reference_counter - generated_counter).elements())[:limit]
    extra = list((generated_counter - reference_counter).elements())[:limit]
    lines = ["# Missing from generated", *missing, "", "# Extra in generated", *extra]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    output: pd.DataFrame,
    row_units: dict[int, list[dict[str, Any]]],
    unassigned: list[str],
    max_units: int,
) -> dict[str, int]:
    n_units = sum(len(units) for units in row_units.values())
    n_truncated = sum(max(0, len(units) - max_units) for units in row_units.values())
    n_declencheurs = 0
    n_declencheurs_without_offsets = 0
    for units in row_units.values():
        for unit in units[:max_units]:
            if not pd.isna(unit.get("declencheur_text")):
                n_declencheurs += 1
                if pd.isna(unit.get("declencheur_offsets_json")):
                    n_declencheurs_without_offsets += 1
    return {
        "n_rows": int(len(output)),
        "n_rows_with_emotion_units": int((output["n_emotion_units"] > 0).sum()),
        "n_emotion_units": int(n_units),
        "n_unassigned_emotion_units": int(len(unassigned)),
        "max_units_per_row": int(max((len(units) for units in row_units.values()), default=0)),
        "n_exported_unit_slots": int(max_units),
        "n_truncated_units": int(n_truncated),
        "n_declencheurs": int(n_declencheurs),
        "n_declencheurs_without_offsets": int(n_declencheurs_without_offsets),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruit un XLSX Glozz avec colonnes Sit_Emo_unit_N_*."
    )
    parser.add_argument("--glozz-dir", type=Path, default=DEFAULT_GLOZZ_DIR)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--skip-xlsx", action="store_true", help="N'écrit pas le XLSX, utile pour les vérifications rapides en Parquet.")
    parser.add_argument(
        "--max-units",
        default="auto",
        help="Nombre de blocs Sit_Emo_unit_N à écrire, ou 'auto' pour ne rien tronquer.",
    )
    parser.add_argument("--unassigned-report", type=Path, default=None)
    parser.add_argument("--reference-xlsx", type=Path, default=None)
    parser.add_argument("--comparison-report", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    annotations = load_emotion_annotations(args.glozz_dir)
    raw_texts = load_raw_texts(args.glozz_dir)
    rows = build_text_rows(args.glozz_dir, annotations)
    row_units, unassigned = assign_units(rows, annotations, raw_texts)
    max_units = resolve_max_units(args.max_units, row_units)
    output = build_output_dataframe(rows, row_units, max_units)

    wrote_output = False
    if args.output_parquet is not None:
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(args.output_parquet, index=False)
        print(f"Écrit parquet : {args.output_parquet}")
        wrote_output = True

    if not args.skip_xlsx and args.output_xlsx is not None:
        args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        output.to_excel(args.output_xlsx, index=False, engine="openpyxl")
        print(f"Écrit XLSX : {args.output_xlsx}")
        wrote_output = True

    if not wrote_output:
        default_parquet = DEFAULT_OUTPUT_PARQUET
        default_parquet.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(default_parquet, index=False)
        print(f"Écrit parquet : {default_parquet}")

    if args.unassigned_report is not None:
        args.unassigned_report.parent.mkdir(parents=True, exist_ok=True)
        args.unassigned_report.write_text("\n".join(unassigned) + ("\n" if unassigned else ""), encoding="utf-8")

    for key, value in summarize(output, row_units, unassigned, max_units).items():
        print(f"{key}: {value}")

    if args.reference_xlsx is not None:
        comparison = compare_with_reference(output, args.reference_xlsx)
        print("Comparaison référence TEXT (audit seulement) :")
        for key, value in comparison.items():
            print(f"{key}: {value}")
        if args.comparison_report is not None:
            write_counter_examples(output, args.reference_xlsx, args.comparison_report)
            print(f"comparison_report: {args.comparison_report}")


if __name__ == "__main__":
    main()
