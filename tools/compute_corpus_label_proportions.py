#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute emotion and mode proportions for CyberAggAdo and TTK-Glozz.

The script deliberately reports two emotion proportions:

* ``emotion_assignment_share``: share among category assignments. This sums to
  1 per corpus over the 12 emotion categories, because multi-emotion units
  contribute one assignment per category.
* ``emotion_unit_presence``: share of annotation units carrying a category.
  This can sum to more than 1 because one unit can carry several categories.

Mode proportions are reported as ``mode_unit_share`` over units with a valid
mode. In TTK-Glozz, only SitEmo units have modes; the separate Glozz ``Autre``
units therefore contribute to emotion proportions but not to mode proportions.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.emotion_taxonomy import ALL_EMOTIONS, MODES, normalize_emotion, normalize_mode


DEFAULT_CYBER_XLSX = PROJECT_ROOT / "data" / "raw" / "xlsx" / "CyberAdoAgg_gold_global_total_latest.xlsx"
DEFAULT_GLOZZ_CSV = PROJECT_ROOT / "results" / "glozz" / "annotations.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "corpus_label_proportions.csv"

CYBER_SOURCE = "CyberAggAdo"
TTK_SOURCE = "TTK-Glozz"
MAX_SPANS = 4


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _parse_emotion_list(value: object, *, context: str) -> list[str]:
    """Parse one CyberAggAdo span category cell into canonical labels."""
    if _is_missing(value):
        return []
    raw = str(value).strip()
    if not raw:
        return []

    labels: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s*\+\s*", raw):
        emotion = normalize_emotion(part)
        if emotion is None:
            raise ValueError(f"Unknown emotion label in {context}: {part!r}")
        if emotion not in seen:
            labels.append(emotion)
            seen.add(emotion)
    return labels


def _required_mode(value: object, *, context: str) -> str:
    mode = normalize_mode(value)
    if mode is None:
        raise ValueError(f"Unknown or missing mode in {context}: {value!r}")
    return mode


def _load_cyber_units(xlsx_path: Path) -> pd.DataFrame:
    """Load CyberAggAdo as one row per SimpleSitEmo-like unit.

    This mirrors the logic of ``pipeline.build_simplesitemo_xlsx`` while keeping
    all valid emotion labels after duplicate-span merging.
    """
    df = pd.read_excel(xlsx_path)
    if "Emo" not in df.columns:
        raise ValueError(f"{xlsx_path} does not contain an 'Emo' column.")

    raw_units: list[dict] = []
    for row_idx, row in df[df["Emo"] == 1].iterrows():
        for span_idx in range(1, MAX_SPANS + 1):
            text = row.get(f"span{span_idx}_text")
            if _is_missing(text) or not str(text).strip():
                continue

            context = f"CyberAggAdo row={row_idx}, span={span_idx}"
            mode = _required_mode(row.get(f"span{span_idx}_mode"), context=context)
            emotions = _parse_emotion_list(row.get(f"span{span_idx}_cat"), context=context)
            if not emotions:
                raise ValueError(f"Missing emotion label in {context}.")

            raw_units.append(
                {
                    "source_row": row_idx,
                    "text_span": str(text).strip(),
                    "mode": mode,
                    "emotions": emotions,
                }
            )

    grouped: dict[tuple[object, str, str], list[str]] = defaultdict(list)
    for unit in raw_units:
        key = (unit["source_row"], unit["text_span"], unit["mode"])
        for emotion in unit["emotions"]:
            if emotion not in grouped[key]:
                grouped[key].append(emotion)

    records = [
        {
            "corpus": CYBER_SOURCE,
            "unit_type": "SimpleSitEmo",
            "mode": mode,
            "emotions": emotions,
        }
        for (_source_row, _text_span, mode), emotions in grouped.items()
    ]
    return pd.DataFrame(records)


def _load_ttk_units(glozz_csv: Path) -> pd.DataFrame:
    """Load TTK-Glozz annotations as SitEmo units plus separate Autre units."""
    df = pd.read_csv(glozz_csv)
    required = {"type", "mode", "categorie1", "categorie2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{glozz_csv} is missing columns: {sorted(missing)}")

    records: list[dict] = []
    for _, row in df.iterrows():
        unit_type = row.get("type")
        if unit_type == "SitEmo":
            emotions = []
            for col in ("categorie1", "categorie2"):
                value = row.get(col)
                emotion = normalize_emotion(value)
                if emotion is None and not _is_missing(value) and str(value).strip() not in {"", "Aucune"}:
                    raise ValueError(
                        f"Unknown emotion label in TTK-Glozz row={row.name}, {col}: {value!r}"
                    )
                if emotion is not None and emotion not in emotions:
                    emotions.append(emotion)
            mode = _required_mode(row.get("mode"), context=f"TTK-Glozz row={row.name}")
            if emotions:
                records.append(
                    {
                        "corpus": TTK_SOURCE,
                        "unit_type": "SitEmo",
                        "mode": mode,
                        "emotions": emotions,
                    }
                )
            else:
                raise ValueError(f"Missing SitEmo emotion label in TTK-Glozz row={row.name}.")
        elif unit_type == "Autre":
            records.append(
                {
                    "corpus": TTK_SOURCE,
                    "unit_type": "Autre",
                    "mode": None,
                    "emotions": ["Autre"],
                }
            )

    return pd.DataFrame(records)


def _iter_emotions(units: pd.DataFrame) -> Iterable[str]:
    for emotions in units["emotions"]:
        for emotion in emotions:
            yield emotion


def _add_emotion_assignment_rows(units: pd.DataFrame, corpus: str, rows: list[dict]) -> None:
    assignments = list(_iter_emotions(units))
    denominator = len(assignments)
    counts = pd.Series(assignments, dtype="object").value_counts().to_dict()

    for emotion in ALL_EMOTIONS:
        count = int(counts.get(emotion, 0))
        rows.append(
            {
                "corpus": corpus,
                "dimension": "emotion",
                "measure": "emotion_assignment_share",
                "label": emotion,
                "count": count,
                "denominator": denominator,
                "proportion": count / denominator if denominator else 0.0,
                "percent": (count / denominator * 100) if denominator else 0.0,
                "denominator_description": "all emotion category assignments; multi-label units contribute one assignment per category",
            }
        )


def _add_emotion_unit_rows(units: pd.DataFrame, corpus: str, rows: list[dict]) -> None:
    denominator = len(units)
    for emotion in ALL_EMOTIONS:
        count = int(units["emotions"].apply(lambda values: emotion in values).sum())
        rows.append(
            {
                "corpus": corpus,
                "dimension": "emotion",
                "measure": "emotion_unit_presence",
                "label": emotion,
                "count": count,
                "denominator": denominator,
                "proportion": count / denominator if denominator else 0.0,
                "percent": (count / denominator * 100) if denominator else 0.0,
                "denominator_description": "all emotion annotation units; multi-label units can be counted in several labels",
            }
        )


def _add_mode_rows(units: pd.DataFrame, corpus: str, rows: list[dict]) -> None:
    mode_units = units[units["mode"].isin(MODES)].copy()
    denominator = len(mode_units)
    counts = mode_units["mode"].value_counts().to_dict()

    for mode in MODES:
        count = int(counts.get(mode, 0))
        rows.append(
            {
                "corpus": corpus,
                "dimension": "mode",
                "measure": "mode_unit_share",
                "label": mode,
                "count": count,
                "denominator": denominator,
                "proportion": count / denominator if denominator else 0.0,
                "percent": (count / denominator * 100) if denominator else 0.0,
                "denominator_description": "all units with a valid mode; TTK-Glozz Autre units have no mode and are excluded",
            }
        )


def compute_proportions(cyber_xlsx: Path, glozz_csv: Path) -> pd.DataFrame:
    cyber_units = _load_cyber_units(cyber_xlsx)
    ttk_units = _load_ttk_units(glozz_csv)

    rows: list[dict] = []
    for corpus, units in ((CYBER_SOURCE, cyber_units), (TTK_SOURCE, ttk_units)):
        _add_emotion_assignment_rows(units, corpus, rows)
        _add_emotion_unit_rows(units, corpus, rows)
        _add_mode_rows(units, corpus, rows)

    result = pd.DataFrame(rows)
    result["proportion"] = result["proportion"].round(8)
    result["percent"] = result["percent"].round(4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute emotion-category and mode proportions for CyberAggAdo and TTK-Glozz."
    )
    parser.add_argument("--cyber-xlsx", type=Path, default=DEFAULT_CYBER_XLSX)
    parser.add_argument("--glozz-csv", type=Path, default=DEFAULT_GLOZZ_CSV)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = compute_proportions(args.cyber_xlsx, args.glozz_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Wrote {len(result)} rows to {args.output}")
    print(
        result.groupby(["corpus", "measure"])["denominator"]
        .first()
        .to_string()
    )


if __name__ == "__main__":
    main()
