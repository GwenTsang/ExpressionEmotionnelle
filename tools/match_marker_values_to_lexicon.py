#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match extracted marker values against the emotional lexicon.

The script first performs exact normalized matching, then proposes conservative
repair candidates for unmatched markers that look like truncated sub-words or
simple morphological variants.

Examples:
    python tools/match_marker_values_to_lexicon.py

    python tools/match_marker_values_to_lexicon.py \
        --markers results/simplesitemo/markers.csv \
        --lexicon emotions/lexique_emotionnel.tsv

    python tools/match_marker_values_to_lexicon.py \
        --global-markers results/simplesitemo_lexicon_matching/global_marker_value_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = BASE_DIR / "results" / "simplesitemo_lexicon_matching"
DEFAULT_MARKERS = BASE_DIR / "results" / "simplesitemo" / "markers.csv"
DEFAULT_LEXICON = BASE_DIR / "emotions" / "lexique_emotionnel.tsv"


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    category: str
    norm: str
    key: str


@dataclass(frozen=True)
class Candidate:
    marker_value: str
    lexicon_term: str
    lexicon_category: str
    repair_type: str
    confidence: str
    score: float
    levenshtein_distance: int
    shared_prefix_len: int
    sequence_ratio: float


def normalize_text(value: object) -> str:
    return str(value).strip().casefold()


def comparison_key(value: object, strip_accents: bool = True) -> str:
    text = normalize_text(value)
    if not strip_accents:
        return text
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def is_wordlike(value: object, min_len: int = 4) -> bool:
    text = str(value).strip()
    return len(text) >= min_len and any(char.isalpha() for char in text)


def common_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def levenshtein_bounded(left: str, right: str, max_distance: int | None = None) -> int:
    """Compute Levenshtein distance, stopping early when the threshold is exceeded."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left

    length_delta = len(left) - len(right)
    if max_distance is not None and length_delta > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, 1):
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            )
            current.append(value)
            if value < row_min:
                row_min = value
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def read_lexicon(path: Path, term_col: str | None, category_col: str | None) -> tuple[pd.DataFrame, list[LexiconEntry]]:
    if not path.is_file():
        raise FileNotFoundError(f"Lexicon not found: {path}")

    lexicon = pd.read_csv(path, sep="\t", keep_default_na=False, low_memory=False)
    if lexicon.empty:
        raise ValueError(f"Lexicon is empty: {path}")

    term_col = term_col or str(lexicon.columns[0])
    category_col = category_col or (str(lexicon.columns[1]) if len(lexicon.columns) > 1 else "")
    if term_col not in lexicon.columns:
        raise ValueError(f"Lexicon term column not found: {term_col}")
    if category_col and category_col not in lexicon.columns:
        raise ValueError(f"Lexicon category column not found: {category_col}")

    entries: list[LexiconEntry] = []
    for _, row in lexicon.iterrows():
        term = str(row[term_col]).strip()
        if not term:
            continue
        category = str(row[category_col]).strip() if category_col else ""
        entries.append(
            LexiconEntry(
                term=term,
                category=category,
                norm=normalize_text(term),
                key=comparison_key(term),
            )
        )
    return lexicon, entries


def build_global_counts_from_marker_files(
    spacy_path: Path,
    stanza_path: Path,
    marker_col: str,
    mode_col: str,
    mode_value: str,
) -> pd.DataFrame:
    spacy = read_csv(spacy_path)
    stanza = read_csv(stanza_path)

    for name, df, path in (("SpaCy", spacy, spacy_path), ("Stanza", stanza, stanza_path)):
        if marker_col not in df.columns:
            raise ValueError(f"{name} file has no '{marker_col}' column: {path}")

    def values_for(df: pd.DataFrame) -> list[str]:
        if mode_col in df.columns:
            selected = df[df[mode_col].astype(str).str.strip().eq(mode_value)]
        else:
            selected = df
        return [str(value) for value in selected[marker_col].tolist()]

    spacy_counts = Counter(values_for(spacy))
    stanza_counts = Counter(values_for(stanza))
    values = sorted(set(spacy_counts) | set(stanza_counts))
    rows = [
        {
            "marker_value": value,
            "spacy_count": spacy_counts.get(value, 0),
            "stanza_count": stanza_counts.get(value, 0),
        }
        for value in values
    ]
    global_counts = pd.DataFrame(rows)
    global_counts["total_count"] = global_counts["spacy_count"] + global_counts["stanza_count"]
    return global_counts.sort_values(["total_count", "marker_value"], ascending=[False, True])


def build_global_counts_from_markers(
    markers_path: Path,
    marker_col: str,
    mode_col: str,
    mode_value: str,
) -> pd.DataFrame:
    markers = read_csv(markers_path)
    if marker_col not in markers.columns:
        raise ValueError(f"Marker file has no '{marker_col}' column: {markers_path}")

    selected = markers
    if mode_col in markers.columns:
        selected = markers[markers[mode_col].astype(str).str.strip().eq(mode_value)]

    counts = Counter(str(value) for value in selected[marker_col].tolist())
    rows = [
        {
            "marker_value": value,
            "spacy_count": 0,
            "stanza_count": 0,
            "total_count": count,
        }
        for value, count in counts.items()
    ]
    global_counts = pd.DataFrame(rows)
    if global_counts.empty:
        return pd.DataFrame(columns=["marker_value", "spacy_count", "stanza_count", "total_count"])
    return global_counts.sort_values(["total_count", "marker_value"], ascending=[False, True])


def load_global_counts(args: argparse.Namespace) -> pd.DataFrame:
    if args.global_markers:
        global_counts = read_csv(Path(args.global_markers))
    elif args.spacy_markers and args.stanza_markers:
        global_counts = build_global_counts_from_marker_files(
            Path(args.spacy_markers),
            Path(args.stanza_markers),
            args.marker_col,
            args.mode_col,
            args.mode_value,
        )
    elif args.markers:
        global_counts = build_global_counts_from_markers(
            Path(args.markers),
            args.marker_col,
            args.mode_col,
            args.mode_value,
        )
    else:
        raise ValueError(
            "Provide --global-markers, --markers, or both --spacy-markers and --stanza-markers."
        )

    if args.marker_col not in global_counts.columns:
        raise ValueError(f"Global marker file has no '{args.marker_col}' column")

    for count_col in ("spacy_count", "stanza_count", "total_count"):
        if count_col not in global_counts.columns:
            global_counts[count_col] = 0

    global_counts["spacy_count"] = pd.to_numeric(global_counts["spacy_count"], errors="coerce").fillna(0).astype(int)
    global_counts["stanza_count"] = pd.to_numeric(global_counts["stanza_count"], errors="coerce").fillna(0).astype(int)
    if global_counts["total_count"].eq(0).all():
        global_counts["total_count"] = global_counts["spacy_count"] + global_counts["stanza_count"]
    else:
        global_counts["total_count"] = pd.to_numeric(global_counts["total_count"], errors="coerce").fillna(0).astype(int)

    return global_counts


def exact_annotate(
    markers: pd.DataFrame,
    marker_col: str,
    lexicon_entries: list[LexiconEntry],
    strip_accents: bool,
) -> pd.DataFrame:
    exact_by_key: dict[str, LexiconEntry] = {}
    exact_categories_by_key: dict[str, set[str]] = defaultdict(set)
    for entry in lexicon_entries:
        key = comparison_key(entry.term, strip_accents=strip_accents)
        exact_by_key.setdefault(key, entry)
        if entry.category:
            exact_categories_by_key[key].add(entry.category)

    annotated = markers.copy()
    annotated["match_key"] = annotated[marker_col].map(lambda value: comparison_key(value, strip_accents=strip_accents))
    annotated["lexicon_match_status"] = annotated["match_key"].map(
        lambda key: "exact" if key in exact_by_key else "no_match"
    )
    annotated["exact_lexicon_term"] = annotated["match_key"].map(
        lambda key: exact_by_key[key].term if key in exact_by_key else ""
    )
    annotated["exact_lexicon_category"] = annotated["match_key"].map(
        lambda key: "|".join(sorted(exact_categories_by_key[key])) if key in exact_categories_by_key else ""
    )
    annotated["in_lexique_emotionnel"] = annotated["lexicon_match_status"].eq("exact")
    return annotated


def build_length_index(entries: Iterable[LexiconEntry]) -> dict[int, list[LexiconEntry]]:
    by_len: dict[int, list[LexiconEntry]] = defaultdict(list)
    for entry in entries:
        by_len[len(entry.key)].append(entry)
    return by_len


def candidate_search_space(
    marker_key: str,
    entries_by_len: dict[int, list[LexiconEntry]],
    max_edit_distance: int,
    prefix_chars: int,
) -> Iterable[LexiconEntry]:
    min_len = max(1, len(marker_key) - max_edit_distance)
    max_len = len(marker_key) + max_edit_distance
    for length in range(min_len, max_len + 1):
        for entry in entries_by_len.get(length, []):
            if prefix_chars <= 0 or marker_key[:prefix_chars] == entry.key[:prefix_chars]:
                yield entry


def propose_candidates_for_marker(
    marker_value: str,
    entries: list[LexiconEntry],
    entries_by_len: dict[int, list[LexiconEntry]],
    *,
    min_word_len: int,
    min_prefix_len: int,
    high_prefix_ratio: float,
    max_edit_distance: int,
    min_sequence_ratio: float,
    fuzzy_prefix_chars: int,
    max_candidates: int,
) -> list[Candidate]:
    if not is_wordlike(marker_value, min_len=min_word_len):
        return []

    marker_key = comparison_key(marker_value)
    if len(marker_key) < min_word_len:
        return []

    candidates: list[Candidate] = []
    seen_terms: set[str] = set()

    def add_candidate(entry: LexiconEntry, repair_type: str, confidence: str, distance_limit: int | None = None) -> None:
        if entry.term in seen_terms:
            return
        distance = levenshtein_bounded(marker_key, entry.key, max_distance=distance_limit)
        prefix_len = common_prefix_len(marker_key, entry.key)
        sequence_ratio = SequenceMatcher(None, marker_key, entry.key).ratio()
        score = max(sequence_ratio, prefix_len / max(len(marker_key), len(entry.key)))
        candidates.append(
            Candidate(
                marker_value=marker_value,
                lexicon_term=entry.term,
                lexicon_category=entry.category,
                repair_type=repair_type,
                confidence=confidence,
                score=score,
                levenshtein_distance=distance,
                shared_prefix_len=prefix_len,
                sequence_ratio=sequence_ratio,
            )
        )
        seen_terms.add(entry.term)

    # Prefix cases catch truncated sub-words like "epouvant" -> "epouvante".
    for entry in entries:
        if len(entry.key) < min_word_len:
            continue
        if entry.key.startswith(marker_key) and len(entry.key) > len(marker_key) and len(marker_key) >= min_prefix_len:
            ratio = len(marker_key) / len(entry.key)
            confidence = "high" if ratio >= high_prefix_ratio else "medium"
            add_candidate(entry, "truncated_prefix_marker", confidence)
        elif marker_key.startswith(entry.key) and len(marker_key) > len(entry.key) and len(entry.key) >= min_prefix_len:
            ratio = len(entry.key) / len(marker_key)
            confidence = "high" if ratio >= high_prefix_ratio else "medium"
            add_candidate(entry, "inflected_or_extended_marker", confidence)

    # Bounded fuzzy cases are indexed by length and prefix to avoid all-pairs edits.
    for entry in candidate_search_space(marker_key, entries_by_len, max_edit_distance, fuzzy_prefix_chars):
        if len(entry.key) < min_word_len:
            continue
        prefix_len = common_prefix_len(marker_key, entry.key)
        min_len = min(len(marker_key), len(entry.key))
        distance = levenshtein_bounded(marker_key, entry.key, max_distance=max_edit_distance)
        if distance > max_edit_distance:
            continue
        sequence_ratio = SequenceMatcher(None, marker_key, entry.key).ratio()

        if prefix_len >= min_prefix_len and prefix_len >= min_len - 2 and distance <= max_edit_distance:
            confidence = "high" if sequence_ratio >= min_sequence_ratio else "medium"
            add_candidate(entry, "shared_long_stem_edit_distance", confidence, distance_limit=max_edit_distance)
        elif (
            marker_key[:fuzzy_prefix_chars] == entry.key[:fuzzy_prefix_chars]
            and distance <= max_edit_distance
            and sequence_ratio >= min_sequence_ratio
        ):
            add_candidate(entry, "same_prefix_fuzzy", "medium", distance_limit=max_edit_distance)

    candidates.sort(
        key=lambda item: (
            item.confidence == "high",
            item.score,
            -item.levenshtein_distance,
            item.shared_prefix_len,
            item.sequence_ratio,
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


def build_candidate_table(
    annotated: pd.DataFrame,
    marker_col: str,
    lexicon_entries: list[LexiconEntry],
    args: argparse.Namespace,
) -> pd.DataFrame:
    entries_by_len = build_length_index(lexicon_entries)
    rows: list[dict[str, object]] = []

    unmatched = annotated[~annotated["in_lexique_emotionnel"]].copy()
    unmatched = unmatched.sort_values(["total_count", marker_col], ascending=[False, True])

    for _, marker_row in unmatched.iterrows():
        marker_value = str(marker_row[marker_col])
        candidates = propose_candidates_for_marker(
            marker_value,
            lexicon_entries,
            entries_by_len,
            min_word_len=args.min_word_len,
            min_prefix_len=args.min_prefix_len,
            high_prefix_ratio=args.high_prefix_ratio,
            max_edit_distance=args.max_edit_distance,
            min_sequence_ratio=args.min_sequence_ratio,
            fuzzy_prefix_chars=args.fuzzy_prefix_chars,
            max_candidates=args.max_candidates,
        )
        for rank, candidate in enumerate(candidates, 1):
            rows.append(
                {
                    "marker_value": marker_value,
                    "spacy_count": int(marker_row.get("spacy_count", 0)),
                    "stanza_count": int(marker_row.get("stanza_count", 0)),
                    "total_count": int(marker_row.get("total_count", 0)),
                    "candidate_rank": rank,
                    "lexicon_term_candidate": candidate.lexicon_term,
                    "lexicon_category": candidate.lexicon_category,
                    "repair_type": candidate.repair_type,
                    "confidence": candidate.confidence,
                    "score": round(candidate.score, 4),
                    "levenshtein_distance_norm_noaccent": candidate.levenshtein_distance,
                    "shared_prefix_len_norm_noaccent": candidate.shared_prefix_len,
                    "sequence_ratio_norm_noaccent": round(candidate.sequence_ratio, 4),
                }
            )

    return pd.DataFrame(rows)


def apply_best_repairs(annotated: pd.DataFrame, candidates: pd.DataFrame, auto_confidence: str) -> pd.DataFrame:
    repaired = annotated.copy()
    repaired["best_candidate_term"] = ""
    repaired["best_candidate_category"] = ""
    repaired["best_candidate_confidence"] = ""
    repaired["best_candidate_repair_type"] = ""

    if candidates.empty:
        return repaired

    rank1 = candidates[candidates["candidate_rank"].eq(1)].copy()
    best_by_marker = {str(row["marker_value"]): row for _, row in rank1.iterrows()}
    auto_values = {"high"} if auto_confidence == "high" else {"high", "medium"}

    for index, row in repaired[~repaired["in_lexique_emotionnel"]].iterrows():
        marker = str(row["marker_value"])
        best = best_by_marker.get(marker)
        if best is None:
            continue
        repaired.at[index, "best_candidate_term"] = best["lexicon_term_candidate"]
        repaired.at[index, "best_candidate_category"] = best["lexicon_category"]
        repaired.at[index, "best_candidate_confidence"] = best["confidence"]
        repaired.at[index, "best_candidate_repair_type"] = best["repair_type"]
        if best["confidence"] in auto_values:
            repaired.at[index, "lexicon_match_status"] = f"repaired_{best['confidence']}"

    return repaired


def write_outputs(
    outdir: Path,
    annotated: pd.DataFrame,
    candidates: pd.DataFrame,
    repaired: pd.DataFrame,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "annotated": outdir / "global_marker_value_counts_with_lexicon_repair.csv",
        "candidates": outdir / "global_marker_values_not_in_lexique_repair_candidates.csv",
        "high_confidence": outdir / "global_marker_values_not_in_lexique_high_confidence_candidates.csv",
        "summary": outdir / "lexicon_repair_summary.csv",
    }

    repaired.to_csv(paths["annotated"], index=False, quoting=csv.QUOTE_MINIMAL)
    candidates.to_csv(paths["candidates"], index=False, quoting=csv.QUOTE_MINIMAL)
    high_confidence = candidates[candidates["confidence"].eq("high")] if not candidates.empty else candidates
    high_confidence.to_csv(paths["high_confidence"], index=False, quoting=csv.QUOTE_MINIMAL)

    summary_rows = []
    total_unique = len(repaired)
    total_occurrences = int(repaired["total_count"].sum()) if "total_count" in repaired.columns else 0
    for status, group in repaired.groupby("lexicon_match_status", dropna=False):
        summary_rows.append(
            {
                "lexicon_match_status": status,
                "unique_marker_values": len(group),
                "unique_marker_share": len(group) / total_unique if total_unique else 0,
                "total_count": int(group["total_count"].sum()) if "total_count" in group.columns else 0,
                "total_count_share": int(group["total_count"].sum()) / total_occurrences if total_occurrences else 0,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("lexicon_match_status")
    summary.to_csv(paths["summary"], index=False)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare marker_value terms with the emotional lexicon and propose "
            "high-confidence repairs for truncated or morphologically varied forms."
        )
    )
    parser.add_argument("--global-markers", default="", help="Optional CSV with precomputed marker_value counts.")
    parser.add_argument("--markers", default=str(DEFAULT_MARKERS), help="Normalized SimpleSitEmo markers CSV.")
    parser.add_argument("--spacy-markers", default="", help="Optional SpaCy markers CSV fallback.")
    parser.add_argument("--stanza-markers", default="", help="Optional Stanza markers CSV fallback.")
    parser.add_argument("--lexicon", default=str(DEFAULT_LEXICON), help="TSV emotional lexicon.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory.")
    parser.add_argument("--marker-col", default="marker_value", help="Marker value column.")
    parser.add_argument("--mode-col", default="mode", help="Mode column when building counts from marker files.")
    parser.add_argument("--mode-value", default="Désignée", help="Mode value to keep when building counts.")
    parser.add_argument("--lexicon-term-col", default=None, help="Lexicon term column. Default: first column.")
    parser.add_argument("--lexicon-category-col", default=None, help="Lexicon category column. Default: second column.")
    parser.add_argument("--keep-accents", action="store_true", help="Do not remove accents for matching keys.")
    parser.add_argument("--min-word-len", type=int, default=4, help="Minimum alphabetic marker length for repair candidates.")
    parser.add_argument("--min-prefix-len", type=int, default=5, help="Minimum shared prefix length for stem repairs.")
    parser.add_argument("--high-prefix-ratio", type=float, default=0.70, help="Prefix coverage threshold for high confidence.")
    parser.add_argument("--max-edit-distance", type=int, default=2, help="Maximum edit distance for fuzzy repair candidates.")
    parser.add_argument("--min-sequence-ratio", type=float, default=0.82, help="Minimum sequence ratio for high fuzzy confidence.")
    parser.add_argument("--fuzzy-prefix-chars", type=int, default=4, help="Required identical prefix length for fuzzy candidates.")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum candidates per unmatched marker.")
    parser.add_argument(
        "--auto-confidence",
        choices=("high", "medium"),
        default="high",
        help="Candidate confidence level that updates lexicon_match_status automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markers = load_global_counts(args)

    _, lexicon_entries = read_lexicon(Path(args.lexicon), args.lexicon_term_col, args.lexicon_category_col)
    annotated = exact_annotate(markers, args.marker_col, lexicon_entries, strip_accents=not args.keep_accents)
    candidates = build_candidate_table(annotated, args.marker_col, lexicon_entries, args)
    repaired = apply_best_repairs(annotated, candidates, args.auto_confidence)
    paths = write_outputs(Path(args.outdir), annotated, candidates, repaired)

    total_unique = len(repaired)
    total_count = int(repaired["total_count"].sum()) if "total_count" in repaired.columns else 0
    exact = repaired[repaired["lexicon_match_status"].eq("exact")]
    repaired_high = repaired[repaired["lexicon_match_status"].eq("repaired_high")]
    matched = repaired[repaired["lexicon_match_status"].isin(["exact", "repaired_high", "repaired_medium"])]

    print("Lexicon matching complete")
    print(f"Marker values: {total_unique:,}")
    print(f"Occurrences total_count: {total_count:,}")
    print(f"Exact matches: {len(exact):,} unique / {int(exact['total_count'].sum()):,} occurrences")
    print(
        "High-confidence repairs: "
        f"{len(repaired_high):,} unique / {int(repaired_high['total_count'].sum()):,} occurrences"
    )
    print(f"Matched after repairs: {len(matched):,} unique / {int(matched['total_count'].sum()):,} occurrences")
    print(f"Repair candidates: {len(candidates):,} rows for {candidates['marker_value'].nunique() if not candidates.empty else 0:,} markers")
    print("Wrote:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
