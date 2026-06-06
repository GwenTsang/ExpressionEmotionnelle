"""Parsing de corpus annotés au format Glozz (.aa/.ac).

API publique :
    from glozz import process_all_corpora, parse_aa_ac_pair, process_corpus

Ce package est volontairement découplé des pipelines d'analyse : il ne
connaît que le format Glozz et produit des DataFrames structurés.
"""

from .glozz_parser import (
    CORPUS_DIRS,
    TARGET_TYPES,
    export_to_csv,
    parse_aa_ac_pair,
    process_all_corpora,
    process_corpus,
)

__all__ = [
    "CORPUS_DIRS",
    "TARGET_TYPES",
    "export_to_csv",
    "parse_aa_ac_pair",
    "process_all_corpora",
    "process_corpus",
]
