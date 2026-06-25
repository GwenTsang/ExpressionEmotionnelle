"""Parsing de corpus annotés au format Glozz (.aa/.ac).

API publique :
    from glozz import process_all_corpora, parse_aa_ac_pair, process_corpus
"""

from .glozz_parser import (
    CORPUS_DIRS,
    TARGET_TYPES,
    export_to_csv,
    parse_aa_ac_pair,
    process_all_corpora,
    process_corpus,
)
from .remarque_parser import (
    parse_all_remarques,
    parse_remarques_from_pair,
    get_unique_remarques,
)
from .extract_remarques import (
    SentenceExtractor,
)

__all__ = [
    "CORPUS_DIRS",
    "TARGET_TYPES",
    "export_to_csv",
    "parse_aa_ac_pair",
    "process_all_corpora",
    "process_corpus",
    "parse_all_remarques",
    "parse_remarques_from_pair",
    "get_unique_remarques",
    "SentenceExtractor",
]

