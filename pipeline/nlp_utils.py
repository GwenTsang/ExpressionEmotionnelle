#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import re
from multiprocessing import get_context

# Regex pour la tokenisation
RE_WORD = re.compile(r"\b[a-zA-ZÀ-ÿœŒæÆ]+(?:['-][a-zA-ZÀ-ÿœŒæÆ]+)*\b", re.UNICODE)
RE_PUNCT = re.compile(r"[!?.,;:…\-—–\"'«»()\[\]]+")

# Liste de stopwords pour filtrage des marqueurs
FR_STOPWORDS = {
    # Pronoms et déterminants
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses",
    "notre", "votre", "leur", "nos", "vos", "leurs",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "y", "en", "eux", "moi", "toi",
    "qui", "que", "quoi", "dont", "où", "lequel", "auquel", "duquel", "laquelle", "lesquels", "lesquelles",
    "ceci", "cela", "ça", "celui", "celle", "ceux", "celles",
    
    # Prépositions et conjonctions
    "à", "pour", "sur", "dans", "avec", "par", "vers", "sous", "sans", "chez", "entre", "depuis",
    "et", "ou", "ni", "mais", "or", "car", "donc",
    "qu", "quand", "comme", "si", "lorsque", "puisque",
    
    # Adverbes fréquents et mots outils
    "ne", "pas", "plus", "moins", "très", "trop", "aussi", "bien", "mal", "alors", "ainsi", "toujours",
    "ici", "là", "oui", "non", "tout", "tous", "toute", "toutes", "personne", "autre", "autres",
    "même", "quelque", "quelques",
    
    # Verbes très fréquents (formes et lemmes)
    "être", "suis", "es", "est", "sommes", "êtes", "sont", "été", "étais", "était", "étions", "étiez", "étaient", "serai", "sera", "serons", "serez", "seront",
    "avoir", "ai", "as", "a", "avons", "avez", "ont", "eu", "avais", "avait", "avions", "aviez", "avaient", "aurai", "aura", "aurons", "aurez", "auront",
    "aller", "vais", "vas", "va", "allons", "allez", "vont",
    "faire", "fais", "fait", "faisons", "faites", "font",
    "pouvoir", "peux", "peut", "pouvons", "pouvez", "peuvent",
    "vouloir", "veux", "veut", "voulons", "voulez", "veulent",
    "devoir", "dois", "doit", "devons", "devez", "doivent",
    "dire", "dis", "dit", "disons", "dites", "disent",
    
    # Bruit additionnel (lettres isolées)
    "c", "j", "m", "n", "s", "t", "y", "soi"
}

# ---------------------------------------------------------------------------
# Backends de lemmatisation
# ---------------------------------------------------------------------------

class LemmatizerBackend:
    """Interface commune pour les backends de lemmatisation."""

    def lemmatize_batch(self, texts: list[str]) -> list[list[str]]:
        """Lemmatise une liste de textes. Retourne une liste de listes de lemmes."""
        raise NotImplementedError


class SpacyBackend(LemmatizerBackend):
    """Backend spaCy avec batch processing via nlp.pipe()."""

    def __init__(self, batch_size: int = 256):
        import spacy

        self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        self.batch_size = batch_size
        print(
            f"Backend spaCy chargé (fr_core_news_sm, batch_size={batch_size})"
        )

    def lemmatize_batch(self, texts: list[str]) -> list[list[str]]:
        """Lemmatise un batch de textes via nlp.pipe()."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
            lemmas = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_space
            ]
            results.append(lemmas)
        return results


_STANZA_PIPELINE = None
_STANZA_BATCH_SIZE = 512


def _configure_cpu_threads() -> None:
    """Limit per-process native thread pools before loading torch/Stanza."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass




def _build_stanza_pipeline():
    import stanza

    return stanza.Pipeline(
        "fr",
        processors="tokenize,lemma",
        use_gpu=False,
        verbose=False,
    )


def _lemmatize_with_stanza_pipeline(nlp, texts: list[str], batch_size: int) -> list[list[str]]:
    import stanza

    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        docs = [stanza.Document([], text=t) for t in batch]
        if hasattr(nlp, "bulk_process"):
            processed = nlp.bulk_process(docs)
        else:
            processed = nlp(docs)
        for doc in processed:
            lemmas = [
                word.lemma.lower()
                for sent in doc.sentences
                for word in sent.words
                if word.text.isalpha() and word.lemma
            ]
            all_results.append(lemmas)
    return all_results


def _init_stanza_worker(batch_size: int) -> None:
    global _STANZA_PIPELINE, _STANZA_BATCH_SIZE

    _configure_cpu_threads()
    _STANZA_BATCH_SIZE = batch_size
    _STANZA_PIPELINE = _build_stanza_pipeline()


def _lemmatize_stanza_chunk(texts: list[str]) -> list[list[str]]:
    if _STANZA_PIPELINE is None:
        raise RuntimeError("Pipeline Stanza non initialisé dans le worker.")
    return _lemmatize_with_stanza_pipeline(_STANZA_PIPELINE, texts, _STANZA_BATCH_SIZE)


class StanzaBackend(LemmatizerBackend):
    """Backend Stanza CPU-only avec batch processing et parallélisme optionnel."""

    def __init__(self, batch_size: int = 256, n_jobs: int = 1):
        if n_jobs < 1:
            raise ValueError("n_jobs doit être >= 1")

        _configure_cpu_threads()

        # --- Chargement de Stanza ---
        try:
            import stanza
        except ImportError:
            raise RuntimeError(
                "Stanza n'est pas installé. "
            )
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.nlp = None if n_jobs > 1 else _build_stanza_pipeline()
        print(f"Backend Stanza chargé (fr, CPU, batch_size={batch_size}, n_jobs={n_jobs})")

    def lemmatize_batch(self, texts: list[str]) -> list[list[str]]:
        """Lemmatise un batch de textes via Stanza sur CPU."""
        if self.n_jobs == 1:
            return _lemmatize_with_stanza_pipeline(self.nlp, texts, self.batch_size)

        chunk_size = max(self.batch_size, math.ceil(len(texts) / self.n_jobs))
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=self.n_jobs,
            initializer=_init_stanza_worker,
            initargs=(self.batch_size,),
        ) as pool:
            chunk_results = pool.map(_lemmatize_stanza_chunk, chunks)
        return [lemmas for chunk in chunk_results for lemmas in chunk]


def get_lemmatizer(
    backend_name,
    batch_size,
    n_jobs
) -> LemmatizerBackend:
    """Factory pour le backend de lemmatisation.

    Parameters
    ----------
    backend_name : str
        "spacy" ou "stanza"
    batch_size : int
        Taille du batch pour le traitement.
    n_jobs : int
        Nombre de processus CPU pour Stanza. Ignoré par spaCy.

    Returns
    -------
    LemmatizerBackend
    """
    if backend_name == "spacy":
        return SpacyBackend(batch_size=batch_size)
    elif backend_name == "stanza":
        return StanzaBackend(batch_size=batch_size, n_jobs=n_jobs)
    else:
        raise ValueError(f"Backend inconnu : {backend_name}. Choix : spacy, stanza")


# ---------------------------------------------------------------------------
# Extraction des marqueurs
# ---------------------------------------------------------------------------

def extract_words(text: str) -> list[str]:
    """Extrait les mots (tokens alphabétiques) en minuscules."""
    if not text or not isinstance(text, str):
        return []
    return [m.lower() for m in RE_WORD.findall(text)]


def extract_punctuations(text: str) -> list[str]:
    """Extrait les signes de ponctuation individuels."""
    if not text or not isinstance(text, str):
        return []
    # On sépare chaque caractère de ponctuation individuellement
    puncts = []
    for match in RE_PUNCT.finditer(text):
        group = match.group()
        # Traiter les points de suspension comme un seul marqueur
        if "…" in group:
            puncts.append("…")
            group = group.replace("…", "")
        # Séquences de points (... → …)
        while "..." in group:
            puncts.append("…")
            group = group.replace("...", "", 1)
        # Chaque caractère de ponctuation restant
        for ch in group:
            if ch in "!?.,;:—–\"'«»()[]!-":
                puncts.append(ch)
    return puncts
