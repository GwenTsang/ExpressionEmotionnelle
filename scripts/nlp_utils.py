#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlp_utils.py — Outils de traitement du langage naturel partagés
Contient les listes de mots vides (stopwords), expressions régulières 
de tokenisation, et classes d'interface pour la lemmatisation (spaCy / Stanza).
"""

import re
import logging

logger = logging.getLogger(__name__)

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
    "qu", "quand", "comme", "si", "lorsque", "puisque", "quoique",
    
    # Adverbes fréquents et mots outils
    "ne", "pas", "plus", "moins", "très", "trop", "aussi", "bien", "mal", "alors", "ainsi", "toujours", "jamais",
    "ici", "là", "oui", "non", "tout", "tous", "toute", "toutes", "rien", "personne", "aucun", "autre", "autres",
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
    "c", "j", "m", "n", "s", "t", "y"
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
        try:
            import spacy
        except ImportError:
            raise RuntimeError(
                "spaCy n'est pas installé. "
                "Installez-le avec : pip install spacy && "
                "python -m spacy download fr_core_news_sm"
            )

        try:
            self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        except OSError:
            raise RuntimeError(
                "Modèle spaCy fr_core_news_sm introuvable. "
                "Installez-le avec : python -m spacy download fr_core_news_sm"
            )

        self.batch_size = batch_size
        logger.info(
            "Backend spaCy chargé (fr_core_news_sm, batch_size=%d)", batch_size
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


class StanzaBackend(LemmatizerBackend):
    """Backend Stanza avec GPU CUDA obligatoire et batch processing."""

    def __init__(self, batch_size: int = 256):
        # --- Vérification GPU CUDA ---
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch n'est pas installé. Stanza nécessite PyTorch avec CUDA. "
                "Installez-le avec : pip install torch"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ERREUR : GPU CUDA non détecté.\n"
                "Stanza ne peut être utilisé qu'avec un GPU CUDA.\n"
                f"  torch.cuda.is_available() = {torch.cuda.is_available()}\n"
                f"  torch.version.cuda = {torch.version.cuda}\n"
                "Utilisez --lemmatizer spacy pour une exécution CPU, "
                "ou vérifiez votre installation CUDA."
            )

        gpu_name = torch.cuda.get_device_name(0)
        logger.info("GPU CUDA détecté : %s", gpu_name)

        # --- Chargement de Stanza ---
        try:
            import stanza
        except ImportError:
            raise RuntimeError(
                "Stanza n'est pas installé. "
                "Installez-le avec : pip install stanza"
            )

        # Télécharger le modèle français si nécessaire
        try:
            stanza.download("fr", processors="tokenize,lemma", verbose=False)
        except Exception:
            pass  # Peut échouer si déjà installé, pas grave

        self.nlp = stanza.Pipeline(
            "fr",
            processors="tokenize,lemma",
            use_gpu=True,
            verbose=False,
        )
        self.batch_size = batch_size
        logger.info(
            "Backend Stanza chargé (fr, GPU CUDA, batch_size=%d)", batch_size
        )

    def lemmatize_batch(self, texts: list[str]) -> list[list[str]]:
        """Lemmatise un batch de textes via stanza bulk_process()."""
        # Stanza attend des Document ou des str. On filtre les vides.
        # bulk_process traite tous les textes en batch GPU
        import stanza

        # Découper en sous-batches pour ne pas saturer la VRAM
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Stanza bulk_process prend une liste de str ou de Document
            docs = [stanza.Document([], text=t) for t in batch]
            processed = self.nlp(docs)
            for doc in processed:
                lemmas = [
                    word.lemma.lower()
                    for sent in doc.sentences
                    for word in sent.words
                    if word.text.isalpha()
                ]
                all_results.append(lemmas)
        return all_results


def get_lemmatizer(
    backend_name: str = "spacy", batch_size: int = 256
) -> LemmatizerBackend:
    """Factory pour le backend de lemmatisation.

    Parameters
    ----------
    backend_name : str
        "spacy" ou "stanza"
    batch_size : int
        Taille du batch pour le traitement.

    Returns
    -------
    LemmatizerBackend
    """
    if backend_name == "spacy":
        return SpacyBackend(batch_size=batch_size)
    elif backend_name == "stanza":
        return StanzaBackend(batch_size=batch_size)
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
