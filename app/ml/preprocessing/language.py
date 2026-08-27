"""Détection de langue légère et remplaçable."""

import re

from app.core.labels import Language

WORD_PATTERN = re.compile(r"[a-zàâçéèêëîïôûùüÿñ'-]+", re.IGNORECASE)
FRENCH_MARKERS = {"le", "la", "les", "des", "une", "dans", "pour", "avec", "est"}
MALAGASY_MARKERS = {"ny", "ary", "izany", "amin'ny", "ho", "dia", "tsy", "izay"}


class LanguageDetector:
    """Détecteur lexical minimal, conçu pour être remplacé ultérieurement."""

    def detect(self, text: str) -> Language:
        """Retourne une langue si les indices sont suffisants, sinon unknown."""
        words = {word.lower() for word in WORD_PATTERN.findall(text)}
        french_score = len(words & FRENCH_MARKERS)
        malagasy_score = len(words & MALAGASY_MARKERS)
        if french_score == 0 and malagasy_score == 0:
            return Language.UNKNOWN
        if french_score and malagasy_score:
            return Language.MIXED
        return Language.FR if french_score else Language.MG
