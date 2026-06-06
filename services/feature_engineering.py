from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from services.preprocessing import Preprocessor
from services.sentiment import SentimentAnalyzer, SENSATIONAL_WORDS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    FEATURE_NAMES = [
        "sentiment_polarity",
        "subjectivity",
        "capital_letter_ratio",
        "exclamation_count",
        "question_mark_count",
        "sensational_keyword_count",
        "average_word_length",
        "article_length",
    ]

    def __init__(self) -> None:
        self.sentiment = SentimentAnalyzer()

    def fit(self, texts: Iterable[str], y: object = None) -> "FeatureEngineer":
        return self

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        rows = [self.extract(text) for text in texts]
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.FEATURE_NAMES, dtype=object)

    def extract(self, text: object) -> list[float]:
        raw = Preprocessor.sanitize_input(text)
        words = re.findall(r"[A-Za-z]+", raw)
        tokens = Preprocessor.tokenize(raw)
        sentiment = self.sentiment.analyze(raw)
        letters = [character for character in raw if character.isalpha()]
        uppercase = sum(character.isupper() for character in letters)
        average_word_length = sum(len(word) for word in words) / max(len(words), 1)
        sensational_count = sum(token in SENSATIONAL_WORDS for token in tokens)
        return [
            (sentiment.polarity + 1.0) / 2.0,
            sentiment.subjectivity,
            uppercase / max(len(letters), 1),
            min(raw.count("!") / 10.0, 1.0),
            min(raw.count("?") / 10.0, 1.0),
            min(sensational_count / 10.0, 1.0),
            min(average_word_length / 15.0, 1.0),
            min(len(words) / 2_000.0, 1.0),
        ]

    def insights(self, text: object) -> dict:
        raw = Preprocessor.sanitize_input(text)
        words = re.findall(r"[A-Za-z]+", raw)
        sentences = max(len(re.findall(r"[.!?]+", raw)), 1)
        syllables = sum(self._estimate_syllables(word) for word in words)
        word_count = len(words)
        readability = 206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllables / max(word_count, 1))
        sentiment = self.sentiment.analyze(raw)
        values = self.extract(raw)
        return {
            "word_count": word_count,
            "character_count": len(raw),
            "sentence_count": sentences,
            "readability_score": round(max(0.0, min(100.0, readability)), 1),
            "average_word_length": round(
                sum(len(word) for word in words) / max(word_count, 1), 2
            ),
            "capital_letter_ratio": round(values[2], 4),
            "exclamation_count": raw.count("!"),
            "question_mark_count": raw.count("?"),
            **sentiment.to_dict(),
        }

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        groups = re.findall(r"[aeiouy]+", word.lower())
        count = len(groups)
        if word.lower().endswith("e") and count > 1:
            count -= 1
        return max(count, 1)
