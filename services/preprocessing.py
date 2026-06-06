from __future__ import annotations

import html
import re
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class Preprocessor(BaseEstimator, TransformerMixin):
    """Deterministic, dependency-light NLP preprocessing for model pipelines."""

    TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    HTML_PATTERN = re.compile(r"<[^>]+>")
    SPACE_PATTERN = re.compile(r"\s+")
    SUFFIXES = (
        "ization", "ational", "fulness", "ousness", "iveness", "tional",
        "ingly", "edly", "ments", "ment", "ness", "able", "ible", "ing",
        "ies", "ied", "ed", "ly", "es", "s",
    )

    def fit(self, texts: Iterable[str], y: object = None) -> "Preprocessor":
        return self

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self.clean_text(text) for text in texts]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is None:
            return np.asarray(["text"], dtype=object)
        return np.asarray(input_features, dtype=object)

    @classmethod
    def sanitize_input(cls, text: object) -> str:
        value = html.unescape(str(text or ""))
        value = cls.HTML_PATTERN.sub(" ", value)
        value = cls.SPACE_PATTERN.sub(" ", value)
        return value.strip()

    @classmethod
    def clean_text(cls, text: object) -> str:
        value = cls.sanitize_input(text).lower()
        value = cls.URL_PATTERN.sub(" ", value)
        return " ".join(cls.tokenize(value))

    @classmethod
    def tokenize(cls, text: object) -> list[str]:
        tokens = cls.TOKEN_PATTERN.findall(str(text).lower())
        return [
            cls.stem(token)
            for token in tokens
            if len(token) > 2 and token not in ENGLISH_STOP_WORDS
        ]

    @classmethod
    def stem(cls, token: str) -> str:
        for suffix in cls.SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                if suffix == "ies":
                    return token[:-3] + "y"
                return token[:-len(suffix)]
        return token
