from __future__ import annotations

from dataclasses import dataclass, asdict

from services.preprocessing import Preprocessor


POSITIVE_WORDS = {
    "accurate", "achievement", "approved", "benefit", "confirmed", "credible",
    "effective", "evidence", "gain", "good", "growth", "healthy", "improved",
    "official", "peer", "positive", "preserve", "profit", "progress", "reliable",
    "research", "safe", "steady", "success", "transparent", "verified",
}

NEGATIVE_WORDS = {
    "alien", "anonymous", "banned", "conspiracy", "corrupt", "cure", "danger",
    "fake", "fraud", "hate", "hidden", "hoax", "magic", "miracle", "overnight",
    "panic", "scam", "secret", "shocking", "stunned", "unbelievable",
    "unverified", "viral", "weird",
}

SENSATIONAL_WORDS = {
    "always", "banned", "breaking", "conspiracy", "every", "exclusive",
    "exposed", "forever", "guaranteed", "guarantees", "immediately", "insane",
    "miracle", "never", "secret", "shocking", "stunned", "unbelievable", "viral",
}


@dataclass(frozen=True)
class SentimentResult:
    polarity: float
    subjectivity: float
    positive_terms: list[str]
    negative_terms: list[str]
    sensational_terms: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class SentimentAnalyzer:
    def analyze(self, text: object) -> SentimentResult:
        tokens = Preprocessor.tokenize(text)
        total = max(len(tokens), 1)
        positive = sorted({token for token in tokens if token in POSITIVE_WORDS})
        negative = sorted({token for token in tokens if token in NEGATIVE_WORDS})
        sensational = sorted({token for token in tokens if token in SENSATIONAL_WORDS})
        polarity = (len(positive) - len(negative)) / total
        subjectivity = (len(positive) + len(negative) + len(sensational)) / total
        return SentimentResult(
            polarity=round(max(-1.0, min(1.0, polarity)), 4),
            subjectivity=round(min(1.0, subjectivity), 4),
            positive_terms=positive,
            negative_terms=negative,
            sensational_terms=sensational,
        )
