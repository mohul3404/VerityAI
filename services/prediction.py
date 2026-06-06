from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import joblib
import numpy as np

from config import Config
from services.feature_engineering import FeatureEngineer
from services.preprocessing import Preprocessor
from services.training import ModelEvaluator, ModelTrainer, ReportGenerator

LOGGER = logging.getLogger(__name__)


class PredictionEngine:
    _pipeline = None
    _metrics: dict | None = None
    _model_mtime: float | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.features = FeatureEngineer()

    def ensure_model(self) -> None:
        if not Config.MODEL_PATH.exists():
            ModelTrainer().train()
        mtime = Config.MODEL_PATH.stat().st_mtime
        if self.__class__._pipeline is None or self.__class__._model_mtime != mtime:
            with self._lock:
                self.__class__._pipeline = joblib.load(Config.MODEL_PATH)
                self.__class__._metrics = json.loads(Config.METRICS_PATH.read_text(encoding="utf-8"))
                self.__class__._model_mtime = mtime

    def predict(self, headline: object, content: object) -> dict:
        self.ensure_model()
        headline_text = Preprocessor.sanitize_input(headline)
        content_text = Preprocessor.sanitize_input(content)
        article = f"{headline_text}. {content_text}".strip()
        probability = float(ModelEvaluator.probabilities(self._pipeline, [article])[0])
        prediction = int(probability >= 0.5)
        confidence = probability if prediction == 1 else 1.0 - probability
        insights = self.features.insights(article)
        explanation = self._explain(article)
        return {
            "prediction": "fake" if prediction else "real",
            "label": "FAKE NEWS" if prediction else "REAL NEWS",
            "confidence": round(confidence, 4),
            "fake_probability": round(probability, 4),
            "risk_score": round(probability * 100, 1),
            "model": self._metrics["selected_model"],
            "insights": insights,
            "explanation": explanation,
        }

    def metrics(self) -> dict:
        self.ensure_model()
        return dict(self._metrics or {})

    def _explain(self, article: str) -> dict:
        pipeline = self._pipeline
        transformed = pipeline.named_steps["features"].transform([article])
        classifier = pipeline.named_steps["classifier"]
        estimator = getattr(classifier, "estimator", classifier)
        names = list(pipeline.named_steps["features"].get_feature_names_out())
        row = transformed.toarray()[0]

        if hasattr(estimator, "coef_"):
            weights = np.asarray(estimator.coef_)[0]
            contributions = row * weights
        elif hasattr(estimator, "feature_importances_"):
            contributions = row * np.asarray(estimator.feature_importances_)
        else:
            top_names, top_values = ReportGenerator.feature_importance(pipeline, limit=10)
            return {
                "positive_indicators": [{"term": name, "influence": value} for name, value in zip(top_names, top_values)],
                "negative_indicators": [],
                "method": "global feature importance",
            }

        positive_indices = np.argsort(contributions)[::-1]
        negative_indices = np.argsort(contributions)
        positive = self._format_contributions(names, contributions, positive_indices, positive=True)
        negative = self._format_contributions(names, contributions, negative_indices, positive=False)
        return {
            "positive_indicators": positive,
            "negative_indicators": negative,
            "method": "local linear feature contribution",
        }

    @staticmethod
    def _format_contributions(
        names: list[str],
        contributions: np.ndarray,
        indices: np.ndarray,
        positive: bool,
    ) -> list[dict]:
        output = []
        for index in indices:
            value = float(contributions[index])
            if (positive and value <= 0) or (not positive and value >= 0):
                continue
            name = (
                names[index]
                .replace("tfidf__vectorizer__", "")
                .replace("tfidf__", "")
                .replace("language__", "")
            )
            output.append({"term": name, "influence": round(abs(value), 5)})
            if len(output) == 6:
                break
        return output


class UsageStats:
    _lock = threading.Lock()

    @classmethod
    def read(cls) -> dict:
        default = {"total_predictions": 0, "fake_predictions": 0, "real_predictions": 0}
        if not Config.STATS_PATH.exists():
            return default
        try:
            return {**default, **json.loads(Config.STATS_PATH.read_text(encoding="utf-8"))}
        except (OSError, json.JSONDecodeError):
            return default

    @classmethod
    def record(cls, label: str) -> dict:
        with cls._lock:
            stats = cls.read()
            stats["total_predictions"] += 1
            stats[f"{label}_predictions"] += 1
            Config.STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            return stats
