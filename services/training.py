from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import Config
from services.feature_engineering import FeatureEngineer
from services.preprocessing import Preprocessor

LOGGER = logging.getLogger(__name__)


class DatasetService:
    TEXT_COLUMNS = ("title", "text", "content", "article", "statement", "body")
    LABEL_COLUMNS = ("label", "target", "class", "type")

    @classmethod
    def load(cls, path: Path) -> pd.DataFrame:
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            raise ValueError(f"Unable to read CSV: {error}") from error
        return cls.normalize(frame)

    @classmethod
    def normalize(cls, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("The CSV file is empty.")
        columns = {str(column).strip().lower(): column for column in frame.columns}
        label_column = next((columns[name] for name in cls.LABEL_COLUMNS if name in columns), None)
        text_columns = [columns[name] for name in cls.TEXT_COLUMNS if name in columns]
        if label_column is None:
            raise ValueError("CSV requires a label, target, class, or type column.")
        if not text_columns:
            raise ValueError("CSV requires title, text, content, article, statement, or body.")

        normalized = pd.DataFrame()
        if "title" in columns:
            normalized["title"] = frame[columns["title"]].fillna("").astype(str)
        else:
            normalized["title"] = ""
        normalized["text"] = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        normalized["label"] = frame[label_column].map(cls.normalize_label)
        normalized = normalized[normalized["text"].str.strip().str.len() >= 10]
        normalized = normalized.dropna(subset=["label"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
        if len(normalized) < 10:
            raise ValueError("Dataset needs at least 10 valid, unique articles.")
        if normalized["label"].nunique() != 2:
            raise ValueError("Dataset must contain both real and fake labels.")
        if normalized["label"].value_counts().min() < 3:
            raise ValueError("Each class needs at least 3 articles.")
        return normalized

    @staticmethod
    def normalize_label(value: Any) -> int:
        normalized = str(value).strip().lower()
        if normalized in {"fake", "false", "1", "unreliable", "fabricated"}:
            return 1
        if normalized in {"real", "true", "0", "reliable", "genuine"}:
            return 0
        raise ValueError(f"Unsupported label value: {value!r}")


class ModelPipeline:
    def __init__(self, max_features: int = Config.MAX_FEATURES) -> None:
        self.max_features = max_features

    def build(self, classifier: Any) -> Pipeline:
        tfidf = Pipeline([
            ("cleaner", Preprocessor()),
            ("vectorizer", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.98,
                max_features=self.max_features,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
        ])
        features = FeatureUnion([
            ("tfidf", tfidf),
            ("language", FeatureEngineer()),
        ])
        return Pipeline([
            ("features", features),
            ("classifier", classifier),
        ])


class ModelEvaluator:
    @staticmethod
    def evaluate(name: str, pipeline: Pipeline, x_test: list[str], y_test: list[int]) -> dict:
        predictions = pipeline.predict(x_test)
        probabilities = ModelEvaluator.probabilities(pipeline, x_test)
        return {
            "model": name,
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        }

    @staticmethod
    def probabilities(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            return pipeline.predict_proba(texts)[:, 1]
        scores = pipeline.decision_function(texts)
        return 1.0 / (1.0 + np.exp(-scores))


class ReportGenerator:
    def generate(
        self,
        metrics: dict,
        pipeline: Pipeline,
        x_test: list[str],
        y_test: list[int],
    ) -> None:
        self._plot_confusion_matrix(metrics["confusion_matrix"])
        self._plot_roc_curve(pipeline, x_test, y_test, metrics["roc_auc"])
        self._plot_feature_importance(pipeline)
        self._generate_pdf(metrics)

    @staticmethod
    def _plot_confusion_matrix(matrix: list[list[int]]) -> None:
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], labels=["Real", "Fake"])
        axis.set_yticks([0, 1], labels=["Real", "Fake"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_title("Confusion Matrix")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row][column], ha="center", va="center", fontsize=18)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(Config.CONFUSION_MATRIX_PATH, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _plot_roc_curve(pipeline: Pipeline, x_test: list[str], y_test: list[int], auc: float) -> None:
        probabilities = ModelEvaluator.probabilities(pipeline, x_test)
        false_positive, true_positive, _ = roc_curve(y_test, probabilities)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(false_positive, true_positive, color="#635bff", linewidth=2.5, label=f"AUC = {auc:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title("Receiver Operating Characteristic")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.18)
        figure.tight_layout()
        figure.savefig(Config.ROC_CURVE_PATH, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _plot_feature_importance(pipeline: Pipeline) -> None:
        names, values = ReportGenerator.feature_importance(pipeline, limit=15)
        figure, axis = plt.subplots(figsize=(8, 5.5))
        axis.barh(names[::-1], values[::-1], color="#635bff")
        axis.set_title("Top Model Features")
        axis.set_xlabel("Absolute influence")
        axis.grid(axis="x", alpha=0.18)
        figure.tight_layout()
        figure.savefig(Config.FEATURE_IMPORTANCE_PATH, dpi=180, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def feature_importance(pipeline: Pipeline, limit: int = 20) -> tuple[list[str], list[float]]:
        features = pipeline.named_steps["features"]
        classifier = pipeline.named_steps["classifier"]
        names = list(features.get_feature_names_out())
        estimator = getattr(classifier, "estimator", classifier)
        if hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_)[0])
        elif hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_)
        else:
            return names[:limit], [0.0] * min(limit, len(names))
        indices = np.argsort(values)[-limit:][::-1]
        clean_names = [
            names[index]
            .replace("tfidf__vectorizer__", "")
            .replace("tfidf__", "")
            .replace("language__", "")
            for index in indices
        ]
        return clean_names, [round(float(values[index]), 6) for index in indices]

    @staticmethod
    def _generate_pdf(metrics: dict) -> None:
        document = SimpleDocTemplate(
            str(Config.PDF_REPORT_PATH),
            pagesize=A4,
            rightMargin=42,
            leftMargin=42,
            topMargin=42,
            bottomMargin=42,
        )
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Fake News Detection - Training Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(
                f"Generated {metrics['trained_at']} | Selected model: {metrics['selected_model']}",
                styles["BodyText"],
            ),
            Spacer(1, 16),
        ]
        table_data = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]]
        for row in metrics["leaderboard"]:
            table_data.append([
                row["model"],
                f"{row['accuracy']:.3f}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}",
                f"{row['roc_auc']:.3f}",
            ])
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#171923")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([table, Spacer(1, 18)])
        for title, path in (
            ("Confusion Matrix", Config.CONFUSION_MATRIX_PATH),
            ("ROC Curve", Config.ROC_CURVE_PATH),
            ("Feature Importance", Config.FEATURE_IMPORTANCE_PATH),
        ):
            if path.exists():
                story.extend([
                    Paragraph(title, styles["Heading2"]),
                    Image(str(path), width=6.3 * inch, height=4.6 * inch),
                    Spacer(1, 12),
                ])
        document.build(story)


class ModelTrainer:
    def __init__(self) -> None:
        self.pipeline_factory = ModelPipeline()
        self.evaluator = ModelEvaluator()
        self.report_generator = ReportGenerator()

    def train(self, dataset_path: Path | None = None) -> dict:
        source = dataset_path or (Config.ACTIVE_DATASET if Config.ACTIVE_DATASET.exists() else Config.SAMPLE_DATASET)
        dataset = DatasetService.load(source)
        dataset.to_csv(Config.ACTIVE_DATASET, index=False)
        texts = dataset["text"].astype(str).tolist()
        labels = dataset["label"].astype(int).tolist()
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=labels,
        )

        candidates = self._candidate_models()
        trained: dict[str, Pipeline] = {}
        leaderboard: list[dict] = []
        for name, classifier in candidates.items():
            LOGGER.info("Training %s", name)
            pipeline = self.pipeline_factory.build(classifier)
            pipeline.fit(x_train, y_train)
            trained[name] = pipeline
            leaderboard.append(self.evaluator.evaluate(name, pipeline, x_test, y_test))

        leaderboard.sort(
            key=lambda item: (item["f1"], item["accuracy"], item["precision"], item["recall"]),
            reverse=True,
        )
        best = leaderboard[0]
        best_pipeline = trained[best["model"]]
        importance_names, importance_values = self.report_generator.feature_importance(best_pipeline)
        trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        metrics = {
            "selected_model": best["model"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
            "confusion_matrix": best["confusion_matrix"],
            "leaderboard": leaderboard,
            "dataset_size": len(dataset),
            "train_size": len(x_train),
            "test_size": len(x_test),
            "class_distribution": {
                "real": int((dataset["label"] == 0).sum()),
                "fake": int((dataset["label"] == 1).sum()),
            },
            "trained_at": trained_at,
            "dataset_file": source.name,
            "feature_importance": [
                {"feature": name, "importance": value}
                for name, value in zip(importance_names, importance_values)
            ],
        }
        joblib.dump(best_pipeline, Config.MODEL_PATH)
        vectorizer = (
            best_pipeline.named_steps["features"]
            .transformer_list[0][1]
            .named_steps["vectorizer"]
        )
        joblib.dump(vectorizer, Config.VECTORIZER_PATH)
        Config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self.report_generator.generate(metrics, best_pipeline, x_test, y_test)
        LOGGER.info("Selected %s with F1 %.4f", best["model"], best["f1"])
        return metrics

    @staticmethod
    def _candidate_models() -> dict[str, Any]:
        models: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1_500,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=220,
                max_depth=18,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
                n_jobs=1,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=12,
                class_weight="balanced",
                random_state=Config.RANDOM_STATE,
            ),
            "Linear SVM": CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", random_state=Config.RANDOM_STATE),
                cv=2,
            ),
            "Naive Bayes": ComplementNB(alpha=0.6),
        }
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=Config.RANDOM_STATE,
            )
        except ImportError:
            LOGGER.info("XGBoost is not installed; continuing with five classifiers.")
        return models
