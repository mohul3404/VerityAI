from __future__ import annotations

import os
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SAMPLE_DATASET = DATA_DIR / "sample_news.csv"
    ACTIVE_DATASET = PROCESSED_DATA_DIR / "active_dataset.csv"

    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "model.pkl"
    VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
    METRICS_PATH = MODEL_DIR / "metrics.json"

    REPORT_DIR = BASE_DIR / "reports"
    CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"
    FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.png"
    ROC_CURVE_PATH = REPORT_DIR / "roc_curve.png"
    PDF_REPORT_PATH = REPORT_DIR / "training_report.pdf"

    LOG_DIR = BASE_DIR / "logs"
    LOG_PATH = LOG_DIR / "app.log"
    STATS_PATH = MODEL_DIR / "usage_stats.json"

    SECRET_KEY = os.environ.get("SECRET_KEY", "development-only-change-before-deployment")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"csv"}
    MIN_ARTICLE_LENGTH = 20
    MAX_ARTICLE_LENGTH = 50_000
    TEST_SIZE = 0.30
    RANDOM_STATE = 42
    MAX_FEATURES = 12_000
    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", "8000"))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    @classmethod
    def ensure_directories(cls) -> None:
        for directory in (
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.REPORT_DIR,
            cls.LOG_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
