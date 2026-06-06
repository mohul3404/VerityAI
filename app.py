from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config
from services.prediction import PredictionEngine, UsageStats
from services.preprocessing import Preprocessor
from services.training import DatasetService, ModelTrainer


def create_app() -> Flask:
    Config.ensure_directories()
    app = Flask(__name__)
    app.config.from_object(Config)
    configure_logging(app)
    engine = PredictionEngine()

    @app.get("/")
    def index():
        return render_template("index.html", active_page="home")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html", active_page="dashboard")

    @app.get("/upload")
    def upload():
        return render_template("upload.html", active_page="upload")

    @app.get("/reports")
    def reports():
        return render_template("reports.html", active_page="reports")

    @app.get("/reports/<path:filename>")
    def report_image(filename: str):
        allowed = {"confusion_matrix.png", "feature_importance.png", "roc_curve.png"}
        if filename not in allowed:
            return api_error("Report not found.", 404)
        path = Config.REPORT_DIR / filename
        if not path.exists():
            ModelTrainer().train()
        return send_file(path)

    @app.get("/api/status")
    def status():
        metrics = engine.metrics()
        return jsonify({"ready": True, "metrics": metrics, "usage": UsageStats.read()})

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        headline = Preprocessor.sanitize_input(payload.get("headline"))
        content = Preprocessor.sanitize_input(payload.get("content"))
        combined = f"{headline} {content}".strip()
        if len(combined) < Config.MIN_ARTICLE_LENGTH:
            return api_error(f"Enter at least {Config.MIN_ARTICLE_LENGTH} characters.", 400)
        if len(combined) > Config.MAX_ARTICLE_LENGTH:
            return api_error(f"Article exceeds {Config.MAX_ARTICLE_LENGTH:,} characters.", 400)
        result = engine.predict(headline, content)
        result["usage"] = UsageStats.record(result["prediction"])
        return jsonify(result)

    @app.post("/api/train")
    def train():
        path = Config.ACTIVE_DATASET if Config.ACTIVE_DATASET.exists() else Config.SAMPLE_DATASET
        metrics = ModelTrainer().train(path)
        return jsonify({"message": "Model training completed.", "metrics": metrics})

    @app.post("/api/upload")
    def upload_dataset():
        if "file" not in request.files:
            return api_error("Choose a CSV file to upload.", 400)
        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename or "")
        if not filename or "." not in filename or filename.rsplit(".", 1)[1].lower() not in Config.ALLOWED_EXTENSIONS:
            return api_error("Only CSV files are accepted.", 400)

        raw_path = Config.RAW_DATA_DIR / f"{uuid4().hex}_{filename}"
        uploaded.save(raw_path)
        try:
            dataset = DatasetService.load(raw_path)
            dataset.to_csv(Config.ACTIVE_DATASET, index=False)
            metrics = ModelTrainer().train(Config.ACTIVE_DATASET)
        except Exception:
            app.logger.exception("Dataset upload or training failed")
            raw_path.unlink(missing_ok=True)
            raise
        return jsonify({
            "message": f"{len(dataset):,} validated rows imported and trained.",
            "metrics": metrics,
        })

    @app.get("/api/dataset")
    def dataset():
        path = Config.ACTIVE_DATASET if Config.ACTIVE_DATASET.exists() else Config.SAMPLE_DATASET
        frame = DatasetService.load(path)
        query = Preprocessor.sanitize_input(request.args.get("search", "")).lower()
        sort = request.args.get("sort", "title")
        direction = request.args.get("direction", "asc")
        page = max(int(request.args.get("page", 1)), 1)
        page_size = min(max(int(request.args.get("page_size", 8)), 5), 50)
        if query:
            frame = frame[
                frame["title"].str.lower().str.contains(query, regex=False)
                | frame["text"].str.lower().str.contains(query, regex=False)
                | frame["label"].astype(str).str.contains(query, regex=False)
            ]
        if sort not in {"title", "label", "text"}:
            sort = "title"
        frame = frame.sort_values(sort, ascending=direction != "desc")
        total = len(frame)
        start = (page - 1) * page_size
        records = frame.iloc[start:start + page_size].copy()
        records["label"] = records["label"].map({0: "real", 1: "fake"})
        records["text"] = records["text"].str.slice(0, 240)
        return jsonify({
            "items": records.to_dict(orient="records"),
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": max((total + page_size - 1) // page_size, 1),
        })

    @app.get("/downloads/<artifact>")
    def download_artifact(artifact: str):
        artifacts = {
            "metrics": (Config.METRICS_PATH, "application/json"),
            "confusion-matrix": (Config.CONFUSION_MATRIX_PATH, "image/png"),
            "feature-importance": (Config.FEATURE_IMPORTANCE_PATH, "image/png"),
            "roc-curve": (Config.ROC_CURVE_PATH, "image/png"),
            "training-report": (Config.PDF_REPORT_PATH, "application/pdf"),
        }
        if artifact not in artifacts:
            return api_error("Report not found.", 404)
        path, mime = artifacts[artifact]
        if not path.exists():
            ModelTrainer().train()
        return send_file(path, mimetype=mime, as_attachment=True, download_name=path.name)

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(error):
        return api_error("File exceeds the 10 MB upload limit.", 413)

    @app.errorhandler(ValueError)
    def validation_error(error):
        return api_error(str(error), 400)

    @app.errorhandler(Exception)
    def unexpected_error(error):
        if isinstance(error, HTTPException):
            return api_error(error.description, error.code or 500)
        app.logger.exception("Unhandled application error")
        return api_error("The request could not be completed. Check the application log.", 500)

    return app


def api_error(message: str, status: int):
    return jsonify({"error": message}), status


def configure_logging(app: Flask) -> None:
    handler = RotatingFileHandler(Config.LOG_PATH, maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    logging.getLogger("services").addHandler(handler)
    logging.getLogger("services").setLevel(logging.INFO)


app = create_app()


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
