from __future__ import annotations

import unittest

from app import app
from services.feature_engineering import FeatureEngineer
from services.preprocessing import Preprocessor
from services.training import DatasetService


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_pages_load(self) -> None:
        for path in ("/", "/dashboard", "/upload", "/reports"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_status_and_dataset_apis(self) -> None:
        status = self.client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json["ready"])
        dataset = self.client.get("/api/dataset?page=1&page_size=5")
        self.assertEqual(dataset.status_code, 200)
        self.assertLessEqual(len(dataset.json["items"]), 5)

    def test_prediction_validation(self) -> None:
        response = self.client.post("/api/predict", json={"headline": "Short", "content": ""})
        self.assertEqual(response.status_code, 400)

    def test_prediction_response(self) -> None:
        response = self.client.post("/api/predict", json={
            "headline": "Secret miracle cure exposed",
            "content": "A viral post guarantees an overnight cure without named researchers or clinical evidence.",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json["prediction"], {"real", "fake"})
        self.assertIn("fake_probability", response.json)
        self.assertIn("explanation", response.json)


class ServiceTests(unittest.TestCase):
    def test_preprocessor_removes_markup_and_stop_words(self) -> None:
        cleaned = Preprocessor.clean_text("<b>The</b> VERIFIED report is available at https://example.com")
        self.assertNotIn("https", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertIn("verif", cleaned)

    def test_feature_engineer_returns_expected_features(self) -> None:
        values = FeatureEngineer().extract("SHOCKING secret claim! Is this real?")
        self.assertEqual(len(values), 8)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_label_normalization(self) -> None:
        self.assertEqual(DatasetService.normalize_label("fake"), 1)
        self.assertEqual(DatasetService.normalize_label("real"), 0)


if __name__ == "__main__":
    unittest.main()
