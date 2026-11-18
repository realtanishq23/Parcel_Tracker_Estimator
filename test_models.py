import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.model import DATA_PATH, FEATURES, train_models


class ParcelModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linear, cls.poisson, cls.metrics, cls.test_predictions = train_models()

    def test_dataset_has_required_columns_and_rows(self):
        data = pd.read_csv(DATA_PATH)
        self.assertGreaterEqual(len(data), 5000)
        self.assertTrue(set(FEATURES + [
            "continuous_delivery_days", "discrete_delivery_days"
        ]).issubset(data.columns))

    def test_metrics_are_finite(self):
        for model in ("linear", "poisson"):
            for metric in ("MAE", "RMSE", "R2"):
                self.assertTrue(np.isfinite(self.metrics[model][metric]))

    def test_models_return_positive_predictions(self):
        sample = pd.DataFrame([
            {"distance_km": 250.0, "package_weight_kg": 1.0, "courier_type": "Express"},
            {"distance_km": 900.0, "package_weight_kg": 8.0, "courier_type": "Standard"},
        ])
        linear_values = self.linear.predict(sample)
        poisson_values = self.poisson.predict(sample)
        self.assertTrue(np.isfinite(linear_values).all())
        self.assertTrue(np.isfinite(poisson_values).all())
        self.assertTrue((linear_values > 0).all())
        self.assertTrue((poisson_values > 0).all())

    def test_longer_distance_increases_estimate(self):
        sample = pd.DataFrame([
            {"distance_km": 100.0, "package_weight_kg": 2.0, "courier_type": "Standard"},
            {"distance_km": 1200.0, "package_weight_kg": 2.0, "courier_type": "Standard"},
        ])
        self.assertGreater(self.linear.predict(sample)[1], self.linear.predict(sample)[0])
        self.assertGreater(self.poisson.predict(sample)[1], self.poisson.predict(sample)[0])

    def test_holdout_predictions_match_test_size(self):
        self.assertEqual(len(self.test_predictions), self.metrics["test_rows"])
        self.assertEqual(self.metrics["train_rows"] + self.metrics["test_rows"],
                         self.metrics["total_rows"])


if __name__ == "__main__":
    unittest.main()
