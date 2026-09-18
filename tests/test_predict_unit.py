"""
Unit test for ML_ENGINE/predict.py:
Feed known malware feature rows from local/ml/artifacts/X_test.csv where y_test = 0
and verify MALWARE verdict.
"""

import os
import sys
import unittest
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ML_ENGINE.predict import MalwarePredictor


class TestPredictUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MalwarePredictor()

        # Paths to test artifacts
        cls.x_test_path = os.path.join(PROJECT_ROOT, "local", "ml", "artifacts", "X_test.csv")
        cls.y_test_path = os.path.join(PROJECT_ROOT, "local", "ml", "artifacts", "y_test.csv")

        cls.assertTrue(
            os.path.isfile(cls.x_test_path),
            f"X_test.csv not found at: {cls.x_test_path}"
        )
        cls.assertTrue(
            os.path.isfile(cls.y_test_path),
            f"y_test.csv not found at: {cls.y_test_path}"
        )

        cls.X_test = pd.read_csv(cls.x_test_path)
        cls.y_test = pd.read_csv(cls.y_test_path)

        # In this dataset: 0 = Malware, 1 = Legitimate
        cls.malware_indices = cls.y_test[cls.y_test.iloc[:, 0] == 0].index.tolist()
        print(f"Total malware samples in X_test (y_test == 0): {len(cls.malware_indices)}")

    def test_known_malware_samples(self):
        print("\n" + "=" * 75)
        print("UNIT TEST 2: ML_ENGINE/predict.py on Known Malware (y_test == 0)")
        print("=" * 75)

        # Test first 20 malware samples with detailed logging and SHAP
        test_sample_count = 20
        sample_indices = self.malware_indices[:test_sample_count]

        malware_count = 0
        for i, idx in enumerate(sample_indices):
            row_dict = self.X_test.iloc[idx].to_dict()
            result = self.predictor.predict(row_dict, explain=True)

            verdict = result["verdict"]
            p_mal = result["malware_probability"]
            risk = result["risk_level"]
            factors = result.get("top_risk_factors", [])

            self.assertEqual(
                verdict,
                "MALWARE",
                f"Row {idx} expected MALWARE but got {verdict} (p={p_mal})"
            )
            self.assertTrue(
                result["is_malware"],
                f"Row {idx} is_malware expected True"
            )
            self.assertGreaterEqual(
                p_mal,
                self.predictor.operating_threshold,
                f"Row {idx} probability {p_mal} below threshold {self.predictor.operating_threshold}"
            )
            self.assertGreaterEqual(
                len(factors),
                1,
                f"Row {idx} missing SHAP top risk factors"
            )

            malware_count += 1
            top_factor_str = f"{factors[0]['feature']}: impact={factors[0]['impact']:.3f}" if factors else "none"
            print(f"  [Sample {i+1:2d} (Row {idx:5d})] Verdict: {verdict} | P(mal): {p_mal:.4f} | Risk: {risk:<16} | Top Driver: {top_factor_str}")

        print(f"\nSuccessfully verified {malware_count}/{test_sample_count} samples classified as MALWARE.")

        # Batch test on 200 random malware samples to verify high recall
        import random
        random.seed(42)
        batch_indices = random.sample(self.malware_indices, min(200, len(self.malware_indices)))
        batch_dicts = [self.X_test.iloc[idx].to_dict() for idx in batch_indices]
        batch_results = self.predictor.predict(batch_dicts, explain=False)

        batch_detected = sum(r["verdict"] == "MALWARE" for r in batch_results)
        detection_rate = (batch_detected / len(batch_results)) * 100
        print(f"Batch Test ({len(batch_results)} malware rows): {batch_detected}/{len(batch_results)} detected ({detection_rate:.2f}% recall)")
        self.assertGreaterEqual(detection_rate, 99.0, f"Detection rate {detection_rate}% below 99%")


if __name__ == "__main__":
    unittest.main()
