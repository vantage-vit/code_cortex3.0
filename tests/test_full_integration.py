"""
Full integration test for MalwareGuard backend API:
Tests end-to-end executable upload via POST /analyze-file and asserts that
the JSON response contains:
  - verdict
  - malware_probability
  - risk_level
  - top_risk_factors
Also verifies Authenticode pre-ML filtering and model metadata endpoints.
"""

import os
import sys
import unittest
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "MalwareGuard")

for p in [PROJECT_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from backend.main import app, startup_event


class TestFullAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run startup event to initialize predictor and risk engine
        import asyncio
        asyncio.run(startup_event())
        cls.client = TestClient(app)

        cls.sysroot = os.environ.get("SystemRoot", "C:\\Windows")
        cls.explorer_path = os.path.join(cls.sysroot, "explorer.exe")
        cls.cmd_path = os.path.join(cls.sysroot, "System32", "cmd.exe")

    def test_01_health_and_model_info(self):
        print("\n" + "=" * 75)
        print("INTEGRATION TEST 1: Health & Model Information Endpoints")
        print("=" * 75)

        # 1. Health check
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("Health response:", data)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["feature_count"], 47)

        # 2. Model info
        res = self.client.get("/model-info")
        self.assertEqual(res.status_code, 200)
        info = res.json()
        print("Model info feature count:", info["feature_count"])
        self.assertEqual(info["feature_count"], 47)
        self.assertEqual(len(info["features"]), 47)

        # 3. Threshold info
        res = self.client.get("/threshold-info")
        self.assertEqual(res.status_code, 200)
        thresh_info = res.json()
        print("Operating threshold info:", thresh_info)
        self.assertEqual(thresh_info["operating_threshold"], 0.10)

    def test_02_signed_trusted_binary_authenticode(self):
        print("\n" + "=" * 75)
        print("INTEGRATION TEST 2: Real Signed Executable (explorer.exe)")
        print("=" * 75)

        self.assertTrue(os.path.isfile(self.explorer_path), "explorer.exe not found")

        with open(self.explorer_path, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/analyze-file",
            files={"file": ("explorer.exe", file_bytes, "application/octet-stream")}
        )

        self.assertEqual(response.status_code, 200, f"Analysis failed: {response.text}")
        data = response.json()

        print("Response received:")
        print(f"  FileName            : {data.get('fileName')}")
        print(f"  Verdict             : {data.get('verdict')}")
        print(f"  Malware Probability : {data.get('malware_probability')}")
        print(f"  Risk Level          : {data.get('risk_level')}")
        print(f"  Authenticode Vendor : {data.get('authenticode', {}).get('vendor')}")

        # Required fields assertion
        self.assertIn("verdict", data)
        self.assertIn("malware_probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("top_risk_factors", data)

        # Authenticode check verification: Verified vendor returns BENIGN / TRUSTED_VENDOR immediately
        self.assertEqual(data["verdict"], "BENIGN")
        self.assertEqual(data["risk_level"], "TRUSTED_VENDOR")
        self.assertEqual(data["malware_probability"], 0.0)
        self.assertFalse(data["is_malware"])

    def test_03_unsigned_real_executable_end_to_end(self):
        print("\n" + "=" * 75)
        print("INTEGRATION TEST 3: Real Executable ML Pipeline (cmd.exe)")
        print("=" * 75)

        self.assertTrue(os.path.isfile(self.cmd_path), "cmd.exe not found")

        with open(self.cmd_path, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/analyze-file",
            files={"file": ("cmd.exe", file_bytes, "application/octet-stream")}
        )

        self.assertEqual(response.status_code, 200, f"Analysis failed: {response.text}")
        data = response.json()

        print("Response JSON:")
        print(json.dumps({
            "verdict": data.get("verdict"),
            "malware_probability": data.get("malware_probability"),
            "risk_level": data.get("risk_level"),
            "top_risk_factors": data.get("top_risk_factors"),
            "operating_threshold": data.get("operating_threshold"),
            "features": data.get("features"),
        }, indent=2))

        # Check required fields
        self.assertIn("verdict", data)
        self.assertIn("malware_probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("top_risk_factors", data)

        # Validate types and values
        self.assertIsInstance(data["verdict"], str)
        self.assertIsInstance(data["malware_probability"], (int, float))
        self.assertIsInstance(data["risk_level"], str)
        self.assertIsInstance(data["top_risk_factors"], list)

        # cmd.exe is benign
        self.assertEqual(data["verdict"], "BENIGN")
        self.assertFalse(data["is_malware"])
        self.assertLess(data["malware_probability"], data["operating_threshold"])

        # Validate SHAP top_risk_factors schema
        self.assertGreaterEqual(len(data["top_risk_factors"]), 1)
        for factor in data["top_risk_factors"]:
            self.assertIn("feature", factor)
            self.assertIn("impact", factor)
            self.assertIn("indicates", factor)

        # Validate frontend compatibility fields
        self.assertIn("fileName", data)
        self.assertIn("md5", data)
        self.assertIn("confidence", data)
        self.assertIn("features", data)
        self.assertIn("topFeatures", data)


if __name__ == "__main__":
    unittest.main()
