import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Mock pipeline agar tes tidak memuat model besar.
pipeline_stub = types.ModuleType("contextlens_pipeline")
pipeline_stub.analyze_text = Mock()

with patch.dict(sys.modules, {"contextlens_pipeline": pipeline_stub}):
    api = importlib.import_module("api")

client = TestClient(api.app)


class TestContextLensAPI(unittest.TestCase):

    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_rejects_empty_or_whitespace(self):
        for text in ("", "   "):
            with self.subTest(text=repr(text)):
                response = client.post("/analyze", json={"text": text})
                self.assertEqual(response.status_code, 422)

    def test_accepts_exactly_5000_characters(self):
        text = "a" * 5000

        with patch.object(api, "analyze_text", return_value={"ok": True}) as mock:
            response = client.post("/analyze", json={"text": text})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        mock.assert_called_once_with(text)

    def test_rejects_5001_characters(self):
        response = client.post(
            "/analyze",
            json={"text": "a" * 5001},
        )

        self.assertEqual(response.status_code, 422)
        self.assertTrue(
            any(
                item.get("type") == "string_too_long"
                for item in response.json()["detail"]
            )
        )

    def test_strips_text_before_analysis(self):
        with patch.object(api, "analyze_text", return_value={"ok": True}) as mock:
            response = client.post(
                "/analyze",
                json={"text": "  halo ContextLens  "},
            )

        self.assertEqual(response.status_code, 200)
        mock.assert_called_once_with("halo ContextLens")

    def test_hides_internal_pipeline_error(self):
        with patch.object(
            api,
            "analyze_text",
            side_effect=RuntimeError("internal-secret-detail"),
        ):
            response = client.post(
                "/analyze",
                json={"text": "Teks untuk pengujian."},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Analisis gagal diproses. Periksa log backend.",
        )
        self.assertNotIn("internal-secret-detail", response.text)


if __name__ == "__main__":
    unittest.main()
