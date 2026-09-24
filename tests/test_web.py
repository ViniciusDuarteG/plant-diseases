"""Testes de upload e integração; não medem a acurácia do modelo treinado."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


def image_bytes(image_format="PNG", size=(3, 2), exif=None):
    buffer = BytesIO()
    with Image.new("RGB", size, color="green") as image:
        kwargs = {"exif": exif} if exif is not None else {}
        image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.model_path = Path(self.temporary.name) / "best.pt"
        self.app = create_app(self.model_path)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def upload(self, data, filename="folha.png", content_type="image/png"):
        return self.client.post("/api/predict", files={"file": (filename, data, content_type)})

    def test_page_and_assets_work_without_checkpoint(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        for asset in ["app.js", "styles.css"]:
            self.assertEqual(self.client.get(f"/static/{asset}").status_code, 200)
        self.assertFalse(self.client.get("/api/status").json()["model_present"])

    def test_missing_model_is_actionable(self):
        response = self.upload(image_bytes())
        self.assertEqual(response.status_code, 503)
        self.assertIn("models/best.pt", response.json()["detail"])

    def test_file_content_is_validated(self):
        for data in [b"", b"not an image"]:
            with self.subTest(data=data):
                self.assertEqual(self.upload(data).status_code, 400)
        self.assertEqual(self.upload(image_bytes("GIF")).status_code, 415)

    def test_upload_size_limit(self):
        with patch("app.main.MAX_UPLOAD_BYTES", 10):
            self.assertEqual(self.upload(b"x" * 11).status_code, 413)

    def test_image_resolution_limit(self):
        with patch("app.main.MAX_IMAGE_PIXELS", 4):
            self.assertEqual(self.upload(image_bytes()).status_code, 413)

    def test_prediction_uses_cpu_sorts_scores_and_reuses_model(self):
        self.model_path.write_bytes(b"test checkpoint placeholder")
        names = {0: "tomato__healthy", 1: "tomato__early_blight", 2: "potato__late_blight"}
        scores = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0.1, 0.8, 0.1]))
        result = SimpleNamespace(names=names, probs=SimpleNamespace(data=scores))
        predictor = Mock(task="classify", names=names)
        predictor.predict.return_value = [result]
        loader = Mock(return_value=predictor)
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=loader)}):
            response = self.upload(image_bytes())
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["prediction"]["plant"], "Tomateiro")
            self.assertEqual(data["prediction"]["condition"], "Pinta-preta")
            self.assertEqual(data["prediction"]["confidence"], 0.8)
            self.assertEqual(len(data["alternatives"]), 2)
            self.assertEqual(self.upload(image_bytes()).status_code, 200)
        loader.assert_called_once_with(str(self.model_path))
        self.assertEqual(predictor.predict.call_args.kwargs["device"], "cpu")
        self.assertTrue(self.client.get("/api/status").json()["model_loaded"])

    def test_exif_orientation_is_applied_before_prediction(self):
        observed = []

        def predict(image):
            observed.append((image.size, image.mode))
            return [{"plant": "Tomateiro", "condition": "Saudável", "confidence": 0.9}]

        self.app.state.classifier.predict = predict
        exif = Image.Exif()
        exif[274] = 6
        response = self.upload(image_bytes("JPEG", exif=exif), "folha.jpg", "image/jpeg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(observed, [((2, 3), "RGB")])

    def test_wrong_checkpoint_is_rejected(self):
        self.model_path.write_bytes(b"test checkpoint placeholder")
        loader = Mock(return_value=SimpleNamespace(task="detect", names={0: "person"}))
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=loader)}):
            with self.assertLogs("app.main", level="ERROR"):
                self.assertEqual(self.upload(image_bytes()).status_code, 503)


if __name__ == "__main__":
    unittest.main()
