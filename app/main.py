"""Aplicação local: python -m uvicorn app.main:app --host 127.0.0.1 --port 8037."""

from io import BytesIO
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Annotated
import warnings

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from app.labels import describe_class


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class PlantClassifier:
    """Carrega o checkpoint uma vez e serializa o acesso ao preditor YOLO."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._lock = Lock()

    def predict(self, image: Image.Image) -> list[dict]:
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise ModelUnavailable(
                        "O modelo ainda não foi instalado. Baixe o best.pt do Colab "
                        "e coloque em models/best.pt."
                    )
                from ultralytics import YOLO

                try:
                    model = YOLO(str(self.model_path))
                    if model.task != "classify" or not all("__" in name for name in model.names.values()):
                        raise ValueError("O checkpoint não contém as classes de plantas esperadas.")
                except Exception as error:
                    logger.exception("Não foi possível carregar o checkpoint de plantas.")
                    raise ModelUnavailable(
                        "Não foi possível carregar o modelo. Use o best.pt do treino "
                        "de plantas e confira as dependências."
                    ) from error
                self._model = model

            result = self._model.predict(source=image, device="cpu", verbose=False)[0]
            if result.probs is None:
                raise ModelUnavailable("O arquivo selecionado não é um modelo de classificação.")
            scores = result.probs.data.cpu().tolist()
            top_ids = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:3]
            return [
                {**describe_class(result.names[index]), "confidence": float(scores[index])}
                for index in top_ids
            ]


def decode_image(contents: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(contents)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise HTTPException(415, "Use uma imagem JPG, PNG ou WebP.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "A imagem deve ter no máximo 20 megapixels.")
                source.load()
                return ImageOps.exif_transpose(source).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, "A resolução da imagem é muito grande.") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(400, "Não foi possível ler a imagem. Envie uma foto válida.") from error


def create_app(model_path: Path | None = None) -> FastAPI:
    application = FastAPI(title="Plant Diseases AI", version="0.1.0")
    configured_path = model_path or Path(os.environ.get("PLANT_MODEL_PATH", "models/best.pt"))
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    application.state.classifier = PlantClassifier(configured_path)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/status")
    def status():
        classifier = application.state.classifier
        return {
            "model_present": classifier.model_path.is_file(),
            "model_loaded": classifier._model is not None,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        }

    @application.post("/api/predict")
    def predict(file: Annotated[UploadFile, File()]):
        try:
            contents = file.file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            file.file.close()
        if not contents:
            raise HTTPException(400, "O arquivo está vazio. Escolha uma foto.")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "A foto deve ter no máximo 10 MB.")
        image = decode_image(contents)
        try:
            predictions = application.state.classifier.predict(image)
        except ModelUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except Exception as error:
            logger.exception("Falha ao classificar a imagem.")
            raise HTTPException(500, "Não foi possível analisar a foto. Confira o servidor e tente novamente.") from error
        finally:
            image.close()
        return {"prediction": predictions[0], "alternatives": predictions[1:]}

    return application


app = create_app()
