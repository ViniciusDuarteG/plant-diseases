from pathlib import Path
import sys

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tomato-disease-v1.pt"
)


def predict(image_path: str):

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        verbose=False,
    )

    result = results[0]

    probabilities = result.probs

    class_id = probabilities.top1

    confidence = probabilities.top1conf.item()

    class_name = result.names[class_id]

    print()
    print("===== DIAGNÓSTICO =====")
    print(f"Classe: {class_name}")
    print(f"Confiança: {confidence:.2%}")
    print("=======================")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Uso:"
            "\npython tests/predict_image.py "
            "imagem.jpg"
        )

        sys.exit(1)

    predict(sys.argv[1])