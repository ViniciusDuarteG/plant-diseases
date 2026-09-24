from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"


def main():

    model = YOLO(
        "yolo26n-cls.pt"
    )

    results = model.train(
        data=str(DATASET_DIR),

        epochs=3,

        imgsz=224,

        batch=16,

        patience=10,

        workers=4,

        project=str(
            PROJECT_ROOT / "runs"
        ),

        name="plant-diseases-smoke",
    )

    print(results)


if __name__ == "__main__":
    main()