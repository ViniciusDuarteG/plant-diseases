from pathlib import Path 
import random 
import shutil
import hashlib


PROJACT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJACT_ROOT / "dataset" / "raw"
DATASET_DIR = PROJACT_ROOT / "dataset"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42 

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

def clear_output_directories():
    for split in ["train", "val", "test"]:
        path = DATASET_DIR / split

        if path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)


def normalize_class_name(name: str) -> str:
    return name.lower().replace("___", "__").replace(" ", "_").replace("-", "_")


def get_images(directory: Path):
    return [
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


def split_images(images):
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_files(files, destination):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in files:
        shutil.copy2(
            file,
            destination / file.name,
        )


def remove_duplicate_images(classes):
    seen = {}
    unique_images = {}
    duplicates = 0

    for class_directory in sorted(classes):
        unique_images[class_directory] = []

        for image_path in sorted(get_images(class_directory)):
            image_hash = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()

            if image_hash in seen:
                previous_class, previous_image = seen[image_hash]

                if previous_class != class_directory.name:
                    raise ValueError(
                        "Imagem idêntica encontrada em classes diferentes:\n"
                        f"  {previous_image}\n"
                        f"  {image_path}\n"
                        "Revise os rótulos antes de continuar."
                    )

                duplicates += 1
                print(f"[DUPLICADA] Ignorada: {image_path}")
                continue

            seen[image_hash] = (
                class_directory.name,
                image_path,
            )

            unique_images[class_directory].append(image_path)

    print(f"\n{duplicates} duplicatas excluídas da seleção.\n")

    return unique_images


def main():
    random.seed(SEED)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {RAW_DIR}"
        )

    classes = [
        directory
        for directory in RAW_DIR.iterdir()
        if directory.is_dir()
    ]

    print(f"{len(classes)} classes encontradas\n")

    unique_images = remove_duplicate_images(classes)

    clear_output_directories()

    for class_directory in sorted(classes):

        class_name = normalize_class_name(
            class_directory.name
        )

        images = unique_images[class_directory]

        if not images:
            print(
                f"[AVISO] Nenhuma imagem em "
                f"{class_directory.name}"
            )
            continue

        train, val, test = split_images(images)

        copy_files(
            train,
            DATASET_DIR / "train" / class_name,
        )

        copy_files(
            val,
            DATASET_DIR / "val" / class_name,
        )

        copy_files(
            test,
            DATASET_DIR / "test" / class_name,
        )

        print(class_name)
        print(f"  train: {len(train)}")
        print(f"  val:   {len(val)}")
        print(f"  test:  {len(test)}")
        print()

    print("Dataset preparado.")


if __name__ == "__main__":
    main()