"""
Downloads PlantVillage from Kaggle via kagglehub and moves the extracted
dataset into <project_root>/PlantVillage so that train.py finds it at the
path it expects (BASE_DIR/PlantVillage).

Idempotent: if PlantVillage/ already exists at the project root, does nothing.
"""

import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).parent.resolve()
TARGET = PROJECT_ROOT / "PlantVillage"


def find_dataset_root(downloaded: Path) -> Path:
    """
    Locate the directory whose immediate children are the class folders.
    kagglehub may return either the dataset root directly, or a wrapper
    directory containing a single 'PlantVillage' subdirectory. Handle both.
    """
    # Case 1: kagglehub returned a wrapper with PlantVillage inside.
    direct = downloaded / "PlantVillage"
    if direct.is_dir():
        return direct

    # Case 2: nested one level deeper (some kaggle datasets do this).
    for child in downloaded.iterdir():
        if child.is_dir():
            inner = child / "PlantVillage"
            if inner.is_dir():
                return inner

    # Case 3: kagglehub returned the dataset root directly.
    return downloaded


def main():
    if TARGET.exists():
        print(f"'{TARGET}' already exists — nothing to do.")
        return

    print("Downloading PlantVillage via kagglehub …")
    downloaded = Path(kagglehub.dataset_download("emmarex/plantdisease"))
    print(f"Downloaded to: {downloaded}")

    src = find_dataset_root(downloaded)
    print(f"Dataset root resolved to: {src}")

    print(f"Moving to: {TARGET}")
    shutil.move(str(src), str(TARGET))
    print("Done — train.py can now find the dataset at PlantVillage/.")


if __name__ == "__main__":
    main()
