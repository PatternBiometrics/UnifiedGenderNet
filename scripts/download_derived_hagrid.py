#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys, zipfile

# Google Drive file IDs
ALIGNED_ZIP_ID = "1-52CWGkVhs4k3uWtvAdSz7tvplSX0_84"  # aligned_dataset.zip
TRAIN_CSV_ID   = "1W136bhwoVzT_ipjzABLie377wNNW8H6b"
VAL_CSV_ID     = "1hilEWBh1GHsi469CQ5zckZnmqBOMo73q"
TEST_CSV_ID    = "1dXKcOGVHGKRSERFoVN1HSSM-g7SlWnlf"

# Local paths
ROOT = Path("data/Shared_Derived_HaGRID_unified_Model_For_Sex_Prediction")
IMAGES_DIR = ROOT / "images" / "content" / "aligned_dataset"
ZIP_PATH = ROOT / "aligned_dataset.zip"
TRAIN_CSV = ROOT / "train.csv"
VAL_CSV   = ROOT / "val.csv"
TEST_CSV  = ROOT / "test.csv"

def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except Exception:
        print("Installing gdown…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--quiet"])

def run(cmd):
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Command failed:", " ".join(cmd))
        raise e

def download_file(file_id: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading →", out_path)
    run([sys.executable, "-m", "gdown", url, "-O", str(out_path)])

def unzip(zip_path: Path, out_dir: Path):
    # If images already present (any jpg), skip
    if out_dir.exists() and any(out_dir.rglob("*.jpg")):
        print("Skip unzip: images present in", out_dir)
        return
    print("Unzipping", zip_path.name, "→", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)
    print("Unzip done.")

def main():
    ensure_gdown()

    # CSVs
    if not TRAIN_CSV.exists():
        download_file(TRAIN_CSV_ID, TRAIN_CSV)
    else:
        print("Skip:", TRAIN_CSV, "exists.")
    if not VAL_CSV.exists():
        download_file(VAL_CSV_ID, VAL_CSV)
    else:
        print("Skip:", VAL_CSV, "exists.")
    if not TEST_CSV.exists():
        download_file(TEST_CSV_ID, TEST_CSV)
    else:
        print("Skip:", TEST_CSV, "exists.")

    # Images
    images_present = IMAGES_DIR.exists() and any(IMAGES_DIR.rglob("*.jpg"))
    if not images_present:
        if not ZIP_PATH.exists():
            download_file(ALIGNED_ZIP_ID, ZIP_PATH)
        else:
            print("Skip:", ZIP_PATH, "exists.")
        unzip(ZIP_PATH, IMAGES_DIR)
    else:
        print("Skip: images already present in", IMAGES_DIR)

    print("All done.")

if __name__ == "__main__":
    main()
