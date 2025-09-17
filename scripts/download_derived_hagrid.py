#!/usr/bin/env python3
"""
Download the derived HaGRID face–hand dataset (images + CSV splits) from Google Drive
and unpack it under:
  data/Shared_Derived_HaGRID_unified_Model_For_Sex_Prediction/

Usage:
  # Download everything (images + CSVs)
  python scripts/download_derived_hagrid.py

  # Only CSVs (if you already have the images)
  python scripts/download_derived_hagrid.py --only-csv

  # Force re-download/unpack
  python scripts/download_derived_hagrid.py --force
"""

import subprocess
import sys
import zipfile
from pathlib import Path
import typer
from rich import print

app = typer.Typer(add_completion=False)

# ----------------------------
# Google Drive file IDs (from your dataset README)
# ----------------------------
ALIGNED_ZIP_ID = "1-52CWGkVhs4k3uWtvAdSz7tvplSX0_84"  # aligned_dataset.zip
TRAIN_CSV_ID   = "1W136bhwoVzT_ipjzABLie377wNNW8H6b"
VAL_CSV_ID     = "1hilEWBh1GHsi469CQ5zckZnmqBOMo73q"
TEST_CSV_ID    = "1dXKcOGVHGKRSERFoVN1HSSM-g7SlWnlf"

# ----------------------------
# Local paths
# ----------------------------
ROOT = Path("data/Shared_Derived_HaGRID_unified_Model_For_Sex_Prediction")
IMAGES_DIR = ROOT / "images" / "content" / "aligned_dataset"
ZIP_PATH = ROOT / "aligned_dataset.zip"
TRAIN_CSV = ROOT / "train.csv"
VAL_CSV   = ROOT / "val.csv"
TEST_CSV  = ROOT / "test.csv"


def run(cmd):
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[red]Command failed:[/] {' '.join(cmd)}")
        sys.exit(e.returncode)


def download_file(file_id: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[cyan]Downloading[/] → {out_path}")
    run(["gdown", url, "-O", str(out_path)])


def unzip(zip_path: Path, out_dir: Path, force: bool = False):
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[yellow]Skip unzip:[/] {out_dir} already populated.")
        return
    print(f"[cyan]Unzipping[/] {zip_path.name} → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir.parent)  # keep internal folder structure if any
    print("[green]Unzip done.[/]")


@app.command()
def main(
    only_csv: bool = typer.Option(False, "--only-csv", help="Download only CSV splits"),
    force: bool = typer.Option(False, "--force", help="Force re-download/unpack"),
):
    # CSVs
    if force or not TRAIN_CSV.exists():
        download_file(TRAIN_CSV_ID, TRAIN_CSV)
    else:
        print(f"[yellow]Skip:[/] {TRAIN_CSV} exists.")
    if force or not VAL_CSV.exists():
        download_file(VAL_CSV_ID, VAL_CSV)
    else:
        print(f"[yellow]Skip:[/] {VAL_CSV} exists.")
    if force or not TEST_CSV.exists():
        download_file(TEST_CSV_ID, TEST_CSV)
    else:
        print(f"[yellow]Skip:[/] {TEST_CSV} exists.")

    if only_csv:
        print("[green]Done (CSV-only mode).[/]")
        return

    # Images ZIP
    if force or not IMAGES_DIR.exists() or not any(IMAGES_DIR.rglob("*.jpg")):
        if force or not ZIP_PATH.exists():
            download_file(ALIGNED_ZIP_ID, ZIP_PATH)
        else:
            print(f"[yellow]Skip:[/] {ZIP_PATH} exists.")
        unzip(ZIP_PATH, IMAGES_DIR, force=force)
    else:
        print(f"[yellow]Skip:[/] images appear to be present in {IMAGES_DIR}")

    print("[green]All done.[/]")


if __name__ == "__main__":
    main()
