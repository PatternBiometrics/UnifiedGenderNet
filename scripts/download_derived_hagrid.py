#!/usr/bin/env python3
# scripts/download_derived_hagrid.py
"""
Download and unpack the Derived-HaGRID hand–face dataset from Google Drive.

Usage:
    python scripts/download_derived_hagrid.py
"""

import sys
import zipfile
from pathlib import Path

import gdown

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FOLDER_ID = "1Rc6ub_qmUMRlae5Fsh11FWE_8X3gWxfL"           # Google-Drive folder
DEST_ROOT = Path("data")                                  # local root
DEST_DATA = DEST_ROOT / "Shared_Derived_HaGRID_unified_Model_For_Sex_Prediction"
DEST_IMG  = DEST_DATA / "images"                          # extracted images

EXPECTED_FILES = {
    "aligned_dataset.zip",
    "train.csv",
    "val.csv",
    "test.csv",
    # "removed_faces.zip",
    # "removed_hands.zip",
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def unzip(path: Path, target_dir: Path) -> None:
    """Extract *path* (ZIP) into *target_dir*."""
    print(f"Unzipping {path.name} → {target_dir}")
    try:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        sys.exit(f"[error] {path} is not a valid ZIP file (corrupt or partial).")
    except Exception as exc:                                # noqa: BLE001
        sys.exit(f"[error] Failed to unzip {path}: {exc}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("⏬  Starting Derived-HaGRID download …")

    DEST_IMG.mkdir(parents=True, exist_ok=True)

    # 1) Download every file in the Drive folder
    try:
        gdown.download_folder(id=FOLDER_ID, output=str(DEST_DATA), quiet=False)
    except Exception as exc:                                # noqa: BLE001
        sys.exit(f"[error] gdown failed: {exc}\n"
                 "Make sure the folder is shared for ‘anyone with the link’.")

    # 2) Post-process each expected file
    for fname in EXPECTED_FILES:
        fpath = DEST_DATA / fname
        if not fpath.exists():
            print(f"[warn] {fname} not found in {DEST_DATA} — skip.")
            continue

        if fpath.suffix == ".zip":
            unzip(fpath, DEST_IMG)
            fpath.unlink()                                 # remove ZIP
        elif fpath.suffix == ".csv":
            print(f"✓  {fname} downloaded.")
        else:
            print(f"[info] Unknown file type {fname} — ignored.")

    print("\n✅  Dataset ready!")
    print(f"   CSVs:    {DEST_DATA}")
    print(f"   Images:  {DEST_IMG}")


if __name__ == "__main__":
    main()
