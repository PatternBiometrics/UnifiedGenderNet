#!/usr/bin/env python3
"""
Download experiment checkpoints from Google Drive.

Usage:
  # Download all checkpoints (default)
  python scripts/download_checkpoints.py

  # Download only the best UMCC checkpoint
  python scripts/download_checkpoints.py --best
"""

import subprocess
import sys
from pathlib import Path
import typer
from rich import print

app = typer.Typer(add_completion=False)

# Google Drive IDs
FOLDER_ID = "1iq91ulrO0WSW778Qd0-O9cD09pe4WaR5"   # folder with all checkpoints
BEST_UMCC_URL = "https://drive.google.com/uc?id=1yK36dx8mdG5dvxZNNc50pHoALl77M9jX"  # direct download link
BEST_FILENAME = "1752668630.pt"

@app.command()
def main(
    out: Path = typer.Option(Path("checkpoints"), help="Output directory"),
    best: bool = typer.Option(False, "--best", help="Download only the best UMCC checkpoint")
):
    out.mkdir(parents=True, exist_ok=True)

    if best:
        print(f"[bold cyan]Downloading best UMCC checkpoint → {out/BEST_FILENAME}[/]")
        cmd = ["gdown", BEST_UMCC_URL, "-O", str(out / BEST_FILENAME)]
    else:
        print(f"[bold cyan]Downloading all checkpoints folder → {out.resolve()}[/]")
        cmd = ["gdown", "--folder", f"https://drive.google.com/drive/folders/{FOLDER_ID}", "-O", str(out)]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[red]Download failed:[/] {e}")
        sys.exit(1)

    print("[green]Done.[/]")


if __name__ == "__main__":
    app()
