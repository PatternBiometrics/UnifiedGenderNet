
"""
Download experiment checkpoints from Google Drive.

Usage:
  # Download all checkpoints
  python scripts/download_checkpoints.py

  # Download only the best UMCC checkpoint
  python scripts/download_checkpoints.py --best-umcc

  # Download only the best MAG checkpoint
  python scripts/download_checkpoints.py --best-mag
"""

import subprocess
import sys
from pathlib import Path
import typer
from rich import print

app = typer.Typer(add_completion=False)

# Google Drive IDs
FOLDER_ID = "1iq91ulrO0WSW778Qd0-O9cD09pe4WaR5"   # all checkpoints folder
BEST_UMCC_URL = "https://drive.google.com/uc?id=1yK36dx8mdG5dvxZNNc50pHoALl77M9jX"
BEST_UMCC_FILE = "1752668630.pt"

BEST_MAG_URL = "https://drive.google.com/uc?id=1dtaaEXrHwNI_vhO327DyUL74_5gFCeSX"
BEST_MAG_FILE = "tf_efficientnetv2_s.in1k_F0_Af_Lb_Z0_gating=soft_seed42.pt"

@app.command()
def main(
    out: Path = typer.Option(Path("checkpoints"), help="Output directory"),
    best_umcc: bool = typer.Option(False, "--best-umcc", help="Download only the best UMCC checkpoint"),
    best_mag: bool = typer.Option(False, "--best-mag", help="Download only the best MAG checkpoint")
):
    out.mkdir(parents=True, exist_ok=True)

    if best_umcc:
        print(f"[cyan]Downloading best UMCC checkpoint → {out/BEST_UMCC_FILE}[/]")
        cmd = ["gdown", BEST_UMCC_URL, "-O", str(out / BEST_UMCC_FILE)]
    elif best_mag:
        print(f"[cyan]Downloading best MAG checkpoint → {out/BEST_MAG_FILE}[/]")
        cmd = ["gdown", BEST_MAG_URL, "-O", str(out / BEST_MAG_FILE)]
    else:
        print(f"[cyan]Downloading all checkpoints folder → {out.resolve()}[/]")
        cmd = ["gdown", "--folder", f"https://drive.google.com/drive/folders/{FOLDER_ID}", "-O", str(out)]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[red]Download failed:[/] {e}")
        sys.exit(1)

    print("[green]Done.[/]")

if __name__ == "__main__":
    app()
