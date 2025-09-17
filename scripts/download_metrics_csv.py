import subprocess
from pathlib import Path
from rich import print

CSV_ID="1jLmMtNAKdMfsJaigezZz-iJa-_T624Kv"

def main():
    out_dir=Path("metrics"); out_dir.mkdir(parents=True,exist_ok=True)
    out_csv=out_dir/"final_metrics_with_full_configurations.csv"
    print(f"[bold]Downloading metrics CSV → {out_csv}[/]")
    subprocess.check_call(["gdown",f"https://drive.google.com/uc?id={CSV_ID}","-O",str(out_csv)])
    print("[green]Done.[/]")

if __name__=="__main__": main()
