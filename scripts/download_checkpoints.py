import subprocess, sys
from pathlib import Path
import typer
from rich import print

app = typer.Typer(add_completion=False)

@app.command()
def main(folder="1iq91ulrO0WSW778Qd0-O9cD09pe4WaR5", out:Path=Path("checkpoints")):
    out.mkdir(parents=True, exist_ok=True)
    print(f"[bold]Downloading checkpoints → {out.resolve()}[/]")
    cmd = ["gdown","--folder",f"https://drive.google.com/drive/folders/{folder}","-O",str(out)]
    try: subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e: sys.exit(1)
    print("[green]Done.[/]")

if __name__=="__main__": app()
