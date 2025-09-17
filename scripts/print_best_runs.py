import pandas as pd, typer
from pathlib import Path
from rich import print
app=typer.Typer(add_completion=False)
CSV=Path("metrics/final_metrics_with_full_configurations.csv")

@app.command()
def main(k:int=10,arch:str=None,modality:str=None):
    if not CSV.exists(): raise SystemExit("metrics CSV missing")
    df=pd.read_csv(CSV)
    if arch and "arch" in df: df=df[df["arch"].str.contains(arch,case=False,na=False)]
    if modality and "test_modality" in df: df=df[df["test_modality"].str.contains(modality,case=False,na=False)]
    metric="bal_acc" if "bal_acc" in df else "accuracy"
    df=df.sort_values(metric,ascending=False).head(k)
    print(df)

if __name__=="__main__": app()
