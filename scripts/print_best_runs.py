import pandas as pd, typer
from pathlib import Path
from rich import print
import ast
from IPython.display import display
app=typer.Typer(add_completion=False)
CSV=Path("metrics/final_metrics_with_full_configurations.csv")

@app.command()
def main(k:int=10,arch:str=None,modality:str=None):
    if not CSV.exists(): raise SystemExit("metrics CSV missing")
    df=pd.read_csv(CSV)
  
    df = pd.read_csv("metrics/final_metrics_with_full_configurations.csv")

    df['configuration'] = df['configuration'].apply(ast.literal_eval)

    df_metrics_configs_ablation = df[df['category'] == 'ablation']
    df_metrics_configs_ablation.loc[:,'val_acc_mean'] = df_metrics_configs_ablation.configuration.apply(lambda x: x['mean_acc'])
    df_flags_idx = df_metrics_configs_ablation.configuration.apply(lambda x: x['flag_on'] == True)

    df_falgs_on = df_metrics_configs_ablation[df_flags_idx]
    df_best_model = df_falgs_on.sort_values(by='val_acc_mean', ascending=False).iloc[0]
    display(df_best_model)

if __name__=="__main__": app()
