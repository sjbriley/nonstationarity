# save this snippet as compare_results.py and run:  python compare_results.py
import pandas as pd
import glob, os, pathlib

RESULT_ROOT = "results"        # parent folder that now holds real/, sim_aug/, sim_msc/, …
MODELS = ["LR", "SVM", "RF", "KNN"]

def main():
    rows = []
    for run_dir in sorted(pathlib.Path(RESULT_ROOT).iterdir()):
        if not run_dir.is_dir():   # skip stray files
            continue
        run_name = run_dir.name    # e.g. real  sim_aug  sim_msc
        for mdl in MODELS:
            csv = run_dir / f"{mdl}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            rows.append({
                "Dataset": run_name,
                "Model": mdl,
                "Mean_Acc":  df["Accuracy"].mean(),
                "Mean_F1":   df["F1_Score"].mean()
            })

    summary = pd.DataFrame(rows).pivot(index="Dataset", columns="Model", values="Mean_F1")
    print(summary.round(3))
    print('='*50)
    print(summary.round(3).to_markdown())
    summary.to_csv("results/summary_f1_by_model.csv")

if __name__ == '__main__':
    main()
