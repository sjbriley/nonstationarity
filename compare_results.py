# save this snippet as compare_results.py and run:  python compare_results.py
import pandas as pd
import pathlib

RESULT_ROOT = "results"        # parent folder that now holds real/, sim_aug/, sim_msc/, …
MODELS = ["LR", "SVM", "RF", "KNN"]

def main():

    RESULT_ROOT = "results"
    MODELS = ["LR", "SVM", "RF", "KNN"]

    rows = []
    for run_dir in pathlib.Path(RESULT_ROOT).iterdir():
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name                # real, sim_aug, sim_msc, …
        for mdl in MODELS:
            csv = run_dir / f"{mdl}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            rows.append({
                "Dataset": run_name,
                "Model":   mdl,
                "Mean_Acc": df["Accuracy"].mean(),
                "Mean_F1":  df["F1_Score"].mean(),
                "Mean_Rec": df["Recall"].mean()
            })

    summary = pd.DataFrame(rows) \
                .pivot(index="Dataset",
                        columns="Model",
                        values="Mean_F1") \
                .round(3)

    print("\n=== Mean F1 by model ===")
    print(summary.to_markdown())
    summary.to_csv("results/summary_f1_by_model.csv")

if __name__ == '__main__':
    main()
