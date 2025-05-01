# save this snippet as compare_results.py and run:  python compare_results.py
import pandas as pd
import pathlib
import sys

def main():

    RESULT_ROOT = str(sys.argv[1])
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

    # -------- Build separate summary tables for F1‑score and Accuracy --------
    df_results = pd.DataFrame(rows)

    summary_f1 = df_results.pivot(index="Dataset",
                                  columns="Model",
                                  values="Mean_F1").round(3)

    summary_acc = df_results.pivot(index="Dataset",
                                   columns="Model",
                                   values="Mean_Acc").round(3)

    print("\n=== Mean F1 by model ===")
    print(summary_f1.to_markdown())

    print("\n=== Mean Accuracy by model ===")
    print(summary_acc.to_markdown())

    # Combine Accuracy and F1 into one table
    merged = summary_acc.add_suffix('_Acc').join(summary_f1.add_suffix('_F1'))
    print("\n=== Combined Accuracy and F1 by model ===")
    print(merged.to_markdown())
    merged.to_csv("results/combined_accuracy_f1_by_model.csv")

    # Write both tables to CSV for later inspection
    pathlib.Path("results").mkdir(exist_ok=True)
    summary_f1.to_csv("results/summary_f1_by_model.csv")
    summary_acc.to_csv("results/summary_acc_by_model.csv")

if __name__ == '__main__':
    main()
