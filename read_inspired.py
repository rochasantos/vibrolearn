import json
import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("results")


def extract_kfold_mean(json_file, experiment_name):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    accuracies = []
    f1_scores = []

    for fold_data in data["scores"].values():
        accuracies.append(fold_data["accuracy_score"])
        f1_scores.append(fold_data["f1_macro"])

    return {
        "experiment": experiment_name,
        "file": json_file.name,
        "accuracy": sum(accuracies) / len(accuracies),
        "f1_macro": sum(f1_scores) / len(f1_scores)
    }


def load_experiment(pattern, experiment_name):
    rows = []

    for file_path in RESULTS_DIR.glob(pattern):
        rows.append(extract_kfold_mean(file_path, experiment_name))

    return pd.DataFrame(rows)


def main():
    # ===============================
    # LOAD DATA
    # ===============================
    df_no_aug = load_experiment(
        "sehri_et_al_inspired_setup__*.json",
        "No augmentation"
    )

    df_aug = load_experiment(
        "sehri_et_al_inspired_setup_aug_*.json",
        "With augmentation"
    )

    df = pd.concat([df_no_aug, df_aug], ignore_index=True)

    # ===============================
    # FINAL MEAN ± STD
    # ===============================
    summary = (
        df.groupby("experiment")
        .agg({
            "accuracy": ["mean", "std"],
            "f1_macro": ["mean", "std"]
        })
    )

    summary.columns = [
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std"
    ]

    summary = summary.reset_index()

    # ===============================
    # SAVE
    # ===============================
    df.to_csv("sehri_runs_mean_per_file.csv", index=False, float_format="%.4f")
    summary.to_csv("sehri_final_summary.csv", index=False, float_format="%.4f")

    print("\n=== FINAL SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()