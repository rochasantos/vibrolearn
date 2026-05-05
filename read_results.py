import json
import pandas as pd
from pathlib import Path

from sqlalchemy import table


RESULTS_DIR = Path("results")


def parse_scores(json_file, experiment_name):
    rows = []

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, values in data["scores"].items():
        _, round_id, fold_id = key.split("_")

        rows.append({
            "experiment": experiment_name,
            "round": int(round_id) + 1,  # rounds 1–8
            "accuracy": values["accuracy_score"],
            "f1_macro": values["f1_macro"]
        })

    return rows


def load_all(pattern, experiment_name):
    rows = []

    for file_path in RESULTS_DIR.glob(pattern):
        rows.extend(parse_scores(file_path, experiment_name))

    return pd.DataFrame(rows)


def compute_table(df):
    # média e std por round
    round_stats = (
        df.groupby(["experiment", "round"])
        .agg({
            "accuracy": ["mean", "std"],
            "f1_macro": ["mean", "std"]
        })
    )

    round_stats.columns = [
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std"
    ]

    round_stats = round_stats.reset_index()

    # média e std geral
    overall_stats = (
        df.groupby("experiment")
        .agg({
            "accuracy": ["mean", "std"],
            "f1_macro": ["mean", "std"]
        })
    )

    overall_stats.columns = [
        "accuracy_mean", "accuracy_std",
        "f1_mean", "f1_std"
    ]

    overall_stats = overall_stats.reset_index()
    overall_stats["round"] = "Mean"

    # juntar
    final = pd.concat([round_stats, overall_stats], ignore_index=True)

    return final


def main():
    df_no_aug = load_all("rauber_de_fe_setup__*.json", "No augmentation")
    df_aug = load_all("rauber_de_fe_setup_aug_*.json", "With augmentation")

    df = pd.concat([df_no_aug, df_aug], ignore_index=True)

    table = compute_table(df)

    # ordenar corretamente (1–8 + Mean)
    table["round_order"] = table["round"].apply(lambda x: 999 if x == "Mean" else int(x))
    table = table.sort_values(["experiment", "round_order"]).drop(columns="round_order")

    # salvar
    table.to_csv("table_round_mean_std.csv", index=False, float_format="%.4f")

    print(table)


if __name__ == "__main__":
    main()