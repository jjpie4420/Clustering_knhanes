# ============================================
# 29_food_group_case_control_compare.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# --------------------------------------------
# LOAD
# --------------------------------------------

df = pd.read_csv(
    r"D:\precision_nutrition\FFQ\processed\analysis_cohort_energy_adjusted.csv"
)

fg = pd.read_csv(
    r"D:\precision_nutrition\FFQ\processed\ffq_food_groups.csv"
)

# --------------------------------------------
# MERGE
# --------------------------------------------

id_col = "ID"

df = df.merge(fg, on=id_col, how="left")

# --------------------------------------------
# TARGETS
# --------------------------------------------

targets = {
    "diabetes": "label_diabetes",
    "hypertension": "label_hypertension",
    "dyslipidemia": "label_dyslipidemia",
    "obesity": "label_obesity"
}

# --------------------------------------------
# FOOD GROUPS
# --------------------------------------------

food_groups = [
    "fg_refined_grain_z",
    "fg_whole_grain_z",
    "fg_fastfood_z",
    "fg_meat_processed_z",
    "fg_fish_seafood_z",
    "fg_vegetable_z",
    "fg_kimchi_fermented_z",
    "fg_fruit_z",
    "fg_dairy_z",
    "fg_sweet_beverage_z",
    "fg_alcohol_z",
    "fg_coffee_tea_z"
]

# --------------------------------------------
# OUTPUT
# --------------------------------------------

summary_rows = []

save_dir = r"D:\precision_nutrition\FFQ\results\food_group_case_control"

import os
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------
# LOOP
# --------------------------------------------

for target_name, label_col in targets.items():

    print(f"\n[TARGET] {target_name}")

    sub = df[df[label_col].isin([0, 1])].copy()

    case = sub[sub[label_col] == 1]
    control = sub[sub[label_col] == 0]

    rows = []

    for fg_col in food_groups:

        case_mean = case[fg_col].mean()
        control_mean = control[fg_col].mean()

        t, p = ttest_ind(
            case[fg_col].dropna(),
            control[fg_col].dropna(),
            equal_var=False
        )

        rows.append({
            "target": target_name,
            "food_group": fg_col,
            "case_mean_z": case_mean,
            "control_mean_z": control_mean,
            "difference": case_mean - control_mean,
            "p_value": p
        })

    res = pd.DataFrame(rows)

    # ----------------------------------------
    # SAVE CSV
    # ----------------------------------------

    res.to_csv(
        f"{save_dir}/{target_name}_foodgroup_compare.csv",
        index=False
    )

    summary_rows.append(res)

    # ----------------------------------------
    # FIGURE
    # ----------------------------------------

    plot_df = res.sort_values("difference")

    plt.figure(figsize=(8, 6))

    colors = [
        "red" if x > 0 else "blue"
        for x in plot_df["difference"]
    ]

    plt.barh(
        plot_df["food_group"],
        plot_df["difference"],
        color=colors
    )

    plt.axvline(0, linestyle="--")

    plt.xlabel("Case - Control Mean Difference (Z-score)")
    plt.title(f"{target_name}: food-group pattern difference")

    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/{target_name}_foodgroup_difference.png",
        dpi=300
    )

    plt.close()

# --------------------------------------------
# COMBINED SUMMARY
# --------------------------------------------

all_summary = pd.concat(summary_rows)

all_summary.to_csv(
    f"{save_dir}/all_foodgroup_summary.csv",
    index=False
)

print("\n[SAVED]")
print(save_dir)