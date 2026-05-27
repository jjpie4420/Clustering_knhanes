# ============================================================
# 113_posthoc_direction_summary.py
#
# 목적:
# same burden 내부에서
# 어떤 food subtype이 실제로 metabolic severity가 높은지
# 방향성 요약
#
# Input:
# - 112 metabolic summary
# - 112 posthoc
#
# Outputs:
# - subtype ranking table
# - significant direction summary
# - heatmap
# ============================================================

import os
from pathlib import Path

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT_DIR = Path(
    BASE_DIR
) / "results" / "112_within_burden_adjusted_models"

OUTDIR = Path(
    BASE_DIR
) / "results" / "113_posthoc_direction_summary"

OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONDITIONS
# ============================================================

condition_list = [
    "preDM_preHTN",
    "preDM_lipid",
    "preHTN_lipid",
    "triple_burden",
]

# ============================================================
# MAIN
# ============================================================

all_direction_rows = []

for condition in condition_list:

    print("\n" + "=" * 80)
    print(condition)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    summary_file = INPUT_DIR / f"112_{condition}_metabolic_summary_by_food_subtype.csv"
    posthoc_file = INPUT_DIR / f"112_{condition}_metabolic_posthoc.csv"

    if not summary_file.exists():
        print("[SKIP] summary missing")
        continue

    summary_df = pd.read_csv(summary_file)

    if posthoc_file.exists():
        posthoc_df = pd.read_csv(posthoc_file)
    else:
        posthoc_df = pd.DataFrame()

    # --------------------------------------------------------
    # Mean pivot
    # --------------------------------------------------------

    mean_pivot = summary_df.pivot(
        index="food_subtype",
        columns="outcome",
        values="mean"
    )

    # --------------------------------------------------------
    # z-score transform
    # --------------------------------------------------------

    z_pivot = mean_pivot.copy()

    for col in z_pivot.columns:

        sd = z_pivot[col].std()

        if pd.isna(sd) or sd == 0:
            z_pivot[col] = 0
        else:
            z_pivot[col] = (
                z_pivot[col] - z_pivot[col].mean()
            ) / sd

    # --------------------------------------------------------
    # Save raw/zscore
    # --------------------------------------------------------

    mean_pivot.to_csv(
        OUTDIR / f"113_{condition}_raw_mean_profile.csv",
        encoding="utf-8-sig"
    )

    z_pivot.to_csv(
        OUTDIR / f"113_{condition}_zscore_profile.csv",
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    preferred_order = [
        "Alcohol/beverage",
        "Refined/processed",
        "Prudent/high-quality",
        "Mixed/intermediate",
    ]

    existing_order = [
        x for x in preferred_order
        if x in z_pivot.index
    ]

    remain = [
        x for x in z_pivot.index
        if x not in existing_order
    ]

    z_pivot = z_pivot.loc[existing_order + remain]

    plt.figure(figsize=(9, 5))

    sns.heatmap(
        z_pivot,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )

    plt.title(
        f"Metabolic severity direction by food subtype\n({condition})",
        fontsize=13,
        weight="bold"
    )

    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()

    plt.savefig(
        OUTDIR / f"113_{condition}_severity_direction_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Rank summary
    # --------------------------------------------------------

    rank_rows = []

    for outcome in mean_pivot.columns:

        temp = (
            mean_pivot[outcome]
            .sort_values(ascending=False)
            .reset_index()
        )

        temp.columns = ["food_subtype", "mean"]

        temp["rank"] = np.arange(1, len(temp) + 1)

        temp["condition"] = condition
        temp["outcome"] = outcome

        rank_rows.append(temp)

    rank_df = pd.concat(rank_rows)

    rank_df.to_csv(
        OUTDIR / f"113_{condition}_subtype_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Significant direction summary
    # --------------------------------------------------------

    if not posthoc_df.empty:

        if "significant_fdr" in posthoc_df.columns:

            sig = posthoc_df[
                posthoc_df["significant_fdr"] == True
            ].copy()

        elif "reject" in posthoc_df.columns:

            sig = posthoc_df[
                posthoc_df["reject"] == True
            ].copy()

        else:
            sig = pd.DataFrame()

        if not sig.empty:

            direction_rows = []

            for _, row in sig.iterrows():

                g1 = row["group1"]
                g2 = row["group2"]

                outcome = row["outcome"]

                mean1 = mean_pivot.loc[g1, outcome]
                mean2 = mean_pivot.loc[g2, outcome]

                if mean1 > mean2:
                    higher = g1
                    lower = g2
                else:
                    higher = g2
                    lower = g1

                direction_rows.append({
                    "condition": condition,
                    "outcome": outcome,
                    "higher_subtype": higher,
                    "lower_subtype": lower,
                    "higher_mean": max(mean1, mean2),
                    "lower_mean": min(mean1, mean2),
                    "mean_difference": abs(mean1 - mean2),
                })

            direction_df = pd.DataFrame(direction_rows)

        else:
            direction_df = pd.DataFrame()

    else:
        direction_df = pd.DataFrame()

    direction_df.to_csv(
        OUTDIR / f"113_{condition}_significant_direction_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_direction_rows.append(direction_df)

    print("[DONE]")
    print(condition)

# ============================================================
# MERGED
# ============================================================

if len(all_direction_rows) > 0:

    merged = pd.concat(
        all_direction_rows,
        ignore_index=True
    )

    merged.to_csv(
        OUTDIR / "113_ALL_significant_direction_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n[SAVED]")
print(OUTDIR)