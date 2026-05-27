# ============================================================
# 91_cross_condition_comparison_summary.py
#
# Compare dietary subtypes across:
# - preDM
# - preHTN
# - borderline lipid
#
# Outputs:
# 1. cross-condition subtype summary table
# 2. metabolic profile heatmap
# 3. dietary score heatmap
# 4. subtype food-profile heatmap
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "91_cross_condition_comparison_summary"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

CONFIGS = {
    "preDM": {
        "condition_label": "preDM",
        "food_profile": os.path.join(BASE_DIR, "results", "50_predm_latent_foodpattern_cluster", "50_cluster_foodgroup_profile.csv"),
        "met_summary": os.path.join(BASE_DIR, "results", "51_predm_cluster_metabolic_anova", "51_cluster_metabolic_profile_summary.csv"),
        "score_summary": os.path.join(BASE_DIR, "results", "56_cluster_guideline_score_compare", "56_cluster_guideline_score_summary.csv"),
        "cluster_map": {
            1: "Refined/beverage",
            2: "Prudent/high-quality",
            3: "Low-intake/intermediate",
        }
    },
    "preHTN": {
        "condition_label": "preHTN",
        "food_profile": os.path.join(BASE_DIR, "results", "70_preHTN_latent_foodpattern_cluster", "70_preHTN_cluster_food_profile.csv"),
        "met_summary": os.path.join(BASE_DIR, "results", "71_preHTN_cluster_metabolic_anova", "71_preHTN_cluster_metabolic_profile_summary.csv"),
        "score_summary": os.path.join(BASE_DIR, "results", "73_preHTN_cluster_guideline_score_compare", "73_preHTN_cluster_guideline_score_summary.csv"),
        "cluster_map": {
            1: "Prudent/high-quality",
            2: "Low-intake/intermediate",
            3: "Refined/beverage",
        }
    },
    "lipid": {
        "condition_label": "Borderline lipid",
        "food_profile": os.path.join(BASE_DIR, "results", "80_lipid_latent_foodpattern_cluster", "80_lipid_cluster_food_profile.csv"),
        "met_summary": os.path.join(BASE_DIR, "results", "81_lipid_cluster_metabolic_anova", "81_lipid_cluster_metabolic_profile_summary.csv"),
        "score_summary": os.path.join(BASE_DIR, "results", "83_lipid_cluster_guideline_score_compare", "83_lipid_cluster_guideline_score_summary.csv"),
        "cluster_map": {
            1: "Refined/beverage",
            2: "Low-intake/intermediate",
            3: "Prudent/high-quality",
        }
    }
}

METABOLIC_MARKERS = [
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

METABOLIC_LABELS = {
    "HE_BMI": "BMI",
    "HE_wc": "Waist",
    "HE_glu": "Glucose",
    "HE_HbA1c": "HbA1c",
    "HE_TG": "TG",
    "HE_HDL_st2": "HDL-C",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
}

SCORE_ORDER = [
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
    "score_RFS_proxy",
]

SCORE_LABELS = {
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_DASH_proxy": "DASH",
    "score_aMED_proxy": "aMED",
    "score_RFS_proxy": "RFS",
}

SUBTYPE_ORDER = [
    "Refined/beverage",
    "Low-intake/intermediate",
    "Prudent/high-quality",
]

# ============================================================
# LOAD AND STANDARDIZE
# ============================================================

food_rows = []
met_rows = []
score_rows = []
summary_rows = []

for cond, cfg in CONFIGS.items():

    # -----------------------------
    # Food profile
    # -----------------------------
    food = pd.read_csv(cfg["food_profile"], encoding="utf-8-sig", index_col=0)

    # index = food labels, columns = cluster
    food.columns = [int(c) for c in food.columns]

    for cl in food.columns:
        subtype = cfg["cluster_map"][cl]

        row = {
            "condition": cond,
            "condition_label": cfg["condition_label"],
            "cluster": cl,
            "subtype": subtype,
        }

        for food_name in food.index:
            row[food_name] = food.loc[food_name, cl]

        food_rows.append(row)

    # -----------------------------
    # Metabolic summary
    # -----------------------------
    met = pd.read_csv(cfg["met_summary"], encoding="utf-8-sig")

    for cl, subtype in cfg["cluster_map"].items():
        sub = met[met["cluster"] == cl].copy()

        row = {
            "condition": cond,
            "condition_label": cfg["condition_label"],
            "cluster": cl,
            "subtype": subtype,
        }

        for marker in METABOLIC_MARKERS:
            v = sub.loc[sub["variable"] == marker, "mean"]
            if len(v) > 0:
                row[METABOLIC_LABELS.get(marker, marker)] = float(v.iloc[0])

        met_rows.append(row)

    # -----------------------------
    # Dietary score summary
    # -----------------------------
    score = pd.read_csv(cfg["score_summary"], encoding="utf-8-sig")

    for cl, subtype in cfg["cluster_map"].items():
        sub = score[score["cluster"] == cl].copy()

        row = {
            "condition": cond,
            "condition_label": cfg["condition_label"],
            "cluster": cl,
            "subtype": subtype,
        }

        for s in SCORE_ORDER:
            v = sub.loc[sub["score"] == s, "mean"]
            if len(v) > 0:
                row[SCORE_LABELS.get(s, s)] = float(v.iloc[0])

        score_rows.append(row)

food_df = pd.DataFrame(food_rows)
met_df = pd.DataFrame(met_rows)
score_df = pd.DataFrame(score_rows)

# ============================================================
# MERGED SUMMARY TABLE
# ============================================================

summary = food_df[["condition", "condition_label", "cluster", "subtype"]].merge(
    met_df,
    on=["condition", "condition_label", "cluster", "subtype"],
    how="left"
).merge(
    score_df,
    on=["condition", "condition_label", "cluster", "subtype"],
    how="left"
)

summary.to_csv(
    os.path.join(OUT_DIR, "91_cross_condition_subtype_summary_table.csv"),
    index=False,
    encoding="utf-8-sig"
)

food_df.to_csv(
    os.path.join(OUT_DIR, "91_cross_condition_food_profile_table.csv"),
    index=False,
    encoding="utf-8-sig"
)

met_df.to_csv(
    os.path.join(OUT_DIR, "91_cross_condition_metabolic_profile_table.csv"),
    index=False,
    encoding="utf-8-sig"
)

score_df.to_csv(
    os.path.join(OUT_DIR, "91_cross_condition_diet_score_table.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("[SUMMARY TABLE]")
print(summary)

# ============================================================
# HELPER: z-score within condition for fair comparison
# ============================================================

def zscore_within_condition(df, value_cols):
    out = df.copy()

    for cond in out["condition"].unique():
        idx = out["condition"] == cond

        for c in value_cols:
            mean = out.loc[idx, c].mean()
            sd = out.loc[idx, c].std()

            if sd == 0 or pd.isna(sd):
                out.loc[idx, c + "_z"] = 0
            else:
                out.loc[idx, c + "_z"] = (out.loc[idx, c] - mean) / sd

    return out

# ============================================================
# FIGURE 1: metabolic profile heatmap
# ============================================================

met_cols = [METABOLIC_LABELS[m] for m in METABOLIC_MARKERS if METABOLIC_LABELS[m] in met_df.columns]

met_z = zscore_within_condition(met_df, met_cols)

heatmap_met = met_z.copy()
heatmap_met["row_label"] = heatmap_met["condition_label"] + " | " + heatmap_met["subtype"]

heatmap_met = heatmap_met.set_index("row_label")[[c + "_z" for c in met_cols]]
heatmap_met.columns = met_cols

plt.figure(figsize=(9, 6))
plt.imshow(heatmap_met, aspect="auto")
plt.colorbar(label="Within-condition z-score")
plt.xticks(range(len(heatmap_met.columns)), heatmap_met.columns, rotation=35, ha="right")
plt.yticks(range(len(heatmap_met.index)), heatmap_met.index)
plt.title("Cross-condition metabolic profile by dietary subtype")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "91_cross_condition_metabolic_profile_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# FIGURE 2: dietary score heatmap
# ============================================================

score_cols = [SCORE_LABELS[s] for s in SCORE_ORDER if SCORE_LABELS[s] in score_df.columns]

score_z = zscore_within_condition(score_df, score_cols)

heatmap_score = score_z.copy()
heatmap_score["row_label"] = heatmap_score["condition_label"] + " | " + heatmap_score["subtype"]

heatmap_score = heatmap_score.set_index("row_label")[[c + "_z" for c in score_cols]]
heatmap_score.columns = score_cols

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_score, aspect="auto")
plt.colorbar(label="Within-condition z-score")
plt.xticks(range(len(heatmap_score.columns)), heatmap_score.columns, rotation=35, ha="right")
plt.yticks(range(len(heatmap_score.index)), heatmap_score.index)
plt.title("Cross-condition dietary quality profile by subtype")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "91_cross_condition_diet_score_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# FIGURE 3: compact cross-condition subtype table plot
# ============================================================

compact = summary[[
    "condition_label",
    "subtype",
    "TG",
    "HDL-C",
    "Waist",
    "AHEI",
    "HEI",
    "DASH",
    "RFS"
]].copy()

compact.to_csv(
    os.path.join(OUT_DIR, "91_compact_cross_condition_key_metrics.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FIGURE 4: refined subtype comparison
# ============================================================

refined = summary[summary["subtype"] == "Refined/beverage"].copy()

key_cols = ["TG", "HDL-C", "Waist", "AHEI", "HEI", "DASH", "RFS"]
refined_z = refined.copy()

for c in key_cols:
    refined_z[c] = (refined_z[c] - summary[c].mean()) / summary[c].std()

refined_plot = refined_z.set_index("condition_label")[key_cols]

plt.figure(figsize=(8, 4.5))
plt.imshow(refined_plot, aspect="auto")
plt.colorbar(label="Overall z-score")
plt.xticks(range(len(key_cols)), key_cols, rotation=35, ha="right")
plt.yticks(range(len(refined_plot.index)), refined_plot.index)
plt.title("Refined/beverage subtype profile across conditions")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "91_refined_beverage_subtype_cross_condition_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n[SAVED]")
print(OUT_DIR)