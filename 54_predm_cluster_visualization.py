# 54_predm_cluster_visualization.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

PROFILE_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "51_predm_cluster_metabolic_anova",
    "51_cluster_metabolic_profile_summary.csv"
)

PREV_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "52_predm_cluster_comorbidity_prevalence",
    "52_cluster_comorbidity_prevalence_wide.csv"
)

CLINICAL_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "50_predm_latent_foodpattern_cluster",
    "50_cluster_clinical_profile.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "54_predm_cluster_visualization"
)

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD
# --------------------------------------------------

profile = pd.read_csv(PROFILE_INPUT, encoding="utf-8-sig")
prev = pd.read_csv(PREV_INPUT, encoding="utf-8-sig")
clinical = pd.read_csv(CLINICAL_INPUT, encoding="utf-8-sig")

# --------------------------------------------------
# LABELS
# --------------------------------------------------

cluster_names = {
    1: "C1\nBeverage/refined",
    2: "C2\nBalanced/traditional",
    3: "C3\nLow-intake/conservative",
}

marker_labels = {
    "HE_TG": "Triglycerides",
    "HE_HDL_st2": "HDL-C",
    "HE_wc": "Waist circumference",
    "HE_BMI": "BMI",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
    "HE_glu": "Fasting glucose",
    "HE_HbA1c": "HbA1c",
}

prevalence_labels = {
    "prehypertension_clean": "PreHTN",
    "dyslipidemia": "Dyslipidemia",
    "obesity": "Obesity",
    "hypertension": "HTN",
    "overweight_clean": "Overweight",
    "borderline_lipid_clean": "Borderline lipid",
}

# --------------------------------------------------
# 1. METABOLIC MARKER BARPLOTS
# --------------------------------------------------

selected_markers = [
    "HE_TG",
    "HE_HDL_st2",
    "HE_wc",
    "HE_sbp",
    "HE_dbp",
]

plot_profile = profile[
    profile["variable"].isin(selected_markers)
].copy()

plot_profile["marker_label"] = plot_profile["variable"].map(marker_labels)
plot_profile["cluster_label"] = plot_profile["cluster"].map(cluster_names)

# each marker separate figure
for marker in selected_markers:

    tmp = plot_profile[plot_profile["variable"] == marker].copy()
    tmp = tmp.sort_values("cluster")

    plt.figure(figsize=(5.5, 4))

    plt.bar(
        tmp["cluster_label"],
        tmp["mean"],
        yerr=tmp["sd"] / np.sqrt(tmp["n"]),
        capsize=4
    )

    plt.ylabel(marker_labels.get(marker, marker))
    plt.title(f"{marker_labels.get(marker, marker)} by preDM dietary subtype")
    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        f"54_metabolic_{marker}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# combined normalized marker plot
wide_profile = plot_profile.pivot(
    index="marker_label",
    columns="cluster",
    values="mean"
)

# row-wise z-score for visualization
wide_z = wide_profile.sub(wide_profile.mean(axis=1), axis=0)
wide_z = wide_z.div(wide_profile.std(axis=1), axis=0)

plt.figure(figsize=(7, 4.5))

x = np.arange(len(wide_z.index))
width = 0.24

for i, cl in enumerate([1, 2, 3]):
    plt.bar(
        x + (i - 1) * width,
        wide_z[cl],
        width=width,
        label=cluster_names[cl].replace("\n", " ")
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, wide_z.index, rotation=30, ha="right")
plt.ylabel("Relative marker level (row-wise z-score)")
plt.title("Metabolic profile by preDM dietary subtype")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "54_metabolic_profile_zscore_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 2. COMORBIDITY PREVALENCE BARPLOT
# --------------------------------------------------

selected_outcomes = [
    "prehypertension_clean",
    "dyslipidemia",
    "obesity",
]

prev_plot = prev[
    prev["outcome"].isin(selected_outcomes)
].copy()

prev_plot["outcome_label"] = prev_plot["outcome"].map(prevalence_labels)

cluster_cols = [
    "cluster_1_prevalence_percent",
    "cluster_2_prevalence_percent",
    "cluster_3_prevalence_percent",
]

x = np.arange(len(prev_plot))
width = 0.24

plt.figure(figsize=(7, 4.5))

for i, col in enumerate(cluster_cols):
    cl = i + 1
    plt.bar(
        x + (i - 1) * width,
        prev_plot[col],
        width=width,
        label=cluster_names[cl].replace("\n", " ")
    )

plt.xticks(x, prev_plot["outcome_label"])
plt.ylabel("Prevalence (%)")
plt.title("Cardiometabolic burden by preDM dietary subtype")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "54_comorbidity_prevalence_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 3. AGE / SEX COMPOSITION
# --------------------------------------------------

clinical["cluster_label"] = clinical["cluster"].map(cluster_names)

# age mean
plt.figure(figsize=(5.5, 4))

plt.bar(
    clinical["cluster_label"],
    clinical["age_mean"]
)

plt.ylabel("Age, years")
plt.title("Age by preDM dietary subtype")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "54_age_by_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# female percentage
plt.figure(figsize=(5.5, 4))

plt.bar(
    clinical["cluster_label"],
    clinical["female_percent"]
)

plt.ylabel("Female (%)")
plt.title("Sex composition by preDM dietary subtype")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "54_female_percent_by_cluster.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# DONE
# --------------------------------------------------

print("[SAVED]")
print(OUT_DIR)

print("\nGenerated figures:")
for f in os.listdir(OUT_DIR):
    print(f)