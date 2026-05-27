# ============================================================
# 92_refined_beverage_cross_condition_anova_heatmap.py
# fixed version
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"D:\precision_nutrition\FFQ"

FULL_DATA = os.path.join(
    BASE_DIR, "results", "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

SCORE_DATA = os.path.join(
    BASE_DIR, "results", "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "92_refined_beverage_cross_condition_anova"
)
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = {
    "preDM": {
        "label": "preDM",
        "target_col": "label_prediabetes_clean2",
        "cluster_file": os.path.join(BASE_DIR, "results", "50_predm_latent_foodpattern_cluster", "50_predm_cluster_assigned_dataset.csv"),
        "refined_cluster": 1,
    },
    "preHTN": {
        "label": "preHTN",
        "target_col": "label_prehypertension_clean2",
        "cluster_file": os.path.join(BASE_DIR, "results", "70_preHTN_latent_foodpattern_cluster", "70_preHTN_cluster_assigned_dataset.csv"),
        "refined_cluster": 3,
    },
    "lipid": {
        "label": "Borderline lipid",
        "target_col": "label_borderline_lipid_clean",
        "cluster_file": os.path.join(BASE_DIR, "results", "80_lipid_latent_foodpattern_cluster", "80_lipid_cluster_assigned_dataset.csv"),
        "refined_cluster": 1,
    },
}

METRICS = {
    "HE_TG": "TG",
    "HE_HDL_st2": "HDL-C",
    "HE_wc": "Waist",
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_DASH_proxy": "DASH",
    "score_aMED_proxy": "aMED",
}

CONDITION_ORDER = ["preDM", "preHTN", "lipid"]

full_df = pd.read_csv(FULL_DATA, encoding="utf-8-sig", low_memory=False)
score_df = pd.read_csv(SCORE_DATA, encoding="utf-8-sig", low_memory=False)

print("[FULL COLUMNS CHECK]")
print([c for c in METRICS if c in full_df.columns])

print("[SCORE COLUMNS CHECK]")
print([c for c in METRICS if c in score_df.columns])

# score variables만 score_df에서 가져오기
score_cols = [c for c in METRICS if c in score_df.columns]

base = full_df.copy()

if score_cols:
    # 중복 컬럼 방지
    add_score = score_df[["ID"] + score_cols].copy()
    base = base.merge(add_score, on="ID", how="left", suffixes=("", "_score"))

# 혹시 suffix가 붙은 경우 원래 이름으로 복구
for c in score_cols:
    if c not in base.columns and f"{c}_score" in base.columns:
        base[c] = base[f"{c}_score"]

print("\n[BASE METRIC AVAILABILITY]")
for c, label in METRICS.items():
    print(c, "FOUND" if c in base.columns else "MISSING")

long_rows = []

for cond, cfg in CONFIGS.items():

    cluster_df = pd.read_csv(cfg["cluster_file"], encoding="utf-8-sig", low_memory=False)

    tmp = base.copy()
    tmp[cfg["target_col"]] = pd.to_numeric(tmp[cfg["target_col"]], errors="coerce")
    tmp = tmp[tmp[cfg["target_col"]] == 1].copy()

    cluster_use = cluster_df[["ID", "cluster"]].copy()
    cluster_use["cluster"] = pd.to_numeric(cluster_use["cluster"], errors="coerce")

    tmp = tmp.merge(cluster_use, on="ID", how="inner")
    tmp = tmp[tmp["cluster"] == cfg["refined_cluster"]].copy()

    tmp["condition"] = cond
    tmp["condition_label"] = cfg["label"]

    keep_cols = ["ID", "condition", "condition_label", "cluster"]
    keep_cols += [c for c in METRICS if c in tmp.columns]

    tmp = tmp[keep_cols].copy()

    long_rows.append(tmp)

long_df = pd.concat(long_rows, axis=0, ignore_index=True)

print("\n[LONG DATA]")
print(long_df.shape)
print(long_df["condition"].value_counts())

metric_cols_found = [c for c in METRICS if c in long_df.columns]

print("\n[METRICS FOUND IN LONG DATA]")
print(metric_cols_found)

if len(metric_cols_found) == 0:
    raise ValueError("No metric columns found. Check variable names in FULL_DATA and SCORE_DATA.")

dup_ids = long_df["ID"].duplicated().sum()
print(f"\n[WARNING] duplicated IDs across condition rows: {dup_ids}")
print("Condition groups can overlap; ANOVA is exploratory.")

long_df.to_csv(
    os.path.join(OUT_DIR, "92_refined_beverage_long_dataset.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# helper
# ============================================================

def pair_key(a, b):
    return tuple(sorted([str(a), str(b)]))

def make_letters(groups, means, sig_pairs):
    groups = [str(g) for g in groups]
    means = {str(k): v for k, v in means.items()}

    sorted_groups = sorted(groups, key=lambda g: means[g], reverse=True)

    letters = {g: "" for g in groups}
    used_letters = []
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    for g in sorted_groups:
        assigned = False

        for letter in used_letters:
            members = [x for x in groups if letter in letters[x]]
            can_share = all(pair_key(g, m) not in sig_pairs for m in members)

            if can_share:
                letters[g] += letter
                assigned = True
                break

        if not assigned:
            new_letter = alphabet[len(used_letters)]
            used_letters.append(new_letter)
            letters[g] += new_letter

    return letters

# ============================================================
# ANOVA + Tukey
# ============================================================

summary_rows = []
anova_rows = []
posthoc_rows = []
letter_rows = []

for metric in metric_cols_found:

    label = METRICS[metric]

    tmp = long_df[["condition", "condition_label", metric]].dropna().copy()

    if tmp["condition"].nunique() < 2:
        continue

    for cond in CONDITION_ORDER:
        x = tmp.loc[tmp["condition"] == cond, metric]

        summary_rows.append({
            "metric": metric,
            "metric_label": label,
            "condition": cond,
            "condition_label": CONFIGS[cond]["label"],
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "sem": x.std() / np.sqrt(len(x)) if len(x) > 0 else np.nan,
        })

    groups = [
        tmp.loc[tmp["condition"] == cond, metric]
        for cond in CONDITION_ORDER
        if len(tmp.loc[tmp["condition"] == cond, metric]) > 0
    ]

    F, p = f_oneway(*groups)

    anova_rows.append({
        "metric": metric,
        "metric_label": label,
        "anova_F": F,
        "anova_p": p,
        "n_total": len(tmp),
    })

    tukey = pairwise_tukeyhsd(
        endog=tmp[metric],
        groups=tmp["condition"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    for c in ["meandiff", "p-adj", "lower", "upper"]:
        tukey_df[c] = pd.to_numeric(tukey_df[c], errors="coerce")

    tukey_df["metric"] = metric
    tukey_df["metric_label"] = label

    posthoc_rows.append(tukey_df)

summary = pd.DataFrame(summary_rows)
anova = pd.DataFrame(anova_rows)

if len(posthoc_rows) > 0:
    posthoc = pd.concat(posthoc_rows, axis=0, ignore_index=True)
else:
    posthoc = pd.DataFrame()

anova["anova_fdr_bh"] = multipletests(
    anova["anova_p"].fillna(1),
    method="fdr_bh"
)[1]

if not posthoc.empty:
    posthoc["fdr_bh"] = multipletests(
        posthoc["p-adj"].fillna(1),
        method="fdr_bh"
    )[1]
    posthoc["significant_fdr"] = posthoc["fdr_bh"] < 0.05

# ============================================================
# letters
# ============================================================

for metric in metric_cols_found:

    label = METRICS[metric]

    sub_sum = summary[summary["metric"] == metric].copy()
    sub_post = posthoc[posthoc["metric"] == metric].copy() if not posthoc.empty else pd.DataFrame()

    means = dict(zip(sub_sum["condition"], sub_sum["mean"]))

    sig_pairs = set()
    if not sub_post.empty:
        for _, row in sub_post.iterrows():
            if row["fdr_bh"] < 0.05:
                sig_pairs.add(pair_key(row["group1"], row["group2"]))

    letters = make_letters(CONDITION_ORDER, means, sig_pairs)

    for cond in CONDITION_ORDER:
        letter_rows.append({
            "metric": metric,
            "metric_label": label,
            "condition": cond,
            "letter": letters[cond],
        })

letters_df = pd.DataFrame(letter_rows)

# ============================================================
# save
# ============================================================

summary.to_csv(
    os.path.join(OUT_DIR, "92_refined_beverage_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

anova.to_csv(
    os.path.join(OUT_DIR, "92_refined_beverage_anova.csv"),
    index=False,
    encoding="utf-8-sig"
)

posthoc.to_csv(
    os.path.join(OUT_DIR, "92_refined_beverage_tukey_posthoc.csv"),
    index=False,
    encoding="utf-8-sig"
)

letters_df.to_csv(
    os.path.join(OUT_DIR, "92_refined_beverage_posthoc_letters.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# heatmap
# ============================================================

wide_mean = summary.pivot(
    index="condition",
    columns="metric_label",
    values="mean"
).reindex(index=CONDITION_ORDER)

wide_letter = letters_df.pivot(
    index="condition",
    columns="metric_label",
    values="letter"
).reindex(index=CONDITION_ORDER)

wide_z = wide_mean.copy()

for col in wide_z.columns:
    sd = wide_z[col].std()
    if sd == 0 or pd.isna(sd):
        wide_z[col] = 0
    else:
        wide_z[col] = (wide_z[col] - wide_z[col].mean()) / sd

plt.figure(figsize=(9.5, 4.7))

plt.imshow(wide_z, aspect="auto", cmap="viridis")
plt.colorbar(label="Across-condition z-score")

plt.xticks(
    range(len(wide_z.columns)),
    wide_z.columns,
    rotation=35,
    ha="right"
)

plt.yticks(
    range(len(wide_z.index)),
    [CONFIGS[c]["label"] for c in wide_z.index]
)

for i, cond in enumerate(wide_z.index):
    for j, metric_label in enumerate(wide_z.columns):

        mean_val = wide_mean.loc[cond, metric_label]
        letter = wide_letter.loc[cond, metric_label]

        text = f"{mean_val:.1f}\n{letter}"

        plt.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if abs(wide_z.iloc[i, j]) > 0.7 else "black"
        )

plt.title(
    "Refined/beverage subtype profile across conditions\n"
    "Numbers = raw means; letters = Tukey post-hoc groups"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "92_refined_beverage_anova_posthoc_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)

print("\n[ANOVA]")
print(anova)

print("\n[POSTHOC LETTERS]")
print(letters_df)