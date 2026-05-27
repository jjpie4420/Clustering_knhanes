# ============================================================
# 90_make_prehtn_lipid_figures_3_4_5.py
#
# Generate Figure 3, 4, 5 for:
# - preHTN
# - borderline lipid
#
# Figure 3: cluster별 metabolic profile
# Figure 4: cluster별 guideline dietary score
# Figure 5: cluster별 metabolic burden distribution
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

BASE_DIR = r"D:\precision_nutrition\FFQ"

OUT_BASE = os.path.join(
    BASE_DIR,
    "results",
    "90_prehtn_lipid_figures_3_4_5"
)
os.makedirs(OUT_BASE, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

CONFIGS = {
    "preHTN": {
        "condition_label": "preHTN",
        "target_col": "label_prehypertension_clean2",
        "cluster_file": os.path.join(BASE_DIR, "results", "70_preHTN_latent_foodpattern_cluster", "70_preHTN_cluster_assigned_dataset.csv"),
        "met_summary": os.path.join(BASE_DIR, "results", "71_preHTN_cluster_metabolic_anova", "71_preHTN_cluster_metabolic_profile_summary.csv"),
        "met_posthoc": os.path.join(BASE_DIR, "results", "72_preHTN_cluster_posthoc", "72_preHTN_cluster_posthoc_tukey.csv"),
        "score_summary": os.path.join(BASE_DIR, "results", "73_preHTN_cluster_guideline_score_compare", "73_preHTN_cluster_guideline_score_summary.csv"),
        "score_posthoc": os.path.join(BASE_DIR, "results", "73_preHTN_cluster_guideline_score_compare", "73_preHTN_cluster_guideline_score_posthoc.csv"),
        "cluster_labels": {
            1: "C1\nBalanced/prudent",
            2: "C2\nLow-intake",
            3: "C3\nRefined/beverage"
        }
    },
    "lipid": {
        "condition_label": "Borderline lipid",
        "target_col": "label_borderline_lipid_clean",
        "cluster_file": os.path.join(BASE_DIR, "results", "80_lipid_latent_foodpattern_cluster", "80_lipid_cluster_assigned_dataset.csv"),
        "met_summary": os.path.join(BASE_DIR, "results", "81_lipid_cluster_metabolic_anova", "81_lipid_cluster_metabolic_profile_summary.csv"),
        "met_posthoc": os.path.join(BASE_DIR, "results", "82_lipid_cluster_posthoc", "82_lipid_cluster_posthoc_tukey.csv"),
        "score_summary": os.path.join(BASE_DIR, "results", "83_lipid_cluster_guideline_score_compare", "83_lipid_cluster_guideline_score_summary.csv"),
        "score_posthoc": os.path.join(BASE_DIR, "results", "83_lipid_cluster_guideline_score_compare", "83_lipid_cluster_guideline_score_posthoc.csv"),
        "cluster_labels": {
            1: "C1\nRefined/beverage",
            2: "C2\nLow-intake",
            3: "C3\nPrudent/traditional"
        }
    }
}

BURDEN_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

CLUSTER_ORDER = [1, 2, 3]
BURDEN_ORDER = ["1", "2", "3+"]

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
    "HE_wc": "Waist circumference",
    "HE_glu": "Fasting glucose",
    "HE_HbA1c": "HbA1c",
    "HE_TG": "Triglycerides",
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

# ============================================================
# COMPACT LETTER DISPLAY
# ============================================================

def get_sig_pairs(posthoc_sub, alpha=0.05):
    sig_pairs = set()

    for _, row in posthoc_sub.iterrows():
        g1 = str(row["group1"])
        g2 = str(row["group2"])

        if "fdr_bh" in row.index:
            p = float(row["fdr_bh"])
        elif "p-adj" in row.index:
            p = float(row["p-adj"])
        else:
            p = 1.0

        if p < alpha:
            sig_pairs.add(tuple(sorted([g1, g2])))

    return sig_pairs


def pair_is_significant(g1, g2, sig_pairs):
    return tuple(sorted([str(g1), str(g2)])) in sig_pairs


def make_compact_letters(groups, means, sig_pairs):
    groups = [str(g) for g in groups]
    means = {str(k): v for k, v in means.items()}

    sorted_groups = sorted(
        groups,
        key=lambda g: means[g],
        reverse=True
    )

    letters = {g: "" for g in groups}
    letter_list = []
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    for g in sorted_groups:
        assigned = False

        for letter in letter_list:
            members = [x for x in groups if letter in letters[x]]

            can_share = all(
                not pair_is_significant(g, m, sig_pairs)
                for m in members
            )

            if can_share:
                letters[g] += letter
                assigned = True
                break

        if not assigned:
            new_letter = alphabet[len(letter_list)]
            letter_list.append(new_letter)
            letters[g] += new_letter

    return letters


# ============================================================
# FIGURE 3: METABOLIC PROFILE
# ============================================================

def make_figure3_metabolic(condition_key, cfg, out_dir):

    summary = pd.read_csv(cfg["met_summary"], encoding="utf-8-sig")
    posthoc = pd.read_csv(cfg["met_posthoc"], encoding="utf-8-sig")

    # --------------------------------------------------------
    # 3A. Row-wise z-score grouped barplot
    # --------------------------------------------------------

    plot_df = summary[summary["variable"].isin(METABOLIC_MARKERS)].copy()

    wide = plot_df.pivot(
        index="variable",
        columns="cluster",
        values="mean"
    ).reindex(METABOLIC_MARKERS)

    wide.index = [METABOLIC_LABELS.get(v, v) for v in wide.index]

    wide_z = wide.sub(wide.mean(axis=1), axis=0)
    wide_z = wide_z.div(wide.std(axis=1), axis=0)

    plt.figure(figsize=(9, 5))

    x = np.arange(len(wide_z.index))
    width = 0.24

    for i, cl in enumerate(CLUSTER_ORDER):
        plt.bar(
            x + (i - 1) * width,
            wide_z[cl],
            width=width,
            label=cfg["cluster_labels"][cl].replace("\n", " ")
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, wide_z.index, rotation=30, ha="right")
    plt.ylabel("Relative level across clusters\n(row-wise z-score)")
    plt.title(f"Figure 3. Metabolic profile by {cfg['condition_label']} dietary subtype")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"figure3_{condition_key}_metabolic_profile_zscore.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # 3B. Individual marker plots with posthoc letters
    # --------------------------------------------------------

    letter_rows = []

    for marker in METABOLIC_MARKERS:

        tmp = summary[summary["variable"] == marker].copy()

        if tmp.empty:
            continue

        tmp = tmp.set_index("cluster").reindex(CLUSTER_ORDER).reset_index()
        tmp["sem"] = tmp["sd"] / np.sqrt(tmp["n"])

        posthoc_sub = posthoc[posthoc["variable"] == marker].copy()
        sig_pairs = get_sig_pairs(posthoc_sub, alpha=0.05)

        means_dict = dict(zip(tmp["cluster"].astype(str), tmp["mean"]))
        letters = make_compact_letters(CLUSTER_ORDER, means_dict, sig_pairs)

        for cl in CLUSTER_ORDER:
            letter_rows.append({
                "condition": condition_key,
                "variable": marker,
                "cluster": cl,
                "mean": means_dict[str(cl)],
                "letter": letters[str(cl)]
            })

        plt.figure(figsize=(5.4, 4.2))

        bars = plt.bar(
            [cfg["cluster_labels"][cl] for cl in CLUSTER_ORDER],
            tmp["mean"],
            yerr=tmp["sem"],
            capsize=4
        )

        y_min = tmp["mean"].min()
        y_max = (tmp["mean"] + tmp["sem"]).max()
        y_range = y_max - y_min if y_max > y_min else 1
        offset = y_range * 0.08

        for i, bar in enumerate(bars):
            cl = CLUSTER_ORDER[i]
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + tmp.loc[i, "sem"] + offset,
                letters[str(cl)],
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold"
            )

        plt.ylabel(METABOLIC_LABELS.get(marker, marker))
        plt.title(f"{METABOLIC_LABELS.get(marker, marker)} by {cfg['condition_label']} subtype")

        lower = max(0, y_min - y_range * 1.2)
        upper = y_max + y_range * 1.8
        plt.ylim(lower, upper)

        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, f"figure3_{condition_key}_{marker}_posthoc.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    pd.DataFrame(letter_rows).to_csv(
        os.path.join(out_dir, f"figure3_{condition_key}_metabolic_posthoc_letters.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# FIGURE 4: GUIDELINE DIETARY SCORES
# ============================================================

def make_figure4_scores(condition_key, cfg, out_dir):

    summary = pd.read_csv(cfg["score_summary"], encoding="utf-8-sig")
    posthoc = pd.read_csv(cfg["score_posthoc"], encoding="utf-8-sig")

    # --------------------------------------------------------
    # 4A. Grouped barplot
    # --------------------------------------------------------

    plot_df = summary[summary["score"].isin(SCORE_ORDER)].copy()

    wide_mean = plot_df.pivot(
        index="score",
        columns="cluster",
        values="mean"
    ).reindex(SCORE_ORDER)

    wide_sem = plot_df.assign(
        sem=plot_df["sd"] / np.sqrt(plot_df["n"])
    ).pivot(
        index="score",
        columns="cluster",
        values="sem"
    ).reindex(SCORE_ORDER)

    plt.figure(figsize=(8, 5))

    x = np.arange(len(SCORE_ORDER))
    width = 0.24

    for i, cl in enumerate(CLUSTER_ORDER):
        plt.bar(
            x + (i - 1) * width,
            wide_mean[cl],
            yerr=wide_sem[cl],
            width=width,
            capsize=3,
            label=cfg["cluster_labels"][cl].replace("\n", " ")
        )

    plt.xticks(x, [SCORE_LABELS[s] for s in SCORE_ORDER])
    plt.ylabel("Dietary quality score")
    plt.title(f"Figure 4. Guideline-based dietary quality by {cfg['condition_label']} subtype")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"figure4_{condition_key}_guideline_scores_grouped.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # 4B. Individual score plots with letters
    # --------------------------------------------------------

    letter_rows = []

    for score in SCORE_ORDER:

        tmp = summary[summary["score"] == score].copy()

        if tmp.empty:
            continue

        tmp = tmp.set_index("cluster").reindex(CLUSTER_ORDER).reset_index()
        tmp["sem"] = tmp["sd"] / np.sqrt(tmp["n"])

        posthoc_sub = posthoc[posthoc["score"] == score].copy()
        sig_pairs = get_sig_pairs(posthoc_sub, alpha=0.05)

        means_dict = dict(zip(tmp["cluster"].astype(str), tmp["mean"]))
        letters = make_compact_letters(CLUSTER_ORDER, means_dict, sig_pairs)

        for cl in CLUSTER_ORDER:
            letter_rows.append({
                "condition": condition_key,
                "score": score,
                "cluster": cl,
                "mean": means_dict[str(cl)],
                "letter": letters[str(cl)]
            })

        plt.figure(figsize=(5.4, 4.2))

        bars = plt.bar(
            [cfg["cluster_labels"][cl] for cl in CLUSTER_ORDER],
            tmp["mean"],
            yerr=tmp["sem"],
            capsize=4
        )

        y_min = tmp["mean"].min()
        y_max = (tmp["mean"] + tmp["sem"]).max()
        y_range = y_max - y_min if y_max > y_min else 1
        offset = y_range * 0.08

        for i, bar in enumerate(bars):
            cl = CLUSTER_ORDER[i]
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + tmp.loc[i, "sem"] + offset,
                letters[str(cl)],
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold"
            )

        plt.ylabel(SCORE_LABELS.get(score, score))
        plt.title(f"{SCORE_LABELS.get(score, score)} by {cfg['condition_label']} subtype")

        lower = max(0, y_min - y_range * 1.2)
        upper = y_max + y_range * 1.8
        plt.ylim(lower, upper)

        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, f"figure4_{condition_key}_{score}_posthoc.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    pd.DataFrame(letter_rows).to_csv(
        os.path.join(out_dir, f"figure4_{condition_key}_score_posthoc_letters.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# FIGURE 5: BURDEN DISTRIBUTION
# ============================================================

def make_figure5_burden(condition_key, cfg, out_dir):

    burden_df = pd.read_csv(BURDEN_INPUT, encoding="utf-8-sig", low_memory=False)
    cluster_df = pd.read_csv(cfg["cluster_file"], encoding="utf-8-sig", low_memory=False)

    burden_df[cfg["target_col"]] = pd.to_numeric(
        burden_df[cfg["target_col"]],
        errors="coerce"
    )

    sub = burden_df[burden_df[cfg["target_col"]] == 1].copy()

    sub = sub.merge(
        cluster_df[["ID", "cluster"]],
        on="ID",
        how="inner"
    )

    sub["cluster"] = pd.to_numeric(sub["cluster"], errors="coerce")
    sub = sub.dropna(subset=["cluster", "semihealthy_burden_group"]).copy()
    sub["cluster"] = sub["cluster"].astype(int)
    sub["semihealthy_burden_group"] = sub["semihealthy_burden_group"].astype(str)

    sub = sub[sub["semihealthy_burden_group"].isin(BURDEN_ORDER)].copy()

    count_table = pd.crosstab(
        sub["cluster"],
        sub["semihealthy_burden_group"]
    ).reindex(
        index=CLUSTER_ORDER,
        columns=BURDEN_ORDER,
        fill_value=0
    )

    percent_by_cluster = pd.crosstab(
        sub["cluster"],
        sub["semihealthy_burden_group"],
        normalize="index"
    ).reindex(
        index=CLUSTER_ORDER,
        columns=BURDEN_ORDER,
        fill_value=0
    ) * 100

    chi2, p, dof, expected = chi2_contingency(count_table)

    # --------------------------------------------------------
    # 5A. Cluster별 burden distribution
    # --------------------------------------------------------

    plt.figure(figsize=(7.2, 4.8))

    bottom = None
    x = range(len(CLUSTER_ORDER))

    for burden in BURDEN_ORDER:
        values = percent_by_cluster[burden].values

        plt.bar(
            x,
            values,
            bottom=bottom,
            label=f"Burden {burden}"
        )

        if bottom is None:
            bottom = values.copy()
        else:
            bottom += values

    plt.xticks(
        x,
        [cfg["cluster_labels"][cl] for cl in CLUSTER_ORDER]
    )

    plt.ylabel("Proportion within subtype (%)")
    plt.xlabel(f"{cfg['condition_label']} dietary subtype")
    plt.title(
        f"Figure 5A. Metabolic burden distribution by {cfg['condition_label']} subtype\n"
        f"Chi-square p = {p:.3g}"
    )

    plt.legend(
        title="Burden group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"figure5A_{condition_key}_cluster_burden_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------------
    # 5B. Burden별 cluster prevalence
    # --------------------------------------------------------

    percent_by_burden = pd.crosstab(
        sub["semihealthy_burden_group"],
        sub["cluster"],
        normalize="index"
    ).reindex(
        index=BURDEN_ORDER,
        columns=CLUSTER_ORDER,
        fill_value=0
    ) * 100

    plt.figure(figsize=(7.2, 4.8))

    bottom = None
    x = range(len(BURDEN_ORDER))

    for cl in CLUSTER_ORDER:
        values = percent_by_burden[cl].values

        plt.bar(
            x,
            values,
            bottom=bottom,
            label=cfg["cluster_labels"][cl].replace("\n", " ")
        )

        if bottom is None:
            bottom = values.copy()
        else:
            bottom += values

    plt.xticks(x, BURDEN_ORDER)
    plt.ylabel("Subtype prevalence within burden group (%)")
    plt.xlabel("Metabolic burden group")
    plt.title(f"Figure 5B. Subtype prevalence across burden groups in {cfg['condition_label']}")

    plt.legend(
        title="Dietary subtype",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"figure5B_{condition_key}_burden_subtype_prevalence.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    count_table.to_csv(
        os.path.join(out_dir, f"figure5_{condition_key}_cluster_burden_count.csv"),
        encoding="utf-8-sig"
    )

    percent_by_cluster.to_csv(
        os.path.join(out_dir, f"figure5_{condition_key}_cluster_burden_percent.csv"),
        encoding="utf-8-sig"
    )

    percent_by_burden.to_csv(
        os.path.join(out_dir, f"figure5_{condition_key}_burden_cluster_percent.csv"),
        encoding="utf-8-sig"
    )


# ============================================================
# RUN
# ============================================================

for condition_key, cfg in CONFIGS.items():

    print("\n" + "=" * 80)
    print(f"[RUNNING] {condition_key}")
    print("=" * 80)

    out_dir = os.path.join(OUT_BASE, condition_key)
    os.makedirs(out_dir, exist_ok=True)

    make_figure3_metabolic(condition_key, cfg, out_dir)
    make_figure4_scores(condition_key, cfg, out_dir)
    make_figure5_burden(condition_key, cfg, out_dir)

    print(f"[SAVED] {out_dir}")

print("\n[DONE]")
print(OUT_BASE)