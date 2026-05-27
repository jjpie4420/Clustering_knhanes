import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

SUMMARY_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "51_predm_cluster_metabolic_anova",
    "51_cluster_metabolic_profile_summary.csv"
)

POSTHOC_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "53_predm_cluster_posthoc",
    "53_cluster_posthoc_tukey.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "65_figure3_predm_cluster_metabolic_profile"
)

os.makedirs(OUT_DIR, exist_ok=True)

summary = pd.read_csv(SUMMARY_INPUT, encoding="utf-8-sig")
posthoc = pd.read_csv(POSTHOC_INPUT, encoding="utf-8-sig")

# --------------------------------------------------
# Settings
# --------------------------------------------------

cluster_order = [1, 2, 3]

cluster_labels = {
    1: "Cluster 1\nBeverage/refined",
    2: "Cluster 2\nBalanced/traditional",
    3: "Cluster 3\nLow-intake/intermediate",
}

markers = [
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

marker_labels = {
    "HE_BMI": "BMI",
    "HE_wc": "Waist circumference (cm)",
    "HE_glu": "Fasting glucose (mg/dL)",
    "HE_HbA1c": "HbA1c (%)",
    "HE_TG": "Triglycerides (mg/dL)",
    "HE_HDL_st2": "HDL-C (mg/dL)",
    "HE_sbp": "SBP (mmHg)",
    "HE_dbp": "DBP (mmHg)",
}

# --------------------------------------------------
# Compact letter display helper
# --------------------------------------------------

def get_sig_pairs(posthoc_sub, alpha=0.05):
    sig_pairs = set()

    for _, row in posthoc_sub.iterrows():
        g1 = str(row["group1"])
        g2 = str(row["group2"])

        # fdr_bh 있으면 그것 기준, 없으면 p-adj
        if "fdr_bh" in row.index:
            p = float(row["fdr_bh"])
        else:
            p = float(row["p-adj"])

        if p < alpha:
            sig_pairs.add(tuple(sorted([g1, g2])))

    return sig_pairs


def pair_is_significant(g1, g2, sig_pairs):
    return tuple(sorted([str(g1), str(g2)])) in sig_pairs


def make_compact_letters(groups, means, sig_pairs):
    groups = list(groups)
    means = {str(k): v for k, v in means.items()}

    sorted_groups = sorted(
        groups,
        key=lambda g: means[str(g)],
        reverse=True
    )

    letters = {str(g): "" for g in groups}
    letter_list = []
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    for g in sorted_groups:
        assigned = False

        for letter in letter_list:
            members = [
                x for x in groups
                if letter in letters[str(x)]
            ]

            can_share = all(
                not pair_is_significant(g, m, sig_pairs)
                for m in members
            )

            if can_share:
                letters[str(g)] += letter
                assigned = True
                break

        if not assigned:
            new_letter = alphabet[len(letter_list)]
            letter_list.append(new_letter)
            letters[str(g)] += new_letter

    return letters


# --------------------------------------------------
# Individual marker plots
# --------------------------------------------------

letter_rows = []

for marker in markers:
    tmp = summary[summary["variable"] == marker].copy()

    if tmp.empty:
        print(f"[SKIP] {marker}")
        continue

    tmp = tmp.set_index("cluster").reindex(cluster_order).reset_index()
    tmp["sem"] = tmp["sd"] / np.sqrt(tmp["n"])

    posthoc_sub = posthoc[posthoc["variable"] == marker].copy()
    sig_pairs = get_sig_pairs(posthoc_sub, alpha=0.05)

    means_dict = dict(zip(tmp["cluster"].astype(str), tmp["mean"]))

    letters = make_compact_letters(
        groups=[str(c) for c in cluster_order],
        means=means_dict,
        sig_pairs=sig_pairs
    )

    for c in cluster_order:
        letter_rows.append({
            "variable": marker,
            "cluster": c,
            "mean": means_dict[str(c)],
            "letter": letters[str(c)]
        })

    plt.figure(figsize=(5.2, 4.2))

    bars = plt.bar(
        [cluster_labels[c] for c in cluster_order],
        tmp["mean"],
        yerr=tmp["sem"],
        capsize=4
    )

    y_min = tmp["mean"].min()
    y_max = (tmp["mean"] + tmp["sem"]).max()
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max) * 0.1, 1)
    label_offset = y_range * 0.08

    for i, bar in enumerate(bars):
        c = cluster_order[i]
        height = bar.get_height()
        sem = tmp.loc[i, "sem"]

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + sem + label_offset,
            letters[str(c)],
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold"
        )

    plt.ylabel(marker_labels.get(marker, marker))
    plt.title(f"{marker_labels.get(marker, marker)} by preDM dietary subtype")

    lower = max(0, y_min - y_range * 1.2)
    upper = y_max + y_range * 1.8
    plt.ylim(lower, upper)

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        f"65_{marker}_by_predm_cluster.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# Combined standardized metabolic profile plot
# row-wise z-score across clusters
# --------------------------------------------------

selected = [
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

plot_df = summary[summary["variable"].isin(selected)].copy()

wide = plot_df.pivot(
    index="variable",
    columns="cluster",
    values="mean"
).reindex(selected)

wide.index = [marker_labels.get(v, v) for v in wide.index]

wide_z = wide.sub(wide.mean(axis=1), axis=0)
wide_z = wide_z.div(wide.std(axis=1), axis=0)

plt.figure(figsize=(8, 4.8))

x = np.arange(len(wide_z.index))
width = 0.24

for i, cl in enumerate(cluster_order):
    plt.bar(
        x + (i - 1) * width,
        wide_z[cl],
        width=width,
        label=cluster_labels[cl].replace("\n", " ")
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, wide_z.index, rotation=30, ha="right")
plt.ylabel("Relative level across clusters\n(row-wise z-score)")
plt.title("Metabolic profile by preDM dietary subtype")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "65_figure3_metabolic_profile_zscore_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Save letters table
# --------------------------------------------------

letters_df = pd.DataFrame(letter_rows)

letters_df.to_csv(
    os.path.join(OUT_DIR, "65_metabolic_posthoc_letters.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("[SAVED]")
print(OUT_DIR)

print("\n[POSTHOC LETTERS]")
print(letters_df)