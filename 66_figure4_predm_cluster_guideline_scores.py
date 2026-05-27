import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

SUMMARY_INPUT = os.path.join(
    BASE_DIR, "results", "56_cluster_guideline_score_compare",
    "56_cluster_guideline_score_summary.csv"
)

POSTHOC_INPUT = os.path.join(
    BASE_DIR, "results", "56_cluster_guideline_score_compare",
    "56_cluster_guideline_score_posthoc.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "66_figure4_predm_cluster_guideline_scores"
)

os.makedirs(OUT_DIR, exist_ok=True)

summary = pd.read_csv(SUMMARY_INPUT, encoding="utf-8-sig")
posthoc = pd.read_csv(POSTHOC_INPUT, encoding="utf-8-sig")

cluster_order = [1, 2, 3]

cluster_labels = {
    1: "Cluster 1\nBeverage/refined",
    2: "Cluster 2\nBalanced/traditional",
    3: "Cluster 3\nLow-intake/intermediate",
}

score_order = [
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
]

score_labels = {
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_DASH_proxy": "DASH",
    "score_aMED_proxy": "aMED",
    "score_RFS_proxy": "RFS",
}

def get_sig_pairs(posthoc_sub, alpha=0.05):
    sig_pairs = set()
    for _, row in posthoc_sub.iterrows():
        g1 = str(row["group1"])
        g2 = str(row["group2"])
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
# 1. Individual score plots with posthoc letters
# --------------------------------------------------

letter_rows = []

for score in score_order:
    tmp = summary[summary["score"] == score].copy()
    tmp = tmp.set_index("cluster").reindex(cluster_order).reset_index()
    tmp["sem"] = tmp["sd"] / np.sqrt(tmp["n"])

    posthoc_sub = posthoc[posthoc["score"] == score].copy()
    sig_pairs = get_sig_pairs(posthoc_sub, alpha=0.05)

    means_dict = dict(zip(tmp["cluster"].astype(str), tmp["mean"]))

    letters = make_compact_letters(
        groups=[str(c) for c in cluster_order],
        means=means_dict,
        sig_pairs=sig_pairs
    )

    for c in cluster_order:
        letter_rows.append({
            "score": score,
            "cluster": c,
            "mean": means_dict[str(c)],
            "letter": letters[str(c)]
        })

    plt.figure(figsize=(5.4, 4.4))

    bars = plt.bar(
        [cluster_labels[c] for c in cluster_order],
        tmp["mean"],
        yerr=tmp["sem"],
        capsize=4
    )

    y_min = tmp["mean"].min()
    y_max = (tmp["mean"] + tmp["sem"]).max()
    y_range = y_max - y_min if y_max > y_min else 1
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
            fontsize=14,
            fontweight="bold"
        )

    plt.ylabel(score_labels.get(score, score))
    plt.title(f"{score_labels.get(score, score)} by preDM dietary subtype")

    lower = max(0, y_min - y_range * 1.2)
    upper = y_max + y_range * 1.8
    plt.ylim(lower, upper)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT_DIR, f"66_{score}_by_predm_cluster_posthoc.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# --------------------------------------------------
# 2. Grouped barplot: AHEI / HEI / DASH / aMED
# --------------------------------------------------

plot_df = summary[summary["score"].isin(score_order)].copy()

wide_mean = plot_df.pivot(
    index="score",
    columns="cluster",
    values="mean"
).reindex(score_order)

wide_sem = plot_df.assign(
    sem=plot_df["sd"] / np.sqrt(plot_df["n"])
).pivot(
    index="score",
    columns="cluster",
    values="sem"
).reindex(score_order)

x = np.arange(len(score_order))
width = 0.24

plt.figure(figsize=(7.2, 4.8))

for i, cl in enumerate(cluster_order):
    plt.bar(
        x + (i - 1) * width,
        wide_mean[cl],
        yerr=wide_sem[cl],
        width=width,
        capsize=3,
        label=cluster_labels[cl].replace("\n", " ")
    )

plt.xticks(x, [score_labels[s] for s in score_order])
plt.ylabel("Dietary quality score")
plt.title("Guideline-based dietary quality by preDM dietary subtype")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "66_figure4_guideline_scores_grouped_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 3. Save letters table
# --------------------------------------------------

letters_df = pd.DataFrame(letter_rows)

letters_df.to_csv(
    os.path.join(OUT_DIR, "66_guideline_score_posthoc_letters.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("[SAVED]")
print(OUT_DIR)

print("\n[POSTHOC LETTERS]")
print(letters_df)