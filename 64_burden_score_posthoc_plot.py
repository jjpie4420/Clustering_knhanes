import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

SUMMARY_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_burden_guideline_score_summary.csv"
)

POSTHOC_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_burden_guideline_score_posthoc.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "64_burden_score_posthoc_plot"
)

os.makedirs(OUT_DIR, exist_ok=True)

summary = pd.read_csv(SUMMARY_INPUT, encoding="utf-8-sig")
posthoc = pd.read_csv(POSTHOC_INPUT, encoding="utf-8-sig")

GROUP_ORDER = ["0", "1", "2", "3+"]

score_labels = {
    "score_aMED_proxy": "aMED",
    "score_DASH_proxy": "DASH",
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_RFS_proxy": "RFS",
}

# --------------------------------------------------
# Helper: compact letter display
# --------------------------------------------------

def get_sig_pairs(posthoc_sub, alpha=0.05):
    """
    Tukey result에서 유의한 pair만 set으로 반환.
    group 이름은 string으로 통일.
    """
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
    """
    간단한 compact letter display 생성.
    원칙:
    - 유의차가 없는 group끼리는 같은 letter 공유 가능.
    - 유의차가 있는 group끼리는 같은 letter 공유 불가.
    """

    groups = list(groups)
    means = {str(k): v for k, v in means.items()}

    # 평균 높은 순서로 letter 할당
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

        # 기존 letter에 들어갈 수 있는지 확인
        for letter in letter_list:
            members = [
                x for x in groups
                if letter in letters[str(x)]
            ]

            # 해당 letter를 가진 모든 group과 유의차가 없어야 같은 letter 가능
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
# Plot each score
# --------------------------------------------------

for score in summary["score"].unique():

    tmp = summary[summary["score"] == score].copy()
    tmp["burden_group"] = tmp["burden_group"].astype(str)

    # order 맞추기
    tmp = tmp.set_index("burden_group").reindex(GROUP_ORDER).reset_index()

    posthoc_sub = posthoc[posthoc["score"] == score].copy()

    sig_pairs = get_sig_pairs(posthoc_sub, alpha=0.05)

    means_dict = dict(zip(tmp["burden_group"], tmp["mean"]))

    letters = make_compact_letters(
        groups=GROUP_ORDER,
        means=means_dict,
        sig_pairs=sig_pairs
    )

    # SEM
    tmp["sem"] = tmp["sd"] / np.sqrt(tmp["n"])

    plt.figure(figsize=(6, 4.5))

    bars = plt.bar(
        tmp["burden_group"],
        tmp["mean"],
        yerr=tmp["sem"],
        capsize=4
    )

    # y-axis margin
    y_min = tmp["mean"].min()
    y_max = (tmp["mean"] + tmp["sem"]).max()
    y_range = y_max - y_min if y_max > y_min else max(y_max * 0.1, 1)

    label_offset = y_range * 0.08

    for i, bar in enumerate(bars):
        group = tmp.loc[i, "burden_group"]
        height = bar.get_height()
        sem = tmp.loc[i, "sem"]

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + sem + label_offset,
            letters.get(group, ""),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold"
        )

    plt.xlabel("Burden group")
    plt.ylabel(score_labels.get(score, score))
    plt.title(f"{score_labels.get(score, score)} by metabolic burden")

    # 그래프가 너무 평평해 보이는 문제 보정
    # 다만 절대값도 보여야 하므로 y축은 0부터 시작하지 않고 mean 중심으로 확대
    lower = max(0, y_min - y_range * 1.2)
    upper = y_max + y_range * 1.8
    plt.ylim(lower, upper)

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        f"64_{score}_by_burden_posthoc_letters.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------
# Also save letters table
# --------------------------------------------------

letter_rows = []

for score in summary["score"].unique():

    tmp = summary[summary["score"] == score].copy()
    tmp["burden_group"] = tmp["burden_group"].astype(str)
    tmp = tmp.set_index("burden_group").reindex(GROUP_ORDER).reset_index()

    posthoc_sub = posthoc[posthoc["score"] == score].copy()
    sig_pairs = get_sig_pairs(posthoc_sub, alpha=0.05)
    means_dict = dict(zip(tmp["burden_group"], tmp["mean"]))

    letters = make_compact_letters(
        groups=GROUP_ORDER,
        means=means_dict,
        sig_pairs=sig_pairs
    )

    for group in GROUP_ORDER:
        letter_rows.append({
            "score": score,
            "burden_group": group,
            "mean": means_dict[group],
            "letter": letters[group]
        })

letters_df = pd.DataFrame(letter_rows)

letters_df.to_csv(
    os.path.join(OUT_DIR, "64_posthoc_compact_letters.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("[SAVED]")
print(OUT_DIR)

print("\n[POSTHOC LETTERS]")
print(letters_df)