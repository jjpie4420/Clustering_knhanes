# ============================================================
# 109_compare_foodonly_vs_integrated_clusters.py
#
# Compare food-only clusters vs integrated diet-metabolic clusters
# across isolated lipid, preHTN, and preDM groups
#
# Outputs:
# - cross-tab
# - row percentage table
# - ARI / NMI
# - heatmap
# ============================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BASE_DIR = r"D:\precision_nutrition\FFQ"

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "109_compare_foodonly_vs_integrated_clusters"
)

os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = {
    "isolated_lipid": {
        "label": "Isolated lipid",
        "input": os.path.join(
            BASE_DIR,
            "results",
            "103_isolated_lipid_integrated_diet_metabolic_cluster",
            "103_isolated_lipid_integrated_cluster_assigned_dataset.csv"
        ),
        "integrated_label_map": {
            1: "Prudent-lower metabolic risk",
            2: "Refined-metabolic risk",
        }
    },
    "isolated_preHTN": {
        "label": "Isolated preHTN",
        "input": os.path.join(
            BASE_DIR,
            "results",
            "104_isolated_preHTN_integrated_diet_metabolic_cluster",
            "104_isolated_preHTN_integrated_cluster_assigned_dataset.csv"
        ),
        "integrated_label_map": {
            1: "Prudent-balanced",
            2: "Beverage/lifestyle burden",
            3: "High-intake resilient",
            4: "Refined dietary",
        }
    },
    "isolated_preDM": {
        "label": "Isolated preDM",
        "input": os.path.join(
            BASE_DIR,
            "results",
            "105_isolated_preDM_integrated_diet_metabolic_cluster",
            "105_isolated_preDM_integrated_cluster_assigned_dataset.csv"
        ),
        "integrated_label_map": {
            1: "Lifestyle-metabolic burden",
            2: "Refined dietary",
            3: "Prudent-metabolic resilient",
        }
    },
}

metric_rows = []

for key, cfg in CONFIGS.items():

    print("\n" + "=" * 80)
    print(cfg["label"])

    df = pd.read_csv(cfg["input"], encoding="utf-8-sig", low_memory=False)

    required = [
        "ID",
        "food_only_cluster",
        "food_only_subtype",
        "integrated_cluster"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{key} missing columns: {missing}")

    tmp = df[required].dropna().copy()

    tmp["food_only_cluster"] = tmp["food_only_cluster"].astype(int)
    tmp["integrated_cluster"] = tmp["integrated_cluster"].astype(int)

    tmp["integrated_label"] = tmp["integrated_cluster"].map(
        cfg["integrated_label_map"]
    )

    # ========================================================
    # ARI / NMI
    # ========================================================

    ari = adjusted_rand_score(
        tmp["food_only_cluster"],
        tmp["integrated_cluster"]
    )

    nmi = normalized_mutual_info_score(
        tmp["food_only_cluster"],
        tmp["integrated_cluster"]
    )

    metric_rows.append({
        "condition": key,
        "condition_label": cfg["label"],
        "n": len(tmp),
        "ARI": ari,
        "NMI": nmi,
        "n_food_only_clusters": tmp["food_only_cluster"].nunique(),
        "n_integrated_clusters": tmp["integrated_cluster"].nunique(),
    })

    print(f"N = {len(tmp)}")
    print(f"ARI = {ari:.3f}")
    print(f"NMI = {nmi:.3f}")

    # ========================================================
    # Cross-tab
    # ========================================================

    ctab = pd.crosstab(
        tmp["food_only_subtype"],
        tmp["integrated_label"]
    )

    row_pct = ctab.div(
        ctab.sum(axis=1),
        axis=0
    ) * 100

    print("\n[CROSSTAB]")
    print(ctab)

    print("\n[ROW %]")
    print(row_pct.round(1))

    # Save
    ctab.to_csv(
        os.path.join(
            OUT_DIR,
            f"109_{key}_crosstab_count.csv"
        ),
        encoding="utf-8-sig"
    )

    row_pct.to_csv(
        os.path.join(
            OUT_DIR,
            f"109_{key}_crosstab_row_percent.csv"
        ),
        encoding="utf-8-sig"
    )

    tmp.to_csv(
        os.path.join(
            OUT_DIR,
            f"109_{key}_comparison_dataset.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # Heatmap count
    # ========================================================

    plt.figure(figsize=(8, 4.8))

    sns.heatmap(
        ctab,
        annot=True,
        fmt="d",
        cmap="viridis"
    )

    plt.title(
        f"{cfg['label']}: food-only vs integrated phenotype\n"
        f"ARI={ari:.3f}, NMI={nmi:.3f}"
    )

    plt.xlabel("Integrated diet-metabolic phenotype")
    plt.ylabel("Food-only subtype")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"109_{key}_crosstab_count_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Heatmap row percentage
    # ========================================================

    plt.figure(figsize=(8, 4.8))

    sns.heatmap(
        row_pct,
        annot=True,
        fmt=".1f",
        cmap="viridis"
    )

    plt.title(
        f"{cfg['label']}: row percentage alignment\n"
        f"Food-only subtype → integrated phenotype"
    )

    plt.xlabel("Integrated diet-metabolic phenotype")
    plt.ylabel("Food-only subtype")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"109_{key}_crosstab_row_percent_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# Metric summary
# ============================================================

metrics = pd.DataFrame(metric_rows)

metrics.to_csv(
    os.path.join(
        OUT_DIR,
        "109_foodonly_vs_integrated_alignment_metrics.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("[ALIGNMENT METRICS]")
print(metrics)

# ============================================================
# ARI / NMI barplot
# ============================================================

plot_df = metrics.melt(
    id_vars=["condition_label"],
    value_vars=["ARI", "NMI"],
    var_name="metric",
    value_name="value"
)

plt.figure(figsize=(7, 4.8))

sns.barplot(
    data=plot_df,
    x="condition_label",
    y="value",
    hue="metric"
)

plt.ylim(0, 1)
plt.ylabel("Cluster alignment score")
plt.xlabel("")
plt.title("Alignment between food-only and integrated phenotypes")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "109_alignment_metrics_barplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)