import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"

OUT_DATA = PROCESSED_DIR / "analysis_cohort_pca_cluster.csv"
OUT_LOADINGS = RESULT_DIR / "25_pca_loadings.csv"
OUT_CLUSTER_SUMMARY = RESULT_DIR / "25_cluster_summary.csv"
OUT_CLUSTER_TARGETS = RESULT_DIR / "25_cluster_target_distribution.csv"

FIG_DIR = RESULT_DIR / "figures_pca_cluster"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FFQ_ADJ = [
    "FQ_PROT_adj", "FQ_FAT_adj", "FQ_SFA_adj", "FQ_MUFA_adj",
    "FQ_PUFA_adj", "FQ_N3_adj", "FQ_N6_adj", "FQ_CHOL_adj",
    "FQ_CHO_adj", "FQ_TDF_adj", "FQ_CA_adj", "FQ_PHOS_adj",
    "FQ_FE_adj", "FQ_NA_adj", "FQ_K_adj", "FQ_VA_adj",
    "FQ_CAROT_adj", "FQ_RETIN_adj", "FQ_B1_adj", "FQ_B2_adj",
    "FQ_NIAC_adj", "FQ_VITC_adj",
]

TARGETS = [
    "label_diabetes",
    "label_hypertension",
    "label_dyslipidemia",
    "label_obesity",
    "label_prediabetes_clean",
    "label_prehypertension_clean",
]

FEATURE_LABELS = {
    "FQ_PROT_adj": "Protein",
    "FQ_FAT_adj": "Fat",
    "FQ_SFA_adj": "SFA",
    "FQ_MUFA_adj": "MUFA",
    "FQ_PUFA_adj": "PUFA",
    "FQ_N3_adj": "n-3 FA",
    "FQ_N6_adj": "n-6 FA",
    "FQ_CHOL_adj": "Cholesterol",
    "FQ_CHO_adj": "Carbohydrate",
    "FQ_TDF_adj": "Fiber",
    "FQ_CA_adj": "Calcium",
    "FQ_PHOS_adj": "Phosphorus",
    "FQ_FE_adj": "Iron",
    "FQ_NA_adj": "Sodium",
    "FQ_K_adj": "Potassium",
    "FQ_VA_adj": "Vitamin A",
    "FQ_CAROT_adj": "Carotene",
    "FQ_RETIN_adj": "Retinol",
    "FQ_B1_adj": "Thiamin",
    "FQ_B2_adj": "Riboflavin",
    "FQ_NIAC_adj": "Niacin",
    "FQ_VITC_adj": "Vitamin C",
}

def existing(cols, df):
    return [c for c in cols if c in df.columns]

def plot_explained_variance(pca):
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(evr) + 1), cum, marker="o")
    plt.axhline(0.70, linestyle="--", linewidth=1)
    plt.axhline(0.80, linestyle="--", linewidth=1)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_explained_variance.png", dpi=300)
    plt.close()

def plot_pca_scatter(df):
    plt.figure(figsize=(7, 6))
    plt.scatter(df["PC1"], df["PC2"], s=8, alpha=0.35)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Dietary pattern PCA: PC1 vs PC2")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_scatter_pc1_pc2.png", dpi=300)
    plt.close()

def plot_loadings(loadings, pc):
    tmp = loadings[["feature_label", pc]].copy()
    tmp["abs_loading"] = tmp[pc].abs()
    tmp = tmp.sort_values("abs_loading", ascending=True).tail(12)

    plt.figure(figsize=(7, 5))
    plt.barh(tmp["feature_label"], tmp[pc])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Loading")
    plt.title(f"Top nutrient loadings: {pc}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"pca_loadings_{pc}.png", dpi=300)
    plt.close()

def choose_k_by_silhouette(X_pca, k_min=2, k_max=6):
    rows = []

    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50,
        )
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)

        rows.append({
            "k": k,
            "silhouette": sil,
        })

    out = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    plt.plot(out["k"], out["silhouette"], marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("K-means cluster selection")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_silhouette.png", dpi=300)
    plt.close()

    return out

def summarize_clusters(df, features):
    rows = []

    for cluster in sorted(df["diet_cluster"].dropna().unique()):
        sub = df[df["diet_cluster"] == cluster]

        row = {
            "diet_cluster": cluster,
            "n": len(sub),
            "pct": len(sub) / len(df) * 100,
            "age_mean": pd.to_numeric(sub["age"], errors="coerce").mean() if "age" in sub.columns else np.nan,
            "bmi_mean": pd.to_numeric(sub["HE_BMI"], errors="coerce").mean() if "HE_BMI" in sub.columns else np.nan,
            "ffq_energy_mean": pd.to_numeric(sub["FQ_EN"], errors="coerce").mean() if "FQ_EN" in sub.columns else np.nan,
        }

        for pc in ["PC1", "PC2", "PC3"]:
            if pc in sub.columns:
                row[f"{pc}_mean"] = sub[pc].mean()

        for f in features:
            row[f"{f}_mean"] = pd.to_numeric(sub[f], errors="coerce").mean()

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_targets_by_cluster(df):
    rows = []

    for cluster in sorted(df["diet_cluster"].dropna().unique()):
        sub = df[df["diet_cluster"] == cluster]

        for target in TARGETS:
            if target not in sub.columns:
                continue

            y = pd.to_numeric(sub[target], errors="coerce")
            y = y[y.isin([0, 1])]

            if len(y) == 0:
                continue

            rows.append({
                "diet_cluster": cluster,
                "target": target,
                "n_nonmissing": len(y),
                "n_positive": int(y.sum()),
                "prevalence": y.mean(),
            })

    return pd.DataFrame(rows)

def plot_cluster_scatter(df):
    plt.figure(figsize=(7, 6))

    for cluster in sorted(df["diet_cluster"].unique()):
        sub = df[df["diet_cluster"] == cluster]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            s=8,
            alpha=0.45,
            label=f"Cluster {cluster}",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Dietary pattern clusters")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_scatter_pc1_pc2.png", dpi=300)
    plt.close()

def plot_target_prevalence(cluster_targets):
    main_targets = [
        "label_diabetes",
        "label_hypertension",
        "label_dyslipidemia",
        "label_obesity",
    ]

    df = cluster_targets[cluster_targets["target"].isin(main_targets)].copy()

    pivot = df.pivot(
        index="diet_cluster",
        columns="target",
        values="prevalence",
    )

    plt.figure(figsize=(8, 5))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Prevalence")
    plt.xlabel("Diet cluster")
    plt.title("Metabolic phenotype prevalence by diet cluster")
    plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_target_prevalence.png", dpi=300)
    plt.close()

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    features = existing(FFQ_ADJ, df)

    print(f"[FEATURES] {len(features)} FFQ energy-adjusted nutrients")

    X = df[features].copy()

    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    ###########################################################
    # PCA
    ###########################################################

    pca_full = PCA(random_state=RANDOM_STATE)
    X_pca_full = pca_full.fit_transform(X_scaled)

    plot_explained_variance(pca_full)

    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)

    n_components_70 = np.argmax(cum >= 0.70) + 1
    n_components_80 = np.argmax(cum >= 0.80) + 1

    print(f"[PCA] components for 70% variance: {n_components_70}")
    print(f"[PCA] components for 80% variance: {n_components_80}")

    n_components = min(5, len(features))

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    for i in range(n_components):
        df[f"PC{i+1}"] = X_pca[:, i]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index().rename(columns={"index": "feature"})

    loadings["feature_label"] = loadings["feature"].map(FEATURE_LABELS)

    for i in range(n_components):
        loadings[f"PC{i+1}_abs"] = loadings[f"PC{i+1}"].abs()

    loadings.to_csv(OUT_LOADINGS, index=False, encoding="utf-8-sig")

    print("\n[PCA LOADINGS TOP]")
    for pc in ["PC1", "PC2", "PC3"]:
        if pc in loadings.columns:
            print(f"\n{pc}")
            print(
                loadings[["feature_label", pc]]
                .assign(abs_loading=lambda x: x[pc].abs())
                .sort_values("abs_loading", ascending=False)
                .head(10)
            )
            plot_loadings(loadings, pc)

    plot_pca_scatter(df)

    ###########################################################
    # CLUSTERING
    ###########################################################

    cluster_input = df[[f"PC{i+1}" for i in range(n_components)]].copy()

    sil = choose_k_by_silhouette(cluster_input, k_min=2, k_max=6)
    sil.to_csv(RESULT_DIR / "25_cluster_silhouette.csv", index=False, encoding="utf-8-sig")

    best_k = int(sil.sort_values("silhouette", ascending=False).iloc[0]["k"])

    print("\n[SILHOUETTE]")
    print(sil)
    print(f"[BEST K] {best_k}")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=50,
    )

    df["diet_cluster"] = kmeans.fit_predict(cluster_input)

    plot_cluster_scatter(df)

    ###########################################################
    # SUMMARIES
    ###########################################################

    cluster_summary = summarize_clusters(df, features)
    cluster_summary.to_csv(OUT_CLUSTER_SUMMARY, index=False, encoding="utf-8-sig")

    cluster_targets = summarize_targets_by_cluster(df)
    cluster_targets.to_csv(OUT_CLUSTER_TARGETS, index=False, encoding="utf-8-sig")

    plot_target_prevalence(cluster_targets)

    ###########################################################
    # SAVE FINAL DATA
    ###########################################################

    df.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(OUT_DATA)
    print(OUT_LOADINGS)
    print(OUT_CLUSTER_SUMMARY)
    print(OUT_CLUSTER_TARGETS)
    print(RESULT_DIR / "25_cluster_silhouette.csv")
    print(FIG_DIR)

if __name__ == "__main__":
    main()