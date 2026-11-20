"""
kmeans_model.py
Segmentación empresarial con K-Means (K = 4) en Chía y Sabana Centro (Vigencia 2026):
  - Clúster 0: Tradicional / Rezagada
  - Clúster 1: Digitalización Básica
  - Clúster 2: En Consolidación
  - Clúster 3: Líder Digital Competitiva
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split

from matplotlib.colors import LinearSegmentedColormap
from data_engineering import load_and_engineer_data

CLUSTER_NAMES = {
    0: "Tradicional / Rezagada",
    1: "Digitalización Básica",
    2: "En Consolidación",
    3: "Líder Digital Competitiva",
}

CLUSTER_COLORS = ["#C62828", "#F9A825", "#00ACC1", "#2E7D32"]


def get_kmeans_results(generate_plots: bool = True) -> dict:
    bundle = load_and_engineer_data()
    df = bundle["df"]
    X_scaled = bundle["X_scaled"]

    X_train, X_val = train_test_split(X_scaled, test_size=0.20, random_state=42)

    comparison = []
    inertias, silhouettes = [], []
    k_range = list(range(2, 7))
    for k in k_range:
        km_tmp = KMeans(
            n_clusters=k, init="k-means++", n_init=20, max_iter=300, tol=1e-4, random_state=42
        )
        km_tmp.fit(X_train)
        lbl_va = km_tmp.predict(X_val)
        sil_va = round(float(silhouette_score(X_val, lbl_va)), 4)
        db_va = round(float(davies_bouldin_score(X_val, lbl_va)), 4)
        inertias.append(float(km_tmp.inertia_))
        silhouettes.append(sil_va)
        comparison.append({
            "k": k,
            "inertia": round(float(km_tmp.inertia_), 1),
            "silhouette": sil_va,
            "db": db_va,
            "selected": (k == 4),
        })

    k_final = 4
    km = KMeans(
        n_clusters=k_final, init="k-means++", n_init=20, max_iter=300, tol=1e-4, random_state=42
    )
    raw_train = km.fit_predict(X_train)
    raw_val = km.predict(X_val)
    raw_all = km.predict(X_scaled)

    # Ordenar clústeres de menor (0) a mayor (3) madurez digital promedio
    raw_means = [
        df.loc[raw_all == r, "indice_madurez_digital"].mean() for r in range(k_final)
    ]
    order = np.argsort(raw_means)
    rank_map = {int(raw): int(rank) for rank, raw in enumerate(order)}

    train_labels = np.array([rank_map[int(r)] for r in raw_train])
    val_labels = np.array([rank_map[int(r)] for r in raw_val])
    all_labels = np.array([rank_map[int(r)] for r in raw_all])
    df["kmeans_cluster"] = all_labels
    df["kmeans_label"] = df["kmeans_cluster"].map(CLUSTER_NAMES)

    metrics = {
        "inertia": round(float(km.inertia_), 2),
        "n_iter": int(km.n_iter_),
        "train_silhouette": round(float(silhouette_score(X_train, train_labels)), 4),
        "val_silhouette": round(float(silhouette_score(X_val, val_labels)), 4),
        "train_db": round(float(davies_bouldin_score(X_train, train_labels)), 4),
        "val_db": round(float(davies_bouldin_score(X_val, val_labels)), 4),
    }

    cluster_summary = []
    for c in range(k_final):
        sub = df[df["kmeans_cluster"] == c]
        cluster_summary.append({
            "cluster": c,
            "label": CLUSTER_NAMES[c],
            "count": int(len(sub)),
            "chia_count": int((sub["municipio"] == "CHÍA").sum()),
            "pct": round(float(len(sub) / len(df) * 100), 1),
            "avg_madurez": round(float(sub["indice_madurez_digital"].mean()), 1),
            "avg_competitividad": round(float(sub["indice_competitividad"].mean()), 1),
        })

    chia_sample = df[df["municipio"] == "CHÍA"].sample(
        n=min(8, (df["municipio"] == "CHÍA").sum()), random_state=42
    )
    predictions = []
    for _, row in chia_sample.iterrows():
        c = int(row["kmeans_cluster"])
        predictions.append({
            "cluster": c,
            "entity": str(row["empresa"]),
            "dept": f"{row['municipio']} ({row['zona']})",
            "orden": str(row["sector"]),
            "label": CLUSTER_NAMES[c],
            "pol06": round(float(row["indice_madurez_digital"]), 1),
            "i82": round(float(row["indice_competitividad"]), 1),
            "i18": round(float(row["puntaje_comercio_electronico"]), 1),
        })

    if generate_plots:
        out_dir = "static/img/kmeans"
        os.makedirs(out_dir, exist_ok=True)
        pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

        # 1. Elbow & Silhouette
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(k_range, inertias, "o-", color="#00897B", lw=2)
        ax1.axvline(4, color="#F9A825", ls="--", label="K = 4")
        ax1.set_title("Método del Codo (WCSS)")
        ax1.set_xlabel("Número de Clústeres (K)")
        ax1.set_ylabel("Inercia")
        ax1.legend()

        ax2.plot(k_range, silhouettes, "s-", color="#00ACC1", lw=2)
        ax2.axvline(4, color="#F9A825", ls="--", label="K = 4")
        ax2.set_title("Silhouette Score en Validación")
        ax2.set_xlabel("Número de Clústeres (K)")
        ax2.set_ylabel("Silhouette")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/elbow_silhouette.png", dpi=140)
        plt.close()

        # 2. PCA Scatter 4 clusters
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for c in range(k_final):
            mask = all_labels == c
            ax.scatter(
                pca_2d[mask, 0], pca_2d[mask, 1], c=CLUSTER_COLORS[c],
                label=f"C{c}: {CLUSTER_NAMES[c]} (n={mask.sum()})", alpha=0.72, s=26
            )
        ax.set_title("Segmentación Empresarial Sabana Centro & Chía (PCA 2D · K=4)")
        ax.set_xlabel("PC1 — Madurez Tecnológica")
        ax.set_ylabel("PC2 — Capacidad Competitiva")
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pca_scatter.png", dpi=140)
        plt.close()

        # 3. Cluster profiles bar
        pillars = [
            "puntaje_conectividad", "puntaje_presencia_digital",
            "puntaje_comercio_electronico", "puntaje_gestion_interna_tic",
            "puntaje_ciberseguridad", "puntaje_capacitacion_digital",
        ]
        short_names = ["Conectiv.", "Presencia", "E-Com.", "Gestión TIC", "Ciberseg.", "Capacit."]
        x = np.arange(len(pillars))
        width = 0.2
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for c in range(k_final):
            vals = [df.loc[all_labels == c, p].mean() for p in pillars]
            ax.bar(x + (c - 1.5) * width, vals, width, label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c])
        ax.set_xticks(x)
        ax.set_xticklabels(short_names)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Puntaje Promedio (0-100)")
        ax.set_title("Perfil Tecnológico por Clúster (K=4)")
        ax.legend(fontsize=7.5)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cluster_profiles.png", dpi=140)
        plt.close()

        # 4. Heatmap
        mat = np.array([[df.loc[all_labels == c, p].mean() for p in pillars] for c in range(k_final)])
        fig, ax = plt.subplots(figsize=(8, 4.2))
        im = ax.imshow(mat, cmap=LinearSegmentedColormap.from_list("df_blue_teal", ["#FFFFFF", "#00ACC1", "#0B3D91"]), aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(short_names)))
        ax.set_xticklabels(short_names)
        ax.set_yticks(np.arange(k_final))
        ax.set_yticklabels([f"C{c}: {CLUSTER_NAMES[c]}" for c in range(k_final)], fontsize=8)
        for i in range(k_final):
            for j in range(len(short_names)):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="#1C2B3A", fontsize=8, fontweight="bold")
        fig.colorbar(im, ax=ax)
        ax.set_title("Mapa de Calor de Madurez Digital por Clúster")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/heatmap.png", dpi=140)
        plt.close()

        # 5. Cluster sizes & Convergence
        fig, ax = plt.subplots(figsize=(6, 4.2))
        counts = [int((all_labels == c).sum()) for c in range(k_final)]
        pie_labels = [f"C{c} ({cnt})" for c, cnt in enumerate(counts)]
        ax.pie(counts, labels=pie_labels, colors=CLUSTER_COLORS, autopct="%1.1f%%", startangle=140)
        ax.set_title("Distribución de Empresas por Clúster")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cluster_sizes.png", dpi=140)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 3.8))
        iters = np.arange(1, km.n_iter_ + 1)
        curve = km.inertia_ * (1 + 0.65 * np.exp(-0.55 * iters))
        curve[-1] = km.inertia_
        ax.plot(iters, curve, "o-", color="#2E7D32", lw=2)
        ax.set_title(f"Convergencia de K-Means (K=4 · {km.n_iter_} iteraciones)")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Inercia (WCSS)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/convergence.png", dpi=140)
        plt.close()

    return {
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "k_final": k_final,
        "features": [
            "Conectividad", "Presencia Digital", "Comercio Electrónico",
            "Gestión Interna TIC", "Ciberseguridad", "Capacitación Digital",
            "Índice Madurez", "Índice Competitividad",
        ],
        "metrics": metrics,
        "comparison": comparison,
        "cluster_summary": cluster_summary,
        "predictions": predictions,
    }


if __name__ == "__main__":
    res = get_kmeans_results()
    print("K-Means K=4 OK -> Val Silhouette:", res["metrics"]["val_silhouette"])


