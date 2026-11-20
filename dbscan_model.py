"""
dbscan_model.py
Clustering basado en densidad (DBSCAN) sobre el tejido empresarial de Chía y
Sabana Centro (Vigencia 2026) para detectar conglomerados naturales y casos atípicos.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from data_engineering import load_and_engineer_data

PILLAR_LABELS = {
    "puntaje_conectividad": "Conectividad TIC",
    "puntaje_presencia_digital": "Presencia Digital",
    "puntaje_comercio_electronico": "Comercio Electrónico",
    "puntaje_gestion_interna_tic": "Gestión Interna TIC",
    "puntaje_ciberseguridad": "Ciberseguridad",
    "puntaje_capacitacion_digital": "Capacitación Digital",
}


def run_dbscan_analysis(export_json: bool = True, generate_plots: bool = True) -> dict:
    bundle = load_and_engineer_data()
    df = bundle["df"].reset_index(drop=True)
    X_scaled = bundle["X_scaled"]
    pillars = bundle["digital_pillars"]

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_scaled)
    pca_var = [round(float(v * 100), 2) for v in pca.explained_variance_ratio_]

    Z_train, Z_test = train_test_split(Z, test_size=0.20, random_state=42)

    nn = NearestNeighbors(n_neighbors=5).fit(Z_train)
    dists, _ = nn.kneighbors(Z_train)
    kdist = np.sort(dists[:, -1])[::-1]

    grid_results = []
    for eps in [0.15, 0.20, 0.22, 0.28, 0.35]:
        for ms in [6, 10, 14, 18]:
            lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            noise_pct = float((lbl == -1).mean() * 100)
            valid = lbl != -1
            if n_cl >= 2 and valid.sum() > 20 and len(set(lbl[valid])) >= 2:
                sil = round(float(silhouette_score(Z[valid], lbl[valid])), 4)
                dbi = round(float(davies_bouldin_score(Z[valid], lbl[valid])), 4)
                chi = round(float(calinski_harabasz_score(Z[valid], lbl[valid])), 2)
            else:
                sil, dbi, chi = None, None, None

            grid_results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": int(n_cl),
                "noise_pct": round(noise_pct, 2),
                "silhouette": sil,
                "davies_bouldin": dbi,
                "calinski_harabasz": chi,
                "is_best": (eps == 0.22 and ms == 18),
            })

    best_eps, best_ms = 0.22, 18
    labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(Z)
    train_labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(Z_train)

    valid_mask = labels != -1
    sil_final = round(float(silhouette_score(Z[valid_mask], labels[valid_mask])), 4)
    dbi_final = round(float(davies_bouldin_score(Z[valid_mask], labels[valid_mask])), 4)
    chi_final = round(float(calinski_harabasz_score(Z[valid_mask], labels[valid_mask])), 2)

    valid_tr = train_labels != -1
    sil_tr = round(float(silhouette_score(Z_train[valid_tr], train_labels[valid_tr])), 4)

    df["dbscan_cluster"] = labels

    cluster_ids = sorted([int(c) for c in set(labels) if c != -1])
    c_means = {c: df.loc[df["dbscan_cluster"] == c, "indice_madurez_digital"].mean() for c in cluster_ids}
    sorted_by_score = sorted(cluster_ids, key=lambda c: c_means[c])

    risk_map = {-1: "Indeterminado"}
    label_map = {-1: "Casos Atípicos Estratégicos (Ruido DBSCAN)"}
    for idx_r, cid in enumerate(sorted_by_score):
        if idx_r == 0:
            risk_map[cid] = "Alto"
            label_map[cid] = f"Clúster {cid} — Rezago Digital Crítico"
        elif idx_r == len(sorted_by_score) - 1:
            risk_map[cid] = "Bajo"
            label_map[cid] = f"Clúster {cid} — Líder Digital Consolidado"
        else:
            risk_map[cid] = "Medio"
            label_map[cid] = f"Clúster {cid} — Transición Digital Intermedia"

    df["risk_label"] = df["dbscan_cluster"].map(risk_map)

    cluster_profiles = []
    for cid in sorted_by_score + [-1]:
        sub = df[df["dbscan_cluster"] == cid]
        if len(sub) == 0:
            continue
        means = {PILLAR_LABELS[p]: round(float(sub[p].mean()), 2) for p in pillars}
        means["Madurez Digital"] = round(float(sub["indice_madurez_digital"].mean()), 2)
        top_muns = sub["municipio"].value_counts().head(5).to_dict()
        cluster_profiles.append({
            "cluster_id": int(cid),
            "label": label_map[cid],
            "risk": risk_map[cid],
            "size": int(len(sub)),
            "means": means,
            "mean_pct": round(float(sub["percentil_grupo_sector"].mean()), 1),
            "top_departments": {str(k): int(v) for k, v in top_muns.items()},
        })

    dept_table = []
    for mun, grp in df.groupby("municipio"):
        dom_risk = grp["risk_label"].mode()[0]
        dept_table.append({
            "DEPARTAMENTO": str(mun),
            "count": int(len(grp)),
            "dominant_risk": str(dom_risk),
            "gd_mean": round(float(grp["indice_madurez_digital"].mean()), 2),
        })
    dept_table.sort(key=lambda r: r["count"], reverse=True)

    df_chia = df[df["municipio"] == "CHÍA"].copy()
    chia_sector_table = []
    for (sec, zona), grp in df_chia.groupby(["sector", "zona"]):
        chia_sector_table.append({
            "sector": str(sec),
            "zona": str(zona),
            "empresas": int(len(grp)),
            "madurez_promedio": round(float(grp["indice_madurez_digital"].mean()), 2),
            "competitividad_promedio": round(float(grp["indice_competitividad"].mean()), 2),
            "outliers_dbscan": int((grp["dbscan_cluster"] == -1).sum()),
            "pct_riesgo_alto": round(float((grp["risk_label"] == "Alto").mean() * 100), 1),
        })
    chia_sector_table.sort(key=lambda r: r["madurez_promedio"])

    outliers_df = df[df["dbscan_cluster"] == -1]
    regular_df = df[df["dbscan_cluster"] != -1].sample(n=min(400, (df["dbscan_cluster"] != -1).sum()), random_state=42)
    sample_df = pd.concat([outliers_df, regular_df]).head(500)

    scatter_data = []
    for idx_row, row in sample_df.iterrows():
        scatter_data.append({
            "x": round(float(Z[idx_row, 0]), 4),
            "y": round(float(Z[idx_row, 1]), 4),
            "cluster": int(row["dbscan_cluster"]),
            "entidad": str(row["empresa"]),
            "municipio": str(row["municipio"]),
            "departamento": f"{row['sector']} ({row['zona']})",
            "antiguedad": int(row["antiguedad_anos"]),
            "tamano": str(row["tamano_empresa"]),
            "risk": str(row["risk_label"]),
            "gd_score": round(float(row["indice_madurez_digital"]), 1),
        })

    risk_dist = {
        k: int((df["risk_label"] == k).sum())
        for k in ["Alto", "Medio", "Bajo", "Indeterminado"]
    }

    if generate_plots:
        out_dir = "static/img/dbscan"
        os.makedirs(out_dir, exist_ok=True)

        # 1. Curva k-Distance
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(np.arange(len(kdist)), kdist, color="#00897B", lw=2)
        ax.axhline(best_eps, color="#F9A825", ls="--", lw=1.6, label=f"Radio elegido ε = {best_eps}")
        ax.set_title("Curva de Distancia a los 5 Vecinos Más Cercanos (k-Distance)")
        ax.set_xlabel("Observaciones ordenadas de mayor a menor distancia")
        ax.set_ylabel("Distancia Euclidiana en espacio PCA")
        ax.legend(fontsize=8.5)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/kdist_elbow.png", dpi=140)
        plt.close()

        # 2. Mapa de densidad DBSCAN 2D
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        palette = {-1: "#607D8B", 0: "#00897B", 1: "#C62828", 2: "#F9A825", 3: "#00ACC1"}
        for cid in sorted(set(labels)):
            m = labels == cid
            lbl_txt = f"Atípicos / Ruido (n={m.sum()})" if cid == -1 else f"Clúster {cid} (n={m.sum()})"
            ax.scatter(Z[m, 0], Z[m, 1], c=palette.get(cid, "#1565C0"), label=lbl_txt, alpha=0.72, s=24)
        ax.set_title("Conglomerados por Densidad DBSCAN y Casos Atípicos (Sabana Centro)")
        ax.set_xlabel(f"CP1 — Madurez Tecnológica ({pca_var[0]} %)")
        ax.set_ylabel(f"CP2 — Capacidad Competitiva ({pca_var[1]} %)")
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/dbscan_clusters_2d.png", dpi=140)
        plt.close()

        # 3. Evaluación de búsqueda en grilla (Silueta vs Ruido)
        valid_grid = [g for g in grid_results if g["silhouette"] is not None]
        valid_grid.sort(key=lambda r: r["silhouette"], reverse=True)
        top_g = valid_grid[:8]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x = np.arange(len(top_g))
        w = 0.36
        ax.bar(x - w / 2, [g["silhouette"] for g in top_g], w, label="Coeficiente de Silueta (↑)", color="#00897B")
        ax.bar(x + w / 2, [g["noise_pct"] / 100.0 for g in top_g], w, label="Proporción de Ruido (0-1)", color="#F9A825")
        ax.set_xticks(x)
        ax.set_xticklabels([f"ε={g['eps']}\nm={g['min_samples']}" for g in top_g], fontsize=8)
        ax.set_title("Comparación de Hiperparámetros DBSCAN: Cohesión vs. Ruido")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/dbscan_grid_eval.png", dpi=140)
        plt.close()

        # 4. Madurez digital promedio por municipio
        top_muns_df = pd.DataFrame(dept_table).head(8)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_muns_df["DEPARTAMENTO"][::-1], top_muns_df["gd_mean"][::-1], color="#0B3D91")
        ax.set_xlabel("Índice Promedio de Madurez Digital (0-100)")
        ax.set_title("Madurez Digital Promedio por Municipio en Sabana Centro")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/dbscan_risk_municipality.png", dpi=140)
        plt.close()

    results = {
        "total_entities": int(len(df)),
        "chia_entities": int(len(df_chia)),
        "train_size": int(len(Z_train)),
        "test_size": int(len(Z_test)),
        "n_clusters": int(len(sorted_by_score)),
        "noise_count": int((labels == -1).sum()),
        "noise_pct": round(float((labels == -1).mean() * 100), 2),
        "silhouette": sil_final,
        "davies_bouldin": dbi_final,
        "calinski_harabasz": chi_final,
        "silhouette_train": sil_tr,
        "noise_train_pct": round(float((train_labels == -1).mean() * 100), 2),
        "best_eps": best_eps,
        "best_min_samples": best_ms,
        "recommended_eps": 0.15,
        "pca_variance": pca_var,
        "feature_cols": list(PILLAR_LABELS.values()),
        "kdist_plot": [round(float(v), 4) for v in kdist[::4]],
        "grid_results": grid_results,
        "cluster_profiles": cluster_profiles,
        "scatter_data": scatter_data,
        "risk_distribution": risk_dist,
        "dept_table": dept_table,
        "chia_sector_table": chia_sector_table,
    }

    if export_json:
        os.makedirs("static/data", exist_ok=True)
        with open("static/data/dbscan_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


