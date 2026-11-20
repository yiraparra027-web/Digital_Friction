"""
PCA_Model.py
Reducción de dimensionalidad con PCA sobre los indicadores digitales y competitivos
del tejido empresarial de Chía y Sabana Centro (Vigencia 2026).
PC1: Madurez Tecnológica | PC2: Capacidad Competitiva.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from data_engineering import load_and_engineer_data

FEATURE_LABELS = {
    "puntaje_conectividad": "Conectividad Digital",
    "puntaje_presencia_digital": "Presencia Digital",
    "puntaje_comercio_electronico": "Comercio Electrónico",
    "puntaje_gestion_interna_tic": "Gestión Interna TIC",
    "puntaje_ciberseguridad": "Ciberseguridad",
    "puntaje_capacitacion_digital": "Capacitación Digital",
    "indice_madurez_digital": "Índice de Madurez Digital",
    "indice_competitividad": "Índice de Competitividad",
}


def get_pca_results(export_json: bool = True, generate_plots: bool = True) -> dict:
    bundle = load_and_engineer_data()
    df = bundle["df"].reset_index(drop=True)
    X_scaled = bundle["X_scaled"]
    feature_cols = bundle["feature_cols"]

    X_train, X_test = train_test_split(X_scaled, test_size=0.20, random_state=42)

    pca_full = PCA(n_components=len(feature_cols), random_state=42)
    pca_full.fit(X_train)

    pca = PCA(n_components=2, svd_solver="full", random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    Z_all = pca.transform(X_scaled)

    X_train_rec = pca.inverse_transform(Z_train)
    X_test_rec = pca.inverse_transform(Z_test)
    train_mse = float(mean_squared_error(X_train, X_train_rec))
    test_mse = float(mean_squared_error(X_test, X_test_rec))
    train_mae = float(mean_absolute_error(X_train, X_train_rec))
    test_mae = float(mean_absolute_error(X_test, X_test_rec))

    ks1_stat, ks1_pval = ks_2samp(Z_train[:, 0], Z_test[:, 0])
    ks2_stat, ks2_pval = ks_2samp(Z_train[:, 1], Z_test[:, 1])

    var_ind = [round(float(v * 100), 2) for v in pca_full.explained_variance_ratio_]
    var_cum = [round(float(v * 100), 2) for v in np.cumsum(pca_full.explained_variance_ratio_)]
    retained_var = round(float(sum(var_ind[:2])), 2)

    # Comparación de agrupamiento: espacio original (8D) vs espacio PCA (2D)
    km_orig = KMeans(n_clusters=4, init="k-means++", n_init=20, random_state=42)
    labels_orig = km_orig.fit_predict(X_scaled)

    km_pca = KMeans(n_clusters=4, init="k-means++", n_init=20, random_state=42)
    labels_pca = km_pca.fit_predict(Z_all)

    sil_orig = float(silhouette_score(X_scaled, labels_orig))
    db_orig = float(davies_bouldin_score(X_scaled, labels_orig))
    ch_orig = float(calinski_harabasz_score(X_scaled, labels_orig))

    sil_pca = float(silhouette_score(Z_all, labels_pca))
    db_pca = float(davies_bouldin_score(Z_all, labels_pca))
    ch_pca = float(calinski_harabasz_score(Z_all, labels_pca))

    imp_sil = round(((sil_pca - sil_orig) / max(sil_orig, 1e-6)) * 100, 1)
    imp_db = round(((db_orig - db_pca) / max(db_orig, 1e-6)) * 100, 1)
    imp_ch = round(((ch_pca - ch_orig) / max(ch_orig, 1e-6)) * 100, 1)
    avg_imp = round(float(np.mean([imp_sil, imp_db, imp_ch])), 1)

    ari = round(float(adjusted_rand_score(labels_orig, labels_pca)), 4)
    nmi = round(float(normalized_mutual_info_score(labels_orig, labels_pca)), 4)
    cm = confusion_matrix(labels_orig, labels_pca)
    row_ind, col_ind = linear_sum_assignment(-cm)
    overall_agree = round(float(cm[row_ind, col_ind].sum() / len(df) * 100), 1)
    disagree = round(100.0 - overall_agree, 1)

    loadings = []
    for i, col in enumerate(feature_cols):
        l1 = float(pca.components_[0, i])
        l2 = float(pca.components_[1, i])
        loadings.append({
            "column": col,
            "name": FEATURE_LABELS[col],
            "loading_pc1": round(l1, 4),
            "loading_pc2": round(l2, 4),
            "abs_loading_pc1": round(abs(l1), 4),
            "abs_loading_pc2": round(abs(l2), 4),
            "interpretation_pc1": "Contribución positiva" if l1 >= 0 else "Contribución negativa",
            "interpretation_pc2": "Contribución positiva" if l2 >= 0 else "Contribución negativa",
        })

    sample_df = df.iloc[:250].copy()
    sample_z = Z_all[:250]
    scatter_points = [
        {
            "pc1": round(float(sample_z[i, 0]), 3),
            "pc2": round(float(sample_z[i, 1]), 3),
            "empresa": str(sample_df.iloc[i]["empresa"]),
            "municipio": str(sample_df.iloc[i]["municipio"]),
            "sector": str(sample_df.iloc[i]["sector"]),
            "quintil": int(sample_df.iloc[i]["quintil_grupo_sector"]),
        }
        for i in range(len(sample_df))
    ]

    if generate_plots:
        out_dir = "static/img/pca"
        os.makedirs(out_dir, exist_ok=True)

        # 1. Gráfico de sedimentación (Varianza individual y acumulada)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        comps = [f"CP{i+1}" for i in range(len(var_ind))]
        ax.bar(comps, var_ind, color="#00897B", alpha=0.85, label="Varianza individual (%)")
        ax.plot(comps, var_cum, "o-", color="#F9A825", lw=2.2, label="Varianza acumulada (%)")
        ax.axhline(95, color="#00ACC1", ls="--", lw=1.2, label="Umbral 95 %")
        ax.set_ylim(0, 108)
        ax.set_ylabel("Porcentaje de varianza (%)")
        ax.set_title("Varianza Explicada por Componente Principal (Chía y Sabana Centro)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/scree_variance.png", dpi=140)
        plt.close()

        # 2. Biplot de cargas factoriales
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(Z_all[:, 0], Z_all[:, 1], c="#607D8B", alpha=0.35, s=16, label="Empresas (2026)")
        for item in loadings:
            vx, vy = item["loading_pc1"] * 5.2, item["loading_pc2"] * 2.2
            ax.arrow(0, 0, vx, vy, color="#C62828", alpha=0.85, head_width=0.14, lw=1.4)
            ax.text(vx * 1.08, vy * 1.08, item["name"], fontsize=7.5, color="#1C2B3A", fontweight="bold")
        ax.axhline(0, color="#CFD8DC", lw=1)
        ax.axvline(0, color="#CFD8DC", lw=1)
        ax.set_xlabel(f"CP1 — Madurez Tecnológica ({var_ind[0]} %)")
        ax.set_ylabel(f"CP2 — Capacidad Competitiva ({var_ind[1]} %)")
        ax.set_title("Biplot PCA: Proyección Empresarial y Vectores de Indicadores")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/loadings_biplot.png", dpi=140)
        plt.close()

        # 3. Consistencia Entrenamiento vs. Prueba (Evaluación)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
        ax1.hist(Z_train[:, 0], bins=24, density=True, alpha=0.6, color="#00897B", label="Entrenamiento (80 %)")
        ax1.hist(Z_test[:, 0], bins=24, density=True, alpha=0.6, color="#00ACC1", label="Prueba (20 %)")
        ax1.set_title(f"Distribución CP1 (p-valor KS = {ks1_pval:.3f})")
        ax1.set_xlabel("Puntuación CP1")
        ax1.set_ylabel("Densidad")
        ax1.legend(fontsize=8)

        ax2.hist(Z_train[:, 1], bins=24, density=True, alpha=0.6, color="#00897B", label="Entrenamiento (80 %)")
        ax2.hist(Z_test[:, 1], bins=24, density=True, alpha=0.6, color="#F9A825", label="Prueba (20 %)")
        ax2.set_title(f"Distribución CP2 (p-valor KS = {ks2_pval:.3f})")
        ax2.set_xlabel("Puntuación CP2")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/reconstruction_consistency.png", dpi=140)
        plt.close()

        # 4. Comparación de calidad de agrupamiento (8D vs PCA 2D)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        metric_names = ["Coef. Silueta (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz / 1000 (↑)"]
        orig_vals = [sil_orig, db_orig, ch_orig / 1000.0]
        pca_vals = [sil_pca, db_pca, ch_pca / 1000.0]
        x = np.arange(len(metric_names))
        w = 0.34
        ax.bar(x - w / 2, orig_vals, w, label="Espacio Original (8 variables)", color="#607D8B")
        ax.bar(x + w / 2, pca_vals, w, label="Espacio Reducido PCA (2 componentes)", color="#00897B")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=8.5)
        ax.set_title("Impacto del PCA en la Calidad de Agrupamiento")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/clustering_comparison.png", dpi=140)
        plt.close()

    results = {
        "total_entities": int(len(df)),
        "chia_entities": int((df["municipio"] == "CHÍA").sum()),
        "n_features": len(feature_cols),
        "n_components": 2,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "axis_meaning": {
            "pc1": "Madurez Tecnológica (adopción integral de herramientas TIC)",
            "pc2": "Capacidad Competitiva (eficiencia operativa, comercio y seguridad)",
        },
        "pca_variance_individual": var_ind,
        "pca_variance_cumulative": var_cum,
        "total_retained_variance": retained_var,
        "pca_components": pca.components_.tolist(),
        "reconstruction": {
            "train_mse": round(train_mse, 4),
            "test_mse": round(test_mse, 4),
            "train_mae": round(train_mae, 4),
            "test_mae": round(test_mae, 4),
            "information_preserved": retained_var,
        },
        "clustering_metrics": {
            "original_space": {
                "silhouette": round(sil_orig, 4),
                "davies_bouldin": round(db_orig, 4),
                "calinski_harabasz": round(ch_orig, 2),
            },
            "pca_space": {
                "silhouette": round(sil_pca, 4),
                "davies_bouldin": round(db_pca, 4),
                "calinski_harabasz": round(ch_pca, 2),
            },
            "improvements": {
                "silhouette": imp_sil,
                "davies_bouldin": imp_db,
                "calinski_harabasz": imp_ch,
            },
            "average_improvement": avg_imp,
        },
        "agreement": {
            "ari": ari,
            "nmi": nmi,
            "overall": overall_agree,
            "overall_agreement": overall_agree,
            "disagreement": disagree,
            "matrix": cm.tolist(),
        },
        "validation": {
            "ks_test": {
                "pc1_stat": round(float(ks1_stat), 4),
                "pc1_pvalue": round(float(ks1_pval), 4),
                "pc2_stat": round(float(ks2_stat), 4),
                "pc2_pvalue": round(float(ks2_pval), 4),
                "consistent": bool(ks1_pval > 0.05),
            },
            "train_stats": {
                "pc1_mean": round(float(Z_train[:, 0].mean()), 4),
                "pc1_std": round(float(Z_train[:, 0].std()), 4),
                "pc2_mean": round(float(Z_train[:, 1].mean()), 4),
                "pc2_std": round(float(Z_train[:, 1].std()), 4),
            },
            "test_stats": {
                "pc1_mean": round(float(Z_test[:, 0].mean()), 4),
                "pc1_std": round(float(Z_test[:, 0].std()), 4),
                "pc2_mean": round(float(Z_test[:, 1].mean()), 4),
                "pc2_std": round(float(Z_test[:, 1].std()), 4),
            },
        },
        "feature_loadings": loadings,
        "feature_names": [FEATURE_LABELS[c] for c in feature_cols],
        "scatter_points": scatter_points,
    }

    if export_json:
        os.makedirs("static/data", exist_ok=True)
        with open("static/data/pca_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


