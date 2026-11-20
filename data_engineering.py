"""
data_engineering.py
Pipeline de Ingeniería de Datos para el tejido empresarial de Chía y Sabana Centro.
Procesa 'sabana_centro_digitalizacion_empresarial.csv' (Vigencia 2026) y genera
features derivadas (gap_to_sector_avg, digital_adoption_index, competitiveness_score)
y matrices estandarizadas listas para PCA, K-Means y DBSCAN.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "sabana_centro_digitalizacion_empresarial.csv"
TARGET_VIGENCIA = 2026

DIGITAL_PILLARS = [
    "puntaje_conectividad",
    "puntaje_presencia_digital",
    "puntaje_comercio_electronico",
    "puntaje_gestion_interna_tic",
    "puntaje_ciberseguridad",
    "puntaje_capacitacion_digital",
]

MODEL_FEATURES = DIGITAL_PILLARS + [
    "indice_madurez_digital",
    "indice_competitividad",
]


def load_and_engineer_data(
    csv_path: str = DATA_PATH,
    vigencia: int = TARGET_VIGENCIA,
    only_chia: bool = False,
) -> dict:
    """Ejecuta limpieza, imputación, variables derivadas y codificación."""
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset=["codigo_empresa", "vigencia"], inplace=True)

    text_cols = [
        "empresa", "departamento", "provincia", "municipio",
        "zona", "sector", "tamano_empresa"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df_v = df[df["vigencia"] == vigencia].copy()
    if only_chia:
        df_v = df_v[df_v["municipio"] == "CHÍA"].copy()

    numeric_cols = MODEL_FEATURES + [
        "num_empleados",
        "antiguedad_anos",
        "ventas_digitales_pct",
        "indice_limitacion_digital",
    ]
    for col in numeric_cols:
        if col in df_v.columns:
            df_v[col] = pd.to_numeric(
                df_v[col].astype(str).str.replace(",", "."), errors="coerce"
            )
            df_v[col] = df_v[col].fillna(
                df_v.groupby("sector")[col].transform("median")
            )
            df_v[col] = df_v[col].fillna(df_v[col].median())

    # ── Features derivadas de negocio ─────────────────────────────────────────
    df_v["digital_adoption_index"] = df_v[DIGITAL_PILLARS].mean(axis=1).round(2)
    df_v["competitiveness_score"] = df_v["indice_competitividad"].round(2)

    chia_sector_means = (
        df[df["vigencia"] == vigencia]
        .query("municipio == 'CHÍA'")
        .groupby("sector")["indice_madurez_digital"]
        .mean()
    )
    df_v["chia_sector_avg"] = df_v["sector"].map(chia_sector_means).fillna(
        df_v.groupby("sector")["indice_madurez_digital"].transform("mean")
    )
    df_v["gap_to_sector_avg"] = (
        df_v["indice_madurez_digital"] - df_v["chia_sector_avg"]
    ).round(2)
    df_v["beats_sector_avg"] = (df_v["gap_to_sector_avg"] >= 0).astype(int)

    # ── Codificación de caracterización empresarial ───────────────────────────
    tamano_map = {"MICRO": 1, "PEQUEÑA": 2, "PEQUENA": 2, "MEDIANA": 3, "GRANDE": 4}
    df_v["tamano_encoded"] = df_v["tamano_empresa"].map(tamano_map).fillna(1).astype(int)
    df_v["zona_urbana_encoded"] = df_v["zona"].str.contains("URBANA").astype(int)

    le_sector = LabelEncoder()
    df_v["sector_encoded"] = le_sector.fit_transform(df_v["sector"])

    sector_dummies = pd.get_dummies(df_v["sector"], prefix="sec", dtype=int)
    df_encoded = pd.concat(
        [df_v.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded[MODEL_FEATURES].values)

    return {
        "df": df_encoded,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "feature_cols": MODEL_FEATURES,
        "digital_pillars": DIGITAL_PILLARS,
        "n_records": int(len(df_encoded)),
        "n_chia": int((df_encoded["municipio"] == "CHÍA").sum()),
    }


if __name__ == "__main__":
    res = load_and_engineer_data()
    print(f"Dataset procesado: {res['n_records']} empresas ({res['n_chia']} en Chía)")
    print(res["df"][["empresa", "municipio", "sector", "digital_adoption_index", "gap_to_sector_avg"]].head(5))


