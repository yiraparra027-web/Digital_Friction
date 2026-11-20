"""
pca_predictor.py
Motor predictivo PCA + K-Means (4 clústeres) para empresas de Chía y Sabana Centro.
Incluye:
  - Indicadores empíricos reales medidos en el CSV para Chía (n = 280, vigencia 2026).
  - Explicaciones en lenguaje cotidiano por cada pilar del simulador.
  - Comparación sectorial en Chía (gap_to_sector_avg) y hoja de ruta prescriptiva.
  - Advertencia operativa cuando el Semáforo de Red está en Rojo (< 40) y elegibilidad
    de Alerta de Rezago Crítico.
"""

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from network_maturity import (
    METHODOLOGY_NOTE,
    NETWORK_ARCHITECTURE,
    compute_network_index,
    has_network_inputs,
)

DATA_PATH = "sabana_centro_digitalizacion_empresarial.csv"
TARGET_VIGENCIA = 2026

FEATURES = [
    "puntaje_conectividad",
    "puntaje_presencia_digital",
    "puntaje_comercio_electronico",
    "puntaje_gestion_interna_tic",
    "puntaje_ciberseguridad",
    "puntaje_capacitacion_digital",
    "indice_madurez_digital",
    "indice_competitividad",
]

PILLAR_COLS = [
    "puntaje_conectividad",
    "puntaje_presencia_digital",
    "puntaje_comercio_electronico",
    "puntaje_gestion_interna_tic",
    "puntaje_ciberseguridad",
    "puntaje_capacitacion_digital",
]

FEATURE_LABELS = {
    "puntaje_conectividad": "Conectividad y calidad de internet",
    "puntaje_presencia_digital": "Presencia digital (Web / Redes / Google)",
    "puntaje_comercio_electronico": "Comercio electrónico y pagos QR",
    "puntaje_gestion_interna_tic": "Gestión interna (Facturación / Inventario)",
    "puntaje_ciberseguridad": "Ciberseguridad y respaldo de datos",
    "puntaje_capacitacion_digital": "Capacitación TIC del equipo",
    "indice_madurez_digital": "Índice global de madurez digital",
    "indice_competitividad": "Índice de competitividad empresarial",
}

FEATURE_EVERYDAY_HINTS = {
    "puntaje_conectividad": "Qué tan rápido y estable es el internet de tu local para cobrar con datáfono o facturar sin caídas.",
    "puntaje_presencia_digital": "Si tus clientes te encuentran fácil en Google Maps, Instagram, WhatsApp Business o página web.",
    "puntaje_comercio_electronico": "Si aceptas pagos con QR, transferencias inmediatas, link de pago o pedidos por internet.",
    "puntaje_gestion_interna_tic": "Si llevas facturas, inventario y contabilidad en un programa o app (y no solo en cuaderno).",
    "puntaje_ciberseguridad": "Si proteges tus cuentas bancarias y redes con clave doble (2FA) y guardas copias de seguridad.",
    "puntaje_capacitacion_digital": "Qué tanto sabe tu equipo usar herramientas digitales para vender, atender y administrar.",
    "indice_madurez_digital": "Nivel general de uso de tecnología en el día a día de tu negocio frente a otras empresas.",
    "indice_competitividad": "Qué tan bien compite tu negocio en ventas, atención al cliente y eficiencia frente a otros del sector.",
}

N_COMPONENTS = 2
N_CLUSTERS = 4
RANDOM_STATE = 42

CLUSTER_LABELS = {
    0: "Clúster 0 — Tradicional / Rezagada",
    1: "Clúster 1 — Digitalización Básica",
    2: "Clúster 2 — En Consolidación",
    3: "Clúster 3 — Líder Digital Competitiva",
}

CLUSTER_RISKS = {
    0: "High",
    1: "Medium-High",
    2: "Medium-Low",
    3: "Low",
}

CLUSTER_COLORS = {
    0: "#C62828",
    1: "#F9A825",
    2: "#00ACC1",
    3: "#2E7D32",
}

CLUSTER_DESCRIPTIONS = {
    0: "Operación predominantemente manual o presencial con alta vulnerabilidad competitiva frente a pares digitalizados de Chía.",
    1: "Uso básico de redes sociales y pagos simples, pero sin integración de software administrativo ni comercio electrónico estructurado.",
    2: "Integra herramientas contables/administrativas y canales híbridos de venta; lista para escalar con analítica y automatización.",
    3: "Alto nivel de madurez tecnológica, seguridad y competitividad; referente sectorial en Sabana Centro.",
}

PILLAR_ACTIONS = {
    "puntaje_conectividad": {
        "titulo": "Mejorar infraestructura y estabilidad de conectividad",
        "accion": "Migrar a plan empresarial de fibra óptica simétrica con respaldo móvil 4G/5G para punto de venta en Chía.",
        "impacto": "Elimina caídas en datáfonos, facturación POS y atención en horas pico.",
    },
    "puntaje_presencia_digital": {
        "titulo": "Activar vitrina digital y posicionamiento local en Chía",
        "accion": "Verificar y optimizar Google Perfil de Negocio con horarios, fotos y reseñas, junto con catálogo activo en WhatsApp Business.",
        "impacto": "Aumenta el descubrimiento por clientes y turistas en Chía y municipios vecinos.",
    },
    "puntaje_comercio_electronico": {
        "titulo": "Habilitar cobros inmediatos y canales de venta digital",
        "accion": "Implementar pagos interoperables (QR / llaves Bre-B / Nequi-Daviplata) y pasarela de pedidos web o enlace de pago.",
        "impacto": "Reduce fricción de compra y eleva la tasa de cierre en ventas no presenciales.",
    },
    "puntaje_gestion_interna_tic": {
        "titulo": "Digitalizar facturación electrónica, inventarios y contabilidad",
        "accion": "Adoptar software administrativo en la nube (Alegra, Siigo o solución gratuita DIAN) conectado al control diario de inventario.",
        "impacto": "Ahorra horas operativas, evita quiebres de stock y cumple normativa tributaria.",
    },
    "puntaje_ciberseguridad": {
        "titulo": "Blindar cuentas financieras y respaldos de información",
        "accion": "Activar autenticación en dos pasos (2FA) en banca y redes, copias automáticas en la nube y control de accesos por empleado.",
        "impacto": "Previene fraudes digitales, secuestro de cuentas y pérdida de registros contables.",
    },
    "puntaje_capacitacion_digital": {
        "titulo": "Formar al equipo en herramientas digitales comerciales",
        "accion": "Inscribir al personal en rutas cortas prácticas de marketing digital, atención omnicanal y hojas de cálculo/IA productiva.",
        "impacto": "Garantiza que la tecnología adquirida realmente se use en la operación diaria.",
    },
}

CLUSTER_ROADMAPS = {
    0: [
        "Paso 1 (Semanas 1–2): Crear o reclamar Google Perfil de Negocio con ubicación exacta en Chía y configurar catálogo en WhatsApp Business.",
        "Paso 2 (Semanas 3–4): Activar cobros digitales con código QR y llaves de transferencia inmediata en caja.",
        "Paso 3 (Mes 2): Implementar facturación electrónica básica y registro digital de ingresos/gastos diarios.",
    ],
    1: [
        "Paso 1 (Semanas 1–3): Centralizar inventario y facturación en un software contable en la nube de bajo costo.",
        "Paso 2 (Mes 2): Incorporar enlaces de pago o tienda catálogo para pedidos a domicilio en Chía y Sabana Centro.",
        "Paso 3 (Mes 3): Activar autenticación en dos pasos (2FA) y respaldos automáticos de bases de clientes.",
    ],
    2: [
        "Paso 1 (Mes 1): Integrar CRM básico para recompra y fidelización de clientes frecuentes.",
        "Paso 2 (Mes 2): Ejecutar pauta digital geolocalizada en Chía / Cajicá / Cota y medir retorno por canal.",
        "Paso 3 (Mes 3): Automatizar reportes de margen por producto y protocolos formales de ciberseguridad.",
    ],
    3: [
        "Paso 1: Implementar analítica predictiva de demanda e inventarios basada en historial de ventas.",
        "Paso 2: Escalar comercio electrónico omnicanal con logística integrada en toda Sabana Centro y Bogotá.",
        "Paso 3: Participar como empresa mentora en redes de innovación empresarial de Chía.",
    ],
}

LOCAL_ALLIANCES = [
    {
        "entidad": "Cámara de Comercio de Bogotá — Sede Chía",
        "programa": "Ruta de Transformación Digital y Fortalecimiento Empresarial",
        "beneficio": "Diagnóstico guiado, asesoría gratuita en comercio electrónico, presencia web y ruedas de negocio en Sabana Centro.",
        "aplica_a": "Todos los clústeres (prioridad Clúster 0 y 1)",
    },
    {
        "entidad": "SENA — Centro de Desarrollo Agroempresarial (Chía)",
        "programa": "Economía Digital, Marketing y Gestión Administrativa TIC",
        "beneficio": "Cursos cortos sin costo y formación a la medida para empleados en ventas digitales, inventario y herramientas ofimáticas.",
        "aplica_a": "Clúster 0, 1 y 2",
    },
    {
        "entidad": "MinTIC Colombia",
        "programa": "MiPyme Digital / Vende Digital",
        "beneficio": "Cofinanciación y acompañamiento técnico para implementación de tiendas virtuales, pasarelas de pago y adopción tecnológica.",
        "aplica_a": "Clúster 1 y 2",
    },
    {
        "entidad": "Consultorios Empresariales Universitarios (Chía — U. de Cundinamarca / U. de la Sabana)",
        "programa": "Clínica de Analítica, Finanzas y Digitalización Pyme",
        "beneficio": "Acompañamiento personalizado por estudiantes y docentes en estructuración de costos, tableros de control y presencia digital.",
        "aplica_a": "Micro y pequeñas empresas de Chía",
    },
]

_scaler = None
_pca = None
_kmeans = None
_cluster_order = None
_feature_stats = None
_reference_points = None
_sector_benchmarks = None
_observatory_summary = None


def _bool_series_pct(series: pd.Series) -> float:
    s = series.astype(str).str.strip().str.upper()
    return round(float(s.isin(["SI", "SÍ", "1", "TRUE", "YES"]).mean() * 100), 1)


def _load_and_fit():
    global _scaler, _pca, _kmeans, _cluster_order, _feature_stats
    global _reference_points, _sector_benchmarks, _observatory_summary

    df = pd.read_csv(DATA_PATH)
    total_raw_records = int(len(df))
    df_v = df[df["vigencia"] == TARGET_VIGENCIA].copy()

    for col in FEATURES:
        df_v[col] = pd.to_numeric(
            df_v[col].astype(str).str.replace(",", "."), errors="coerce"
        )

    data = df_v.dropna(subset=FEATURES).reset_index(drop=True)

    _feature_stats = {
        col: {
            "label": FEATURE_LABELS[col],
            "everyday_hint": FEATURE_EVERYDAY_HINTS[col],
            "min": round(float(data[col].min()), 2),
            "max": round(float(data[col].max()), 2),
            "mean": round(float(data[col].mean()), 2),
        }
        for col in FEATURES
    }

    X = data[FEATURES].values
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _pca = PCA(n_components=N_COMPONENTS, svd_solver="full", random_state=RANDOM_STATE)
    X_pca = _pca.fit_transform(X_scaled)

    _kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
    raw_labels = _kmeans.fit_predict(X_pca)

    centers = _kmeans.cluster_centers_
    order = np.argsort(centers[:, 0])
    _cluster_order = {int(raw): int(rank) for rank, raw in enumerate(order)}
    ordered_labels = np.array([_cluster_order[int(r)] for r in raw_labels])

    sample_idx = data.sample(n=min(140, len(data)), random_state=RANDOM_STATE).index.values
    _reference_points = [
        {
            "x": round(float(X_pca[i, 0]), 3),
            "y": round(float(X_pca[i, 1]), 3),
            "cluster": _cluster_order[int(raw_labels[i])],
            "municipio": str(data.loc[i, "municipio"]),
            "sector": str(data.loc[i, "sector"]),
            "empresa": str(data.loc[i, "empresa"]),
        }
        for i in sample_idx
    ]

    df_chia = data[data["municipio"] == "CHÍA"].copy()
    _sector_benchmarks = {}
    for sec in sorted(data["sector"].unique()):
        sub = df_chia[df_chia["sector"] == sec]
        if len(sub) == 0:
            sub = data[data["sector"] == sec]
        _sector_benchmarks[str(sec)] = {
            "sector": str(sec),
            "n_chia": int((df_chia["sector"] == sec).sum()),
            "means": {f: round(float(sub[f].mean()), 2) for f in FEATURES},
        }

    high_crit_mask = data["nivel_limitacion_digital"].isin(["ALTA", "CRÍTICA", "CRITICA"])
    chia_high_crit_mask = df_chia["nivel_limitacion_digital"].isin(["ALTA", "CRÍTICA", "CRITICA"])

    chia_urb = df_chia[df_chia["zona"].astype(str).str.upper().str.contains("URBANA")]
    chia_rur = df_chia[~df_chia["zona"].astype(str).str.upper().str.contains("URBANA")]

    _observatory_summary = {
        "vigencia": TARGET_VIGENCIA,
        "region": "Sabana Centro — Cundinamarca",
        "focus_municipality": "CHÍA",
        "total_raw_records": total_raw_records,
        "total_enterprises": int(len(data)),
        "chia_enterprises": int(len(df_chia)),
        "chia_urban_count": int(len(chia_urb)),
        "chia_rural_count": int(len(chia_rur)),
        "avg_maturity_sabana": round(float(data["indice_madurez_digital"].mean()), 2),
        "avg_maturity_chia": round(float(df_chia["indice_madurez_digital"].mean()), 2),
        "avg_competitiveness_sabana": round(float(data["indice_competitividad"].mean()), 2),
        "avg_competitiveness_chia": round(float(df_chia["indice_competitividad"].mean()), 2),
        "high_critical_pct_sabana": round(float(high_crit_mask.mean() * 100), 1),
        "high_critical_pct_chia": round(float(chia_high_crit_mask.mean() * 100), 1),
        "chia_connectivity_urban": round(float(chia_urb["puntaje_conectividad"].mean()), 1),
        "chia_connectivity_rural": round(float(chia_rur["puntaje_conectividad"].mean()), 1),
        "chia_ecommerce_urban": round(float(chia_urb["puntaje_comercio_electronico"].mean()), 1),
        "chia_ecommerce_rural": round(float(chia_rur["puntaje_comercio_electronico"].mean()), 1),
        "chia_cybersecurity_avg": round(float(df_chia["puntaje_ciberseguridad"].mean()), 1),
        "chia_internal_tic_avg": round(float(df_chia["puntaje_gestion_interna_tic"].mean()), 1),
        "chia_infra_cost_barrier": round(float(pd.to_numeric(df_chia["barrera_costo_infraestructura"], errors="coerce").mean()), 1),
        "chia_digital_sales_pct": round(float(pd.to_numeric(df_chia["ventas_digitales_pct"], errors="coerce").mean()), 1),
        "chia_digital_payments_pct": _bool_series_pct(df_chia["usa_pagos_digitales"]),
        "chia_einvoice_pct": _bool_series_pct(df_chia["usa_facturacion_electronica"]),
        "cluster_distribution": {
            CLUSTER_LABELS[c]: int((ordered_labels == c).sum()) for c in range(N_CLUSTERS)
        },
    }


def get_feature_stats() -> dict:
    if _scaler is None:
        _load_and_fit()
    return _feature_stats


def get_sector_benchmarks() -> dict:
    if _scaler is None:
        _load_and_fit()
    return _sector_benchmarks


def get_observatory_summary() -> dict:
    if _observatory_summary is None:
        _load_and_fit()
    return _observatory_summary


def predict(values: dict) -> dict:
    t0 = time.perf_counter()
    if _scaler is None:
        _load_and_fit()

    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Faltan variables requeridas: {', '.join(missing)}")

    sector = str(values.get("sector") or next(iter(_sector_benchmarks.keys())))
    zona = str(values.get("zona") or "URBANA")
    if sector not in _sector_benchmarks:
        sector = next(iter(_sector_benchmarks.keys()))

    x = np.array([[float(values[f]) for f in FEATURES]], dtype=float)
    x_scaled = _scaler.transform(x)
    x_pca = _pca.transform(x_scaled)

    raw_label = int(_kmeans.predict(x_pca)[0])
    cluster = _cluster_order[raw_label]

    dists = np.linalg.norm(_kmeans.cluster_centers_ - x_pca, axis=1)
    ordered_dists = np.zeros(N_CLUSTERS)
    for raw, rank in _cluster_order.items():
        ordered_dists[rank] = dists[raw]

    neg = -ordered_dists
    neg -= neg.max()
    exp = np.exp(neg)
    probs = (exp / exp.sum()).tolist()
    confidence = float(probs[cluster])

    sec_bench = _sector_benchmarks[sector]
    sec_means = sec_bench["means"]
    sector_comparison = []
    negative_gaps = []

    for f in FEATURES:
        user_val = round(float(values[f]), 2)
        avg_val = round(float(sec_means[f]), 2)
        gap = round(user_val - avg_val, 2)
        sector_comparison.append({
            "feature": f,
            "label": FEATURE_LABELS[f],
            "user_value": user_val,
            "sector_avg": avg_val,
            "gap": gap,
        })
        if f in PILLAR_COLS:
            negative_gaps.append((gap, user_val, avg_val, f))

    negative_gaps.sort(key=lambda t: (t[0], t[1]))
    priority_actions = []
    for gap, user_val, avg_val, col in negative_gaps[:3]:
        act = PILLAR_ACTIONS[col]
        priority_actions.append({
            "pillar": FEATURE_LABELS[col],
            "user_value": user_val,
            "sector_avg": avg_val,
            "gap": gap,
            "titulo": act["titulo"],
            "accion": act["accion"],
            "impacto": act["impacto"],
        })

    loadings = [
        {
            "feature": FEATURE_LABELS[f],
            "column": f,
            "pc1": round(float(_pca.components_[0, i]), 4),
            "pc2": round(float(_pca.components_[1, i]), 4),
        }
        for i, f in enumerate(FEATURES)
    ]

    overall_gap = round(
        float(values["indice_madurez_digital"]) - float(sec_means["indice_madurez_digital"]),
        2,
    )

    network = None
    adjusted_maturity = None
    operational_warning = None
    if has_network_inputs(values):
        network = compute_network_index(values)
        base_mat = float(values["indice_madurez_digital"])
        adjusted_maturity = round(max(0.0, min(100.0, base_mat + network["adjustment"])), 2)
        network["base_maturity"] = round(base_mat, 2)
        network["adjusted_maturity"] = adjusted_maturity
        network["limits_cluster"] = bool(network["bottleneck"] and cluster >= 1)
        if network["limits_cluster"]:
            operational_warning = (
                f"Aunque tus indicadores comerciales ubican al negocio en {CLUSTER_LABELS[cluster]}, "
                f"tu conectividad está en Semáforo Rojo ({network['index']}/100). En la práctica, "
                "las caídas de red frenan la operación en caja con un riesgo equivalente al Clúster 0."
            )

    is_net_red = bool(network and network["bottleneck"])
    critical_eligible = bool(cluster == 0 or is_net_red)
    if is_net_red and cluster == 0:
        critical_reason = "Empresa en Clúster 0 (Rezago Crítico) y Semáforo de Red en Rojo (< 40/100)."
    elif is_net_red:
        critical_reason = f"Semáforo de Red en Rojo ({network['index']}/100): cuello de botella crítico de infraestructura."
    elif cluster == 0:
        critical_reason = "Empresa clasificada en Clúster 0 — Tradicional / Rezagada frente a su sector en Chía."
    else:
        critical_reason = "No presenta rezago crítico inmediato; se recomienda enviar el informe preventivo completo."

    top_action_text = (
        network["top_urgent_action"]
        if is_net_red
        else (
            f"{priority_actions[0]['titulo']}: {priority_actions[0]['accion']}"
            if priority_actions
            else CLUSTER_ROADMAPS[cluster][0]
        )
    )

    return {
        "cluster": cluster,
        "label": CLUSTER_LABELS[cluster],
        "risk": CLUSTER_RISKS[cluster],
        "color": CLUSTER_COLORS[cluster],
        "description": CLUSTER_DESCRIPTIONS[cluster],
        "operational_warning": operational_warning,
        "pc1": round(float(x_pca[0, 0]), 4),
        "pc2": round(float(x_pca[0, 1]), 4),
        "pc1_variance": round(float(_pca.explained_variance_ratio_[0] * 100), 2),
        "pc2_variance": round(float(_pca.explained_variance_ratio_[1] * 100), 2),
        "confidence": round(confidence, 4),
        "cluster_probs": [round(p, 4) for p in probs],
        "pca_components": loadings,
        "reference_points": _reference_points,
        "sector_info": {
            "sector": sector,
            "zona": zona,
            "n_chia": sec_bench["n_chia"],
            "user_maturity": round(float(values["indice_madurez_digital"]), 2),
            "sector_maturity_avg": sec_means["indice_madurez_digital"],
            "overall_gap": overall_gap,
            "comparison": sector_comparison,
        },
        "prescriptive_plan": {
            "cluster_steps": CLUSTER_ROADMAPS[cluster],
            "priority_actions": priority_actions,
        },
        "network": network,
        "network_methodology": METHODOLOGY_NOTE,
        "network_architecture": NETWORK_ARCHITECTURE[cluster],
        "local_alliances": LOCAL_ALLIANCES,
        "critical_alert": {
            "eligible": critical_eligible,
            "reason": critical_reason,
            "priority_action": top_action_text,
        },
        "server_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


_load_and_fit()