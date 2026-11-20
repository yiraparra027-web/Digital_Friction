"""
network_maturity.py
Índice de Madurez de Red y Conectividad + catálogo de Arquitectura y Soluciones
de Comunicación de Datos por perfil de madurez (clúster).

Decisión de diseño y transparencia metodológica
-----------------------------------------------
1. El modelo PCA (8 variables) + K-Means (K=4) se entrena exclusivamente con
   ``sabana_centro_digitalizacion_empresarial.csv``, dataset que NO contiene
   telemetría de capas física, enlace o red (VLAN, QoS, TLS 1.3, failover).
2. Para preservar la validez estadística del modelo empírico, las 7 entradas
   de infraestructura se calculan en un índice heurístico complementario (0-100)
   basado en criterios de ingeniería de comunicación de datos (SLA para POS en
   la nube, facturación electrónica y seguridad transaccional).
3. En la interfaz empresarial el resultado se presenta mediante un SEMÁFORO
   operativo (Verde / Amarillo / Rojo) y en lenguaje cotidiano, dejando los
   pesos técnicos explícitos como supuesto de diseño.
"""

METHODOLOGY_NOTE = {
    "naturaleza": "Heurística de diseño técnico (criterio de ingeniería, no regresión del CSV)",
    "justificacion": (
        "El archivo sabana_centro_digitalizacion_empresarial.csv mide adopción TIC y "
        "competitividad, pero no incluye variables de red local ni última milla "
        "(VLAN, cifrado TLS, enlaces de respaldo o latencia). Por ello, los pesos "
        "(25 % medio, 20 % ancho de banda, 15 % latencia, 15 % VLAN, 15 % cifrado, "
        "10 % failover) y los umbrales (50 Mbps subida, 100 Mbps bajada, 20–200 ms) "
        "se definieron bajo criterios operativos de disponibilidad y seguridad "
        "transaccional para MiPymes."
    ),
    "interaccion_modelo": (
        "No altera las 8 variables de entrada del PCA/K-Means ni invalida sus métricas "
        "(Silueta, Davies-Bouldin). Ajusta la madurez mostrada en ±8 puntos y, cuando "
        "el semáforo de red está en Rojo (< 40 pts), activa una alerta de cuello de "
        "botella que advierte degradación operativa equivalente al Clúster 0."
    ),
}

MEDIUM_OPTIONS = {
    "FTTH": {
        "label": "Fibra óptica (FTTH / dedicada)",
        "everyday": "Internet rápido y estable por cable de fibra hasta el local.",
        "score": 100,
        "nota": "Canal guiado, baja latencia y alta simetría.",
    },
    "RADIOENLACE": {
        "label": "Radioenlace inalámbrico (antena externa)",
        "everyday": "Antena en el techo apuntando a una torre del proveedor (común en veredas).",
        "score": 65,
        "nota": "Depende de línea de vista e interferencia climática o en banda libre.",
    },
    "MOVIL_FWA": {
        "label": "Internet móvil / módem con SIM (4G / 5G)",
        "everyday": "Módem inalámbrico con tarjeta SIM o compartir datos desde el celular.",
        "score": 55,
        "nota": "Latencia y velocidad variables según la congestión de la antena celular.",
    },
    "COBRE_ADSL": {
        "label": "Cable telefónico antiguo / Cobre (ADSL)",
        "everyday": "Conexión tradicional por línea telefónica; suele ser lenta al enviar datos.",
        "score": 25,
        "nota": "Alta asimetría y ancho de banda de subida muy limitado.",
    },
}
DEFAULT_MEDIUM = "COBRE_ADSL"

# Pesos del índice (suman 1.0 — supuesto heurístico de diseño técnico)
WEIGHTS = {
    "medio": 0.25,
    "ancho_banda": 0.20,
    "latencia": 0.15,
    "vlan": 0.15,
    "cifrado": 0.15,
    "failover": 0.10,
}

MAX_ADJUSTMENT = 8.0
NEUTRAL_POINT = 50.0
BOTTLENECK_THRESHOLD = 40.0

UPLOAD_TARGET_MBPS = 50.0
DOWNLOAD_TARGET_MBPS = 100.0
LATENCY_BEST_MS = 20.0
LATENCY_WORST_MS = 200.0


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def _num(values, key, default):
    try:
        v = values.get(key, default)
        return float(default if v in (None, "") else v)
    except (TypeError, ValueError):
        return float(default)


def _bool(values, key):
    v = values.get(key, False)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "si", "sí", "on")
    return bool(v)


def has_network_inputs(values: dict) -> bool:
    keys = (
        "red_medio",
        "red_upload_mbps",
        "red_download_mbps",
        "red_latencia_ms",
        "red_vlan",
        "red_cifrado",
        "red_failover",
    )
    return any(k in values for k in keys)


def compute_network_index(values: dict) -> dict:
    """Calcula el Índice de Madurez de Red y su Semáforo Operativo."""
    medio_key = str(values.get("red_medio") or DEFAULT_MEDIUM)
    if medio_key not in MEDIUM_OPTIONS:
        medio_key = DEFAULT_MEDIUM
    medio = MEDIUM_OPTIONS[medio_key]

    up = _clamp(_num(values, "red_upload_mbps", 5), 0, 10_000)
    down = _clamp(_num(values, "red_download_mbps", 20), 0, 10_000)
    lat = _clamp(_num(values, "red_latencia_ms", 80), 0, 5_000)

    bw_score = _clamp(
        0.6 * (up / UPLOAD_TARGET_MBPS) * 100
        + 0.4 * (down / DOWNLOAD_TARGET_MBPS) * 100
    )
    lat_score = _clamp(
        (LATENCY_WORST_MS - lat) / (LATENCY_WORST_MS - LATENCY_BEST_MS) * 100
    )
    vlan = _bool(values, "red_vlan")
    cifrado = _bool(values, "red_cifrado")
    failover = _bool(values, "red_failover")

    comp = {
        "medio": medio["score"],
        "ancho_banda": round(bw_score, 1),
        "latencia": round(lat_score, 1),
        "vlan": 100 if vlan else 0,
        "cifrado": 100 if cifrado else 0,
        "failover": 100 if failover else 0,
    }
    index = round(sum(comp[k] * WEIGHTS[k] for k in WEIGHTS), 1)

    if index >= 70:
        semaforo = "verde"
        semaforo_titulo = "Semáforo Verde · Red lista para operar"
        semaforo_resumen = (
            "Tu conexión y seguridad básica soportan datáfonos, facturación electrónica "
            "y ventas digitales sin frenar la operación diaria."
        )
        level, risk = "Robusta", "Low"
    elif index >= BOTTLENECK_THRESHOLD:
        semaforo = "amarillo"
        semaforo_titulo = "Semáforo Amarillo · Funciona, pero con riesgos en hora pico"
        semaforo_resumen = (
            "Tu internet permite operar hoy, pero ante alta afluencia de clientes o una "
            "caída del proveedor podrías perder ventas o sufrir demoras en pagos."
        )
        level, risk = "Intermedia", "Medium-High"
    else:
        semaforo = "rojo"
        semaforo_titulo = "Semáforo Rojo · Cuello de botella crítico"
        semaforo_resumen = (
            "Tu infraestructura actual frena la digitalización: alto riesgo de caídas en "
            "datáfonos, fallos de facturación y exposición de datos comerciales."
        )
        level, risk = "Crítica", "High"

    adjustment = round((index - NEUTRAL_POINT) / NEUTRAL_POINT * MAX_ADJUSTMENT, 2)
    adjustment = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment))

    pillars = [
        ("medio", "Tipo de conexión a internet", "Cable de fibra, antena o datos móviles"),
        ("ancho_banda", "Velocidad de subida y bajada", "Capacidad para enviar facturas y cobros"),
        ("latencia", "Rapidez de respuesta (latencia)", "Tiempo que tarda en responder el datáfono"),
        ("vlan", "Wi-Fi separado (VLAN)", "Clientes separados de la red de caja y POS"),
        ("cifrado", "Conexión segura (HTTPS / TLS)", "Candado digital que protege cobros y claves"),
        ("failover", "Plan B si se cae el internet", "Respaldo automático con datos celulares"),
    ]
    components = [
        {
            "key": k,
            "label": lbl,
            "everyday": desc,
            "score": comp[k],
            "weight_pct": int(round(WEIGHTS[k] * 100)),
        }
        for k, lbl, desc in pillars
    ]

    findings = []
    short_flags = []
    if not failover:
        short_flags.append("Sin respaldo de internet (failover)")
        findings.append((
            "critical",
            "Sin plan B si se cae el internet: si falla tu operador principal, se detienen datáfonos, pagos QR y facturación electrónica.",
        ))
    if not cifrado:
        short_flags.append("Sin cifrado activo (TLS/HTTPS)")
        findings.append((
            "critical",
            "Sin candado de cifrado activo: los datos de cobros y claves administrativas viajan expuestos en la red.",
        ))
    if not vlan:
        short_flags.append("Wi-Fi sin separar (sin VLAN)")
        findings.append((
            "warning",
            "Wi-Fi compartido sin separación: los celulares de clientes usan la misma red que la caja, poniendo lentos los datáfonos en horas pico.",
        ))
    if lat > 100:
        short_flags.append(f"Respuesta lenta ({lat:.0f} ms)")
        findings.append((
            "warning",
            f"Respuesta lenta de red ({lat:.0f} ms): puede provocar errores o demoras al pasar tarjetas y emitir facturas.",
        ))
    if up < 10:
        short_flags.append(f"Subida baja ({up:g} Mbps)")
        findings.append((
            "warning",
            f"Velocidad de subida baja ({up:g} Mbps): insuficiente para sincronizar inventarios en la nube y facturar rápido.",
        ))
    if medio_key in ("COBRE_ADSL", "MOVIL_FWA", "RADIOENLACE"):
        findings.append((
            "info",
            f"Conexión actual «{medio['label']}»: {medio['everyday']}",
        ))

    recommendations = []
    if not failover:
        recommendations.append({
            "titulo": "Activar un plan B automático si se cae el internet (Multi-WAN)",
            "accion": "Conectar un router doble entrada con fibra principal + módem celular 4G/5G de respaldo automático.",
            "impacto": "Evita perder ventas con tarjeta o QR cuando falla el operador en Chía.",
        })
    if medio_key in ("COBRE_ADSL", "RADIOENLACE", "MOVIL_FWA"):
        recommendations.append({
            "titulo": "Migrar a conexión de fibra óptica empresarial",
            "accion": "Solicitar fibra óptica (FTTH) o, en zona veredal sin fibra, antena direccional dedicada con sistema que guarde ventas sin conexión.",
            "impacto": "Reduce caídas y acelera datáfonos y facturación electrónica.",
        })
    if not vlan:
        recommendations.append({
            "titulo": "Separar el Wi-Fi de clientes del Wi-Fi de la caja (VLAN + prioridad)",
            "accion": "Configurar dos redes distintas en el router y dar prioridad de velocidad a la caja y datáfonos.",
            "impacto": "Evita que el internet del negocio se ponga lento cuando hay muchos clientes conectados.",
        })
    if not cifrado:
        recommendations.append({
            "titulo": "Activar cifrado de seguridad en sistemas y accesos",
            "accion": "Usar únicamente plataformas web con candado HTTPS / TLS 1.3 y clave segura WPA2/WPA3 en el router.",
            "impacto": "Protege cuentas bancarias, facturación y datos de clientes frente a intrusos.",
        })

    top_urgent_action = (
        recommendations[0]["titulo"] + ": " + recommendations[0]["accion"]
        if recommendations
        else "Mantener monitoreo preventivo y pruebas periódicas del respaldo de conectividad."
    )

    return {
        "index": index,
        "semaforo": semaforo,
        "semaforo_titulo": semaforo_titulo,
        "semaforo_resumen": semaforo_resumen,
        "short_flags": short_flags,
        "top_urgent_action": top_urgent_action,
        "level": level,
        "risk": risk,
        "adjustment": adjustment,
        "bottleneck": index < BOTTLENECK_THRESHOLD,
        "is_heuristic": True,
        "methodology_note": METHODOLOGY_NOTE,
        "components": components,
        "inputs": {
            "medio": medio_key,
            "medio_label": medio["label"],
            "upload_mbps": up,
            "download_mbps": down,
            "latencia_ms": lat,
            "vlan": vlan,
            "cifrado": cifrado,
            "failover": failover,
        },
        "findings": [{"severity": s, "text": t} for s, t in findings],
        "recommendations": recommendations[:4],
    }


NETWORK_ARCHITECTURE = {
    0: {
        "perfil": "Clúster 0 · Tradicional / Rezagada (Prioridad: Conectividad Básica Estable)",
        "icono": "bi-reception-1",
        "resumen": "Prioridad: lograr una conexión estable en el local o vereda que no detenga cobros ni registros.",
        "soluciones": [
            "Internet fijo inalámbrico (FWA 4G/5G) con antena externa de alta ganancia en zonas sin fibra.",
            "Radioenlace punto a punto dedicado cuando el local está en zona veredal de Chía.",
            "Aplicaciones de caja e inventario que funcionen sin internet y sincronicen al recuperar señal (modo offline-first).",
        ],
        "tecnologias": ["FWA 4G/5G", "Radioenlace PtP", "Modo Offline-first", "Sincronización asíncrona"],
    },
    1: {
        "perfil": "Clúster 1 · Digitalización Básica (Prioridad: Ordenar y Proteger el Wi-Fi)",
        "icono": "bi-diagram-2",
        "resumen": "Prioridad: separar la red de clientes de la caja y asegurar los pagos digitales.",
        "soluciones": [
            "Separar redes (VLAN): un Wi-Fi exclusivo para datáfonos/caja y otro aparte para clientes e invitados.",
            "Prioridad de tráfico (QoS) para que los pagos con tarjeta nunca se frenen por descargas o videos.",
            "Uso obligatorio de páginas y pasarelas cifradas (HTTPS / TLS 1.3).",
        ],
        "tecnologias": ["Wi-Fi separado (VLAN)", "Prioridad QoS", "HTTPS", "TLS 1.3"],
    },
    2: {
        "perfil": "Clúster 2 · En Consolidación (Prioridad: Cero Caídas en Hora Pico)",
        "icono": "bi-hdd-network",
        "resumen": "Prioridad: continuidad permanente de ventas, facturación e inventarios en la nube.",
        "soluciones": [
            "Router doble conexión (Multi-WAN) con cambio automático si falla el operador principal.",
            "Combinación de fibra óptica empresarial + respaldo celular LTE/5G activo.",
            "Alta disponibilidad para pasarelas de pago y sincronización continua de inventarios.",
        ],
        "tecnologias": ["Doble internet (Multi-WAN)", "Balanceo de carga", "Respaldo automático (Failover)", "LTE/5G backup"],
    },
    3: {
        "perfil": "Clúster 3 · Líder Digital Competitiva (Prioridad: Integración entre Sedes)",
        "icono": "bi-globe-americas",
        "resumen": "Prioridad: conectar múltiples puntos de venta en Sabana Centro con datos en tiempo real.",
        "soluciones": [
            "Red inteligente SD-WAN para administrar varios enlaces y sedes desde un panel central.",
            "Túneles privados cifrados (VPN Site-to-Site) entre bodega, puntos de venta y administración.",
            "Conexión automática entre tienda web, inventario y contabilidad mediante APIs REST/JSON.",
        ],
        "tecnologias": ["SD-WAN", "VPN entre sedes", "APIs REST/JSON", "Telemetría en tiempo real"],
    },
}