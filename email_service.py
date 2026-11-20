import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

# Parámetros de control de tráfico (Rate Limiting por IP + Sesión)
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 600  # 10 minutos

# Control de concurrencia saliente hacia Brevo y almacenamiento en memoria
_OUTBOUND_SEMAPHORE = threading.Semaphore(3)
_HISTORY_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()

_DISPATCH_HISTORY: deque = deque(maxlen=50)
_RATE_BUCKETS: Dict[str, List[float]] = {}

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

EMAIL_ARCHITECTURE_DOCS: Dict[str, Any] = {
    "1_recepcion_y_control_de_cuota": {
        "validacion_entrada": "Verifica que el correo destinatario cumpla con el formato estándar RFC-5322 y sanea el nombre de la empresa.",
        "proteccion_anti_spam": f"Aplica un límite de {RATE_LIMIT_MAX_REQUESTS} envíos cada {RATE_LIMIT_WINDOW_SECONDS // 60} minutos combinando la IP del cliente y su identificador de sesión.",
    },
    "2_diagnostico_ml_y_plantilla_html": {
        "evaluacion_sectorial": "Calcula el puntaje del Semáforo de Red, el Índice de Madurez Digital y el clúster empresarial (PCA + K-Means) para Chía.",
        "renderizado_ejecutivo": "Construye un documento HTML autocontenido con estilos en línea compatibles con Gmail, Outlook y clientes móviles.",
    },
    "3_cola_asincrona_en_flask": {
        "respuesta_inmediata": "Delega el envío a un hilo en segundo plano y responde al instante HTTP 202 Accepted para no bloquear la navegación.",
        "control_de_trafico": "Gestiona hasta 3 conexiones salientes simultáneas mediante semáforo y reintenta automáticamente ante fallos temporales de red.",
    },
    "4_entrega_transaccional_brevo": {
        "pasarela_api_v3": "Transmite el correo por HTTPS/TLS hacia POST /v3/smtp/email autenticándose con la llave BREVO_API_KEY.",
        "auditoria_y_telemetria": "Registra en tiempo real el código HTTP final (201 entregado, 401 llave inválida, 429 límite excedido) y la latencia en milisegundos.",
    },
}


def _clean_env(var_name: str, default: str = "") -> str:
    raw = os.environ.get(var_name, default) or default
    return raw.strip().strip('"').strip("'")


def _enmascarar_email(correo: str) -> str:
    if "@" not in correo:
        return correo
    usuario, dominio = correo.split("@", 1)
    if len(usuario) <= 2:
        prefijo = usuario[0] + "••" if usuario else "••"
    else:
        prefijo = usuario[:2] + "••••"
    return f"{prefijo}@{dominio}"


def validar_email(email: str) -> bool:
    if not isinstance(email, str):
        return False
    correo = email.strip()
    if len(correo) < 6 or len(correo) > 254:
        return False
    return bool(EMAIL_REGEX.match(correo))


def verificar_rate_limit(ip: str, session_id: str) -> Tuple[bool, str, int]:
    clave = f"{ip}:{session_id}"
    ahora = time.time()
    limite_inicio = ahora - RATE_LIMIT_WINDOW_SECONDS

    with _RATE_LOCK:
        marcas = _RATE_BUCKETS.get(clave, [])
        marcas = [t for t in marcas if t > limite_inicio]

        if len(marcas) >= RATE_LIMIT_MAX_REQUESTS:
            _RATE_BUCKETS[clave] = marcas
            espera_seg = max(1, int(marcas[0] + RATE_LIMIT_WINDOW_SECONDS - ahora))
            return (
                False,
                f"Has alcanzado el máximo de {RATE_LIMIT_MAX_REQUESTS} envíos por sesión/IP. Podrás enviar nuevamente en {espera_seg} segundos.",
                0,
            )

        marcas.append(ahora)
        _RATE_BUCKETS[clave] = marcas
        restantes = max(0, RATE_LIMIT_MAX_REQUESTS - len(marcas))
        return True, "", restantes


def estado_configuracion_brevo() -> Dict[str, Any]:
    api_key = _clean_env("BREVO_API_KEY")
    sender_email = _clean_env("BREVO_SENDER_EMAIL") or _clean_env("MAIL_DEFAULT_SENDER")
    sender_name = _clean_env("BREVO_SENDER_NAME", "Observatorio Digital Friction Chía")

    missing_vars: List[str] = []
    if not api_key:
        missing_vars.append("BREVO_API_KEY")
    if not sender_email:
        missing_vars.append("BREVO_SENDER_EMAIL")

    warnings: List[str] = []
    if api_key.startswith("xsmtpsib-"):
        warnings.append("La clave configurada es SMTP (xsmtpsib-); para la API REST debes usar una API Key v3 (xkeysib-).")
    if sender_email.lower().endswith("@tudominio.com") or sender_email.lower().endswith("@example.com"):
        warnings.append("El correo remitente es de ejemplo; reemplázalo en .env por un remitente verificado en Brevo.")

    configured = len(missing_vars) == 0
    masked = _enmascarar_email(sender_email) if sender_email else "No configurado"
    return {
        "configured": configured,
        "missing_vars": missing_vars,
        "warnings": warnings,
        "sender": masked,
        "sender_email": sender_email,
        "sender_masked": masked,
        "sender_name": sender_name,
        "api_url": BREVO_API_URL,
        "rate_limit_max": RATE_LIMIT_MAX_REQUESTS,
    }


def _registrar_historial(
    destinatario: str,
    tipo: str,
    estado: str,
    http_code: int,
    latency_ms: float,
    detalle: str,
) -> None:
    registro = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "destinatario": _enmascarar_email(destinatario),
        "tipo": tipo,
        "estado": estado,
        "http_code": http_code,
        "latency_ms": round(latency_ms, 1),
        "detalle": detalle,
    }
    with _HISTORY_LOCK:
        _DISPATCH_HISTORY.appendleft(registro)


def get_dispatch_history() -> List[Dict[str, Any]]:
    with _HISTORY_LOCK:
        return list(_DISPATCH_HISTORY)


def _extraer_metricas_diagnostico(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza las métricas del modelo PCA / K-Means para construir el reporte."""
    inputs = diag.get("inputs") if isinstance(diag.get("inputs"), dict) else {}

    sector = str(diag.get("sector") or inputs.get("sector") or "COMERCIO AL POR MENOR").upper()
    zona = str(diag.get("zona") or inputs.get("zona") or "URBANA").upper()

    cluster_id = diag.get("cluster", 0)
    cluster_nombre = str(
        diag.get("cluster_label")
        or diag.get("cluster_name")
        or diag.get("segmento")
        or f"Clúster {cluster_id}"
    )

    madurez = float(
        diag.get("indice_madurez_digital")
        or inputs.get("indice_madurez_digital")
        or diag.get("madurez_digital")
        or 34.0
    )
    competitividad = float(
        diag.get("indice_competitividad")
        or inputs.get("indice_competitividad")
        or diag.get("competitividad")
        or 38.0
    )

    red_score_raw = (
        diag.get("network_score")
        or diag.get("puntaje_red")
        or diag.get("semaforo_puntaje")
    )
    if red_score_raw is not None:
        red_score = float(red_score_raw)
    else:
        down = float(diag.get("red_download_mbps") or inputs.get("red_download_mbps") or 20)
        up = float(diag.get("red_upload_mbps") or inputs.get("red_upload_mbps") or 5)
        lat = float(diag.get("red_latencia_ms") or inputs.get("red_latencia_ms") or 95)
        failover = bool(diag.get("red_failover") if "red_failover" in diag else inputs.get("red_failover", False))
        vlan = bool(diag.get("red_vlan") if "red_vlan" in diag else inputs.get("red_vlan", False))
        cifrado = bool(diag.get("red_cifrado") if "red_cifrado" in diag else inputs.get("red_cifrado", False))

        score_bw = min(35.0, (down / 100.0) * 20.0 + (up / 50.0) * 15.0)
        score_lat = max(0.0, 25.0 - max(0.0, lat - 20.0) * 0.18)
        score_sec = (15.0 if failover else 0.0) + (12.5 if vlan else 0.0) + (12.5 if cifrado else 0.0)
        red_score = round(min(100.0, max(5.0, score_bw + score_lat + score_sec)), 1)

    if red_score < 45:
        semaforo_color = "#dc2626"
        semaforo_bg = "#fef2f2"
        semaforo_estado = "Semáforo de Red en Rojo · Cuello de botella crítico de infraestructura"
    elif red_score < 72:
        semaforo_color = "#d97706"
        semaforo_bg = "#fffbeb"
        semaforo_estado = "Semáforo de Red en Amarillo · Capacidad intermedia con riesgo operativo"
    else:
        semaforo_color = "#15803d"
        semaforo_bg = "#f0fdf4"
        semaforo_estado = "Semáforo de Red en Verde · Infraestructura preparada para escalar"

    accion_prioritaria = str(
        diag.get("accion_prioritaria")
        or diag.get("priority_action")
        or (
            "Activar un plan B automático ante caídas de internet (Multi-WAN): conectar un router de doble entrada "
            "con fibra principal + módem celular 4G/5G de respaldo y segmentar la red interna mediante VLAN."
            if red_score < 60
            else "Mantener monitoreo continuo de latencia, respaldos automáticos en la nube y canales digitales activos."
        )
    )

    arquitectura = str(
        diag.get("arquitectura_recomendada")
        or (
            "Arquitectura Pyme Resiliente (Chía): Enlace principal de fibra óptica simétrica + respaldo celular 4G/5G "
            "en router Dual-WAN con failover automático, segmentación VLAN (POS / Oficina / Invitados) y cifrado WPA3."
        )
    )

    recomendaciones = diag.get("recomendaciones") or diag.get("recommendations")
    if not isinstance(recomendaciones, list) or not recomendaciones:
        recomendaciones = [
            "Garantizar continuidad de conexión mediante fibra óptica empresarial y respaldo móvil automático (Multi-WAN).",
            "Separar la red de facturación electrónica y datáfonos (POS) de la red WiFi de visitantes mediante VLANs.",
            "Habilitar canales de venta y atención digital integrados con medios de pago electrónicos y logística en Sabana Centro.",
            "Vincular al personal en rutas gratuitas de apropiación TIC, comercio electrónico y ciberseguridad básica.",
        ]

    return {
        "sector": sector,
        "zona": zona,
        "cluster_nombre": cluster_nombre,
        "madurez": round(madurez, 1),
        "competitividad": round(competitividad, 1),
        "red_score": round(red_score, 1),
        "semaforo_color": semaforo_color,
        "semaforo_bg": semaforo_bg,
        "semaforo_estado": semaforo_estado,
        "accion_prioritaria": accion_prioritaria,
        "arquitectura": arquitectura,
        "recomendaciones": recomendaciones,
    }


def construir_html_informe_diagnostico(diag: Dict[str, Any], nombre_empresa: str) -> str:
    empresa = html.escape(nombre_empresa or "Tu Empresa en Chía")
    m = _extraer_metricas_diagnostico(diag)
    fecha_gen = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    items_rec = "".join(
        f'<li style="margin-bottom:8px;color:#334155;">{html.escape(str(r))}</li>'
        for r in m["recomendaciones"]
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Diagnóstico Completo de Madurez Digital — {empresa}</title>
</head>
<body style="margin:0;padding:0;background-color:#f8fafc;font-family:Arial,Helvetica,sans-serif;color:#0f172a;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f8fafc;padding:24px 12px;">
    <tr>
      <td align="center">
        <table role="presentation" width="640" cellspacing="0" cellpadding="0" style="max-width:640px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;">
          <tr>
            <td style="background:#0b4a9f;padding:28px 32px;color:#ffffff;">
              <div style="font-size:12px;letter-spacing:0.6px;text-transform:uppercase;color:#bfdbfe;font-weight:bold;">
                Digital Friction · Observatorio de Competitividad Empresarial en Chía
              </div>
              <h1 style="margin:8px 0 4px 0;font-size:22px;line-height:1.3;">
                Diagnóstico Ejecutivo de Digitalización y Conectividad
              </h1>
              <p style="margin:0;font-size:14px;color:#e0f2fe;">
                Empresa: <strong>{empresa}</strong> · Sector: {html.escape(m["sector"])} ({html.escape(m["zona"])})
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:24px 32px 12px 32px;">
              <p style="margin:0 0 16px 0;font-size:13px;color:#475569;line-height:1.5;">
                Este informe resume el estado actual de tu infraestructura tecnológica y madurez digital frente al tejido empresarial de Chía, junto con las acciones prioritarias recomendadas.
              </p>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td width="32%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;text-align:center;">
                    <div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:bold;">Semáforo de Red</div>
                    <div style="font-size:22px;font-weight:bold;color:{m['semaforo_color']};margin:4px 0;">{m['red_score']}/100</div>
                    <div style="font-size:11px;color:#475569;">Capacidad y Respaldo</div>
                  </td>
                  <td width="2%"></td>
                  <td width="32%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;text-align:center;">
                    <div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:bold;">Madurez Digital</div>
                    <div style="font-size:22px;font-weight:bold;color:#0b4a9f;margin:4px 0;">{m['madurez']}/100</div>
                    <div style="font-size:11px;color:#475569;">Adopción TIC</div>
                  </td>
                  <td width="2%"></td>
                  <td width="32%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;text-align:center;">
                    <div style="font-size:11px;color:#64748b;text-transform:uppercase;font-weight:bold;">Competitividad</div>
                    <div style="font-size:22px;font-weight:bold;color:#15803d;margin:4px 0;">{m['competitividad']}/100</div>
                    <div style="font-size:11px;color:#475569;">Perfil Sectorial Chía</div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <tr>
            <td style="padding:12px 32px;">
              <div style="background:{m['semaforo_bg']};border-left:4px solid {m['semaforo_color']};border-radius:8px;padding:16px;">
                <div style="font-size:14px;font-weight:bold;color:{m['semaforo_color']};margin-bottom:6px;">
                  {html.escape(m['semaforo_estado'])} · {html.escape(m['cluster_nombre'])}
                </div>
                <div style="font-size:13px;color:#1e293b;line-height:1.5;">
                  <strong>Acción prioritaria:</strong> {html.escape(m['accion_prioritaria'])}
                </div>
              </div>
            </td>
          </tr>

          <tr>
            <td style="padding:12px 32px;">
              <h2 style="font-size:16px;margin:0 0 8px 0;color:#0f172a;">Arquitectura Tecnológica Recomendada</h2>
              <p style="margin:0;font-size:13px;line-height:1.55;color:#334155;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;">
                {html.escape(m['arquitectura'])}
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:12px 32px;">
              <h2 style="font-size:16px;margin:0 0 8px 0;color:#0f172a;">Plan de Mejora Sugerido</h2>
              <ul style="margin:0;padding-left:20px;font-size:13px;line-height:1.5;">
                {items_rec}
              </ul>
            </td>
          </tr>

          <tr>
            <td style="padding:12px 32px 24px 32px;">
              <div style="background:#e8f1ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px 16px;">
                <div style="font-size:13px;font-weight:bold;color:#0b4a9f;margin-bottom:6px;">
                  Contactos y Programas de Apoyo en Chía
                </div>
                <div style="font-size:12px;color:#1e3a8a;line-height:1.55;">
                  • <strong>Cámara de Comercio de Bogotá (Sede Chía):</strong> Ruta de transformación digital y fortalecimiento comercial.<br>
                  • <strong>SENA (Centro de Desarrollo Agroempresarial Chía):</strong> Capacitación técnica sin costo en herramientas digitales y seguridad informática.<br>
                  • <strong>MinTIC:</strong> Programas de productividad tecnológica y diagnóstico digital para MiPymes.
                </div>
              </div>
            </td>
          </tr>

          <tr>
            <td style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:16px 32px;font-size:11px;color:#64748b;text-align:center;">
              Enviado por <strong>Digital Friction Chía</strong> · {fecha_gen}
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def construir_html_alerta_critica(diag: Dict[str, Any], nombre_empresa: str) -> str:
    empresa = html.escape(nombre_empresa or "Tu Empresa en Chía")
    m = _extraer_metricas_diagnostico(diag)
    fecha_gen = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Alerta de Rezago Crítico Detectada — {empresa}</title>
</head>
<body style="margin:0;padding:0;background-color:#fef2f2;font-family:Arial,Helvetica,sans-serif;color:#0f172a;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#fef2f2;padding:24px 12px;">
    <tr>
      <td align="center">
        <table role="presentation" width="620" cellspacing="0" cellpadding="0" style="max-width:620px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;border:2px solid #dc2626;">
          <tr>
            <td style="background:#dc2626;padding:24px 28px;color:#ffffff;">
              <div style="font-size:12px;font-weight:bold;letter-spacing:0.6px;text-transform:uppercase;color:#fee2e2;">
                Alerta Prioritaria · Digital Friction Chía
              </div>
              <h1 style="margin:8px 0 4px 0;font-size:21px;">
                Alerta de Rezago Crítico Detectada
              </h1>
              <p style="margin:0;font-size:14px;color:#fef2f2;">
                Empresa: <strong>{empresa}</strong> · Sector: {html.escape(m['sector'])}
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:24px 28px;">
              <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:16px;margin-bottom:18px;">
                <div style="font-size:15px;font-weight:bold;color:#b91c1c;margin-bottom:6px;">
                  {html.escape(m['semaforo_estado'])} ({m['red_score']}/100)
                </div>
                <p style="margin:0;font-size:13px;color:#7f1d1d;line-height:1.55;">
                  Tu diagnóstico ubica a <strong>{empresa}</strong> en el perfil <strong>{html.escape(m['cluster_nombre'])}</strong>
                  (Madurez Digital: <strong>{m['madurez']}/100</strong>). Este nivel representa un cuello de botella operativo
                  que puede interrumpir ventas electrónicas, datáfonos y facturación ante fallas de conectividad.
                </p>
              </div>

              <h2 style="font-size:15px;margin:0 0 8px 0;color:#0f172a;">Acción Prioritaria Recomendada</h2>
              <div style="background:#f8fafc;border-left:4px solid #dc2626;padding:14px 16px;border-radius:6px;font-size:13px;line-height:1.55;color:#1e293b;margin-bottom:18px;">
                {html.escape(m['accion_prioritaria'])}
              </div>

              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="font-size:12px;border-collapse:collapse;">
                <tr>
                  <td style="padding:10px;border:1px solid #e2e8f0;background:#f8fafc;font-weight:bold;width:38%;">Arquitectura Sugerida</td>
                  <td style="padding:10px;border:1px solid #e2e8f0;color:#334155;">{html.escape(m['arquitectura'])}</td>
                </tr>
                <tr>
                  <td style="padding:10px;border:1px solid #e2e8f0;background:#f8fafc;font-weight:bold;">Acompañamiento Local</td>
                  <td style="padding:10px;border:1px solid #e2e8f0;color:#334155;">CCB Sede Chía · SENA Sabana Centro · MinTIC</td>
                </tr>
              </table>
            </td>
          </tr>

          <tr>
            <td style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:14px 28px;font-size:11px;color:#64748b;text-align:center;">
              Notificación automática · Digital Friction Chía · {fecha_gen}
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def enviar_correo_brevo(
    destinatario: str,
    asunto: str,
    html_content: str,
    tipo: str = "Informe Completo",
) -> Dict[str, Any]:
    """Despacha un correo transaccional vía Brevo API v3 con control de concurrencia y telemetría."""
    t0 = time.perf_counter()
    cfg = estado_configuracion_brevo()

    if not cfg["configured"]:
        faltantes = ", ".join(cfg["missing_vars"])
        latency = (time.perf_counter() - t0) * 1000
        detalle = f"Faltan variables en .env: {faltantes}"
        _registrar_historial(destinatario, tipo, "Sin credenciales (.env)", 503, latency, detalle)
        return {"ok": False, "http_code": 503, "latency_ms": round(latency, 1), "detalle": detalle}

    api_key = _clean_env("BREVO_API_KEY")
    sender_email = cfg["sender_email"]
    sender_name = cfg["sender_name"]

    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": destinatario}],
        "subject": asunto,
        "htmlContent": html_content,
        "tags": ["digital-friction-chia", tipo.lower().replace(" ", "-")],
    }
    body_bytes = json.dumps(payload).encode("utf-8")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": api_key,
        "user-agent": "DigitalFrictionChia/1.0",
    }

    with _OUTBOUND_SEMAPHORE:
        max_intentos = 2
        for intento in range(max_intentos):
            t_req = time.perf_counter()
            req = urllib.request.Request(
                BREVO_API_URL,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    status_code = resp.getcode()
                    resp_text = resp.read().decode("utf-8", errors="replace")
                    latency = (time.perf_counter() - t_req) * 1000
                    detalle = resp_text or "Aceptado por Brevo"
                    _registrar_historial(
                        destinatario,
                        tipo,
                        f"Enviado (HTTP {status_code})",
                        status_code,
                        latency,
                        detalle,
                    )
                    return {
                        "ok": True,
                        "http_code": status_code,
                        "latency_ms": round(latency, 1),
                        "detalle": detalle,
                    }

            except urllib.error.HTTPError as e:
                status_code = e.code
                err_body = e.read().decode("utf-8", errors="replace")
                latency = (time.perf_counter() - t_req) * 1000

                if status_code >= 500 and intento < max_intentos - 1:
                    time.sleep(1.2)
                    continue

                if status_code == 401:
                    sugerencia = (
                        "Usa una API Key v3 (xkeysib-...) en BREVO_API_KEY, no una clave SMTP."
                        if api_key.startswith("xsmtpsib-")
                        else "Verifica que BREVO_API_KEY en .env sea válida y esté activa en Brevo."
                    )
                    detalle = f"{err_body} | {sugerencia}"
                elif status_code == 400 and "sender" in err_body.lower():
                    detalle = f"{err_body} | Verifica que {sender_email} esté autorizado en Brevo -> Senders."
                else:
                    detalle = err_body or str(e)

                _registrar_historial(
                    destinatario,
                    tipo,
                    f"Error Brevo (HTTP {status_code})",
                    status_code,
                    latency,
                    detalle,
                )
                return {
                    "ok": False,
                    "http_code": status_code,
                    "latency_ms": round(latency, 1),
                    "detalle": detalle,
                }

            except Exception as e:
                latency = (time.perf_counter() - t_req) * 1000
                if intento < max_intentos - 1:
                    time.sleep(1.2)
                    continue
                detalle = f"Error de conexión hacia Brevo: {e}"
                _registrar_historial(
                    destinatario,
                    tipo,
                    "Error de Red",
                    0,
                    latency,
                    detalle,
                )
                return {
                    "ok": False,
                    "http_code": 0,
                    "latency_ms": round(latency, 1),
                    "detalle": detalle,
                }

    return {"ok": False, "http_code": 0, "latency_ms": 0.0, "detalle": "Error inesperado en cola"}