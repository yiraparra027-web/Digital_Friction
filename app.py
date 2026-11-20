import os
import threading
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

load_dotenv()

from dbscan_model import run_dbscan_analysis
from email_service import (
    EMAIL_ARCHITECTURE_DOCS,
    construir_html_alerta_critica,
    construir_html_informe_diagnostico,
    enviar_correo_brevo,
    estado_configuracion_brevo,
    get_dispatch_history,
    validar_email,
    verificar_rate_limit,
)
from kmeans_model import get_kmeans_results
from network_maturity import MEDIUM_OPTIONS, METHODOLOGY_NOTE, NETWORK_ARCHITECTURE
from PCA_Model import get_pca_results
from pca_predictor import (
    get_feature_stats,
    get_observatory_summary,
    get_sector_benchmarks,
    predict as pca_predict,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "digital-friction-chia-2026-secret")


def _get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def _get_client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"


@app.route("/")
def home():
    summary = get_observatory_summary()
    return render_template("index.html", summary=summary)


@app.route("/crispml")
def crispml():
    return render_template("crispml.html")


@app.route("/business-understanding")
def business_understanding():
    summary = get_observatory_summary()
    return render_template("business_understanding.html", summary=summary)


@app.route("/data-understanding")
def data_understanding():
    return render_template("data_understanding.html")


@app.route("/data-engineering")
def data_engineering():
    return render_template(
        "data_engineering.html",
        network_methodology=METHODOLOGY_NOTE,
    )


@app.route("/dbscan-model")
def dbscan_model():
    results = run_dbscan_analysis(export_json=True, generate_plots=True)
    return render_template("dbscan_model.html", results=results)


@app.route("/PCA-Model")
def PCA_Model():
    results = get_pca_results(export_json=True, generate_plots=True)
    return render_template("PCA_Model.html", results=results)


@app.route("/kmeans")
def kmeans():
    results = get_kmeans_results(generate_plots=True)
    return render_template(
        "kmeans.html", results=results, network_architecture=NETWORK_ARCHITECTURE
    )


@app.route("/dbscan-evaluation")
def dbscan_evaluation():
    results = run_dbscan_analysis(export_json=True, generate_plots=True)
    return render_template("dbscan_evaluation.html", results=results)


@app.route("/pca-evaluation")
def pca_evaluation():
    results = get_pca_results(export_json=True, generate_plots=True)
    return render_template("PCA_evaluation.html", results=results)


@app.route("/kmeans-evaluation")
def kmeans_evaluation():
    results = get_kmeans_results(generate_plots=True)
    return render_template("kmeans_evaluation.html", results=results)


@app.route("/pca-prediction")
def pca_prediction():
    stats = get_feature_stats()
    features = list(stats.keys())
    sectors = list(get_sector_benchmarks().keys())
    email_cfg = estado_configuracion_brevo()
    return render_template(
        "pca_prediction.html",
        features=features,
        stats=stats,
        sectors=sectors,
        medium_options=MEDIUM_OPTIONS,
        network_methodology=METHODOLOGY_NOTE,
        email_cfg=email_cfg,
    )


@app.route("/pca-predict", methods=["POST"])
def pca_predict_api():
    values = request.get_json(force=True)
    try:
        result = pca_predict(values)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/email_service")
def email_service_page():
    email_cfg = estado_configuracion_brevo()
    history = get_dispatch_history()
    sectors = list(get_sector_benchmarks().keys())
    return render_template(
        "email_service.html",
        email_cfg=email_cfg,
        arch_docs=EMAIL_ARCHITECTURE_DOCS,
        history=history,
        sectors=sectors,
    )


def _resolver_diagnostico_desde_payload(data: dict) -> dict:
    if isinstance(data.get("diagnostico"), dict) and "cluster" in data["diagnostico"]:
        return data["diagnostico"]
    sim_inputs = data.get("inputs") or {
        "sector": data.get("sector", "COMERCIO AL POR MENOR"),
        "zona": data.get("zona", "URBANA"),
        "puntaje_conectividad": 32,
        "puntaje_presencia_digital": 28,
        "puntaje_comercio_electronico": 22,
        "puntaje_gestion_interna_tic": 26,
        "puntaje_ciberseguridad": 20,
        "puntaje_capacitacion_digital": 25,
        "indice_madurez_digital": 26,
        "indice_competitividad": 30,
        "red_medio": "COBRE_ADSL",
        "red_upload_mbps": 3,
        "red_download_mbps": 15,
        "red_latencia_ms": 135,
        "red_vlan": False,
        "red_cifrado": False,
        "red_failover": False,
    }
    return pca_predict(sim_inputs)


@app.route("/preview-correo", methods=["POST"])
def preview_correo():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    nombre = str(data.get("nombre") or "Mi Empresa en Chía").strip()
    tipo = str(data.get("tipo") or "informe").strip().lower()
    try:
        diag = _resolver_diagnostico_desde_payload(data)
        html_out = (
            construir_html_alerta_critica(diag, nombre)
            if tipo == "alerta"
            else construir_html_informe_diagnostico(diag, nombre)
        )
        return jsonify({
            "ok": True,
            "tipo": tipo,
            "html": html_out,
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/enviar-informe", methods=["POST"])
def enviar_informe():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    destinatario = str(data.get("email") or "").strip()
    nombre = str(data.get("nombre") or "Tu Empresa").strip()

    if not validar_email(destinatario):
        return jsonify({
            "error": "Ingresa un correo electrónico válido (ejemplo: gerencia@tuempresa.com).",
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        }), 400

    permitido, err_msg, restantes = verificar_rate_limit(_get_client_ip(), _get_session_id())
    if not permitido:
        return jsonify({
            "error": err_msg,
            "remaining": 0,
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        }), 429

    try:
        diag = _resolver_diagnostico_desde_payload(data)
    except Exception as e:
        return jsonify({"error": f"No se pudo construir el diagnóstico: {e}"}), 400

    html_out = construir_html_informe_diagnostico(diag, nombre)
    asunto = f"Informe de Madurez Digital y Conectividad — {nombre} (Chía)"

    cfg = estado_configuracion_brevo()
    if not cfg["configured"]:
        faltantes = ", ".join(cfg["missing_vars"])
        enviar_correo_brevo(destinatario, asunto, html_out, tipo="Informe Completo")
        return jsonify({
            "status": "sin_credenciales",
            "configured": False,
            "mensaje": f"No se envió el correo real porque faltan variables de entorno ({faltantes}). Puedes revisar la vista previa completa del informe.",
            "missing_vars": cfg["missing_vars"],
            "remaining": restantes,
            "preview_html": html_out,
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        }), 503

    thread = threading.Thread(
        target=enviar_correo_brevo,
        args=(destinatario, asunto, html_out, "Informe Completo"),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "en_cola",
        "configured": True,
        "mensaje": f"Informe en cola de envío asíncrono (HTTP 202 Accepted) hacia {destinatario}.",
        "remaining": restantes,
        "preview_html": html_out,
        "server_ms": round((time.perf_counter() - t0) * 1000, 2),
    }), 202


@app.route("/enviar-alerta", methods=["POST"])
def enviar_alerta():
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    destinatario = str(data.get("email") or "").strip()
    nombre = str(data.get("nombre") or "Tu Empresa").strip()

    if not validar_email(destinatario):
        return jsonify({
            "error": "Ingresa un correo electrónico válido para enviar la alerta.",
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        }), 400

    permitido, err_msg, restantes = verificar_rate_limit(_get_client_ip(), _get_session_id())
    if not permitido:
        return jsonify({
            "error": err_msg,
            "remaining": 0,
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        }), 429

    try:
        diag = _resolver_diagnostico_desde_payload(data)
    except Exception as e:
        return jsonify({"error": f"No se pudo evaluar la alerta: {e}"}), 400

    html_out = construir_html_alerta_critica(diag, nombre)
    asunto = f"[ALERTA PRIORITARIA] Rezago Crítico / Conectividad — {nombre} (Chía)"

    cfg = estado_configuracion_brevo()
    if not cfg["configured"]:
        faltantes = ", ".join(cfg["missing_vars"])
        enviar_correo_brevo(destinatario, asunto, html_out, tipo="Alerta Crítica")
        return jsonify({
            "status": "sin_credenciales",
            "configured": False,
            "mensaje": f"Faltan variables de entorno ({faltantes}) para el envío real. Se generó la vista previa de la alerta.",
            "missing_vars": cfg["missing_vars"],
            "remaining": restantes,
            "preview_html": html_out,
            "server_ms": round((time.perf_counter() - t0) * 1000, 2),
        }), 503

    thread = threading.Thread(
        target=enviar_correo_brevo,
        args=(destinatario, asunto, html_out, "Alerta Crítica"),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "en_cola",
        "configured": True,
        "mensaje": f"Alerta prioritaria en cola de envío (HTTP 202 Accepted) hacia {destinatario}.",
        "remaining": restantes,
        "preview_html": html_out,
        "server_ms": round((time.perf_counter() - t0) * 1000, 2),
    }), 202


if __name__ == "__main__":
    app.run(debug=True)