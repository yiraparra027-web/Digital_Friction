# Digital Friction Chía

Observatorio de madurez digital empresarial de Chía y Sabana Centro (Flask + scikit-learn, metodología CRISP-ML(Q)).

## Ejecución local
```bash
pip install -r requirements.txt
# Coloca en la raíz el dataset: sabana_centro_digitalizacion_empresarial.csv
python app.py
```
Producción (Render): `gunicorn app:app`.

## Actualización integral (resumen)
1. **Colorimetría global**: `static/css/style.css` (variables `:root`, sobrescritura de variables Bootstrap) y gráficas matplotlib/Chart.js.
2. **Renombramiento**: menús, encabezados y tarjetas (Evaluación empresarial, Mapa de madurez, Perfiles de madurez, Alertas de rezago crítico, Precisión del diagnóstico, Solidez de indicadores, Precisión de perfiles, Fiabilidad de alertas).
3. **/business-understanding**: 4 tarjetas de cuellos de botella de red y tabla "Del síntoma empresarial a la causa en red".
4. **/pca-prediction**: parámetros de Infraestructura y Transmisión de Datos. El modelo PCA/K-Means se reentrena desde el CSV, que no trae variables de red; por eso se integran como **Índice de Madurez de Red y Conectividad** (`network_maturity.py`), que ajusta de forma acotada (±8 pts) la madurez mostrada sin alterar las 8 dimensiones del modelo.
5. **/kmeans y resultados del simulador**: bloque "Arquitectura y Soluciones de Comunicación de Datos" por clúster.
6. **/data-engineering**: tarjeta "Arquitectura de Red y Flujo de Comunicación del Sistema". **/pca-prediction**: telemetría RTT con `performance.now()` (más `server_ms` de inferencia).

## Archivos nuevos y modificados
- Nuevo: `network_maturity.py`
- Modificados: `app.py`, `pca_predictor.py`, `kmeans_model.py`, `PCA_Model.py`, `dbscan_model.py`, `static/css/style.css` y las plantillas de `templates/`.
