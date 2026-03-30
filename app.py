"""
API REST — Predicción de Churn Bancario
Modelo: XGBoost optimizado
Autores: Kelly Escalante · Jorge Martinez · Rebeca Prior
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Cargar artefactos del modelo ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")

modelo         = joblib.load(os.path.join(MODELS_DIR, "modelo_churn_bancario.joblib"))
scaler         = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
capping_bounds = joblib.load(os.path.join(MODELS_DIR, "capping_bounds.joblib"))
feature_names  = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))


def preprocesar(data: dict) -> pd.DataFrame:
    """
    Aplica el mismo pipeline de preprocesado que en el notebook:
    1. Capping P1-P99 (con bounds aprendidos en train)
    2. Feature Engineering
    3. One-Hot Encoding manual
    4. Alineación de columnas
    5. StandardScaler (con parámetros aprendidos en train)
    """
    df = pd.DataFrame([data])

    # 1. Capping P1-P99
    for col, (p1, p99) in capping_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=p1, upper=p99)

    # 2. Feature engineering
    df["balance_per_product"] = df["Balance"] / (df["NumOfProducts"] + 1)
    df["HasBalance"]          = (df["Balance"] > 0).astype(int)
    df["EngagedCustomer"]     = (
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] > 1)
    ).astype(int)
    df["SalaryAgeRatio"] = df["EstimatedSalary"] / (df["Age"] + 1)

    # 3. One-Hot Encoding manual
    df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Geography_Spain"]   = (df["Geography"] == "Spain").astype(int)
    df["Gender_Male"]       = (df["Gender"] == "Male").astype(int)
    df = df.drop(columns=["Geography", "Gender"])

    # 4. Alinear columnas con las del modelo
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # 5. Escalar
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# ── ENDPOINT 1: Landing Page (JSON) ──────────────────────────────────────────
@app.route("/", methods=["GET"])
def landing():
    info = {
        "nombre": "API de Predicción de Churn Bancario",
        "descripcion": "Predice si un cliente bancario abandonará el banco (Exited=1) o permanecerá (Exited=0).",
        "modelo": "XGBoost optimizado con RandomizedSearchCV",
        "metrica_principal": "ROC-AUC > 0.86",
        "autores": ["Kelly Escalante", "Jorge Martinez", "Rebeca Prior"],
        "endpoints": {
            "GET /":      "Esta página — documentación en JSON",
            "GET /docs":  "Documentación visual en HTML — para humanos",
            "GET /predict":  "Predicción de churn con parámetros en la URL (query params)",
            "POST /predict": "Predicción de churn con JSON en el body",
            "GET /health":   "Estado del servidor y modelo cargado",
        },
        "ejemplo_get": (
            "/predict?CreditScore=600&Geography=France&Gender=Female"
            "&Age=40&Tenure=3&Balance=60000&NumOfProducts=2"
            "&HasCrCard=1&IsActiveMember=1&EstimatedSalary=50000"
        ),
        "ejemplo_post_body": {
            "CreditScore": 600, "Geography": "France", "Gender": "Female",
            "Age": 40, "Tenure": 3, "Balance": 60000, "NumOfProducts": 2,
            "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000
        },
        "campos_requeridos": [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ],
        "valores_Geography": ["France", "Germany", "Spain"],
        "valores_Gender": ["Male", "Female"]
    }
    return jsonify(info), 200


# ── ENDPOINT 2: Documentación visual (HTML) ───────────────────────────────────
@app.route("/docs", methods=["GET"])
def docs():
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Churn Bancario</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f4f8; color: #333; }
        header { background: linear-gradient(135deg, #1F4E79, #2E75B6); color: white; padding: 40px; text-align: center; }
        header h1 { font-size: 2.2em; margin-bottom: 8px; }
        header p  { font-size: 1.1em; opacity: 0.9; }
        .badge { display: inline-block; background: rgba(255,255,255,0.2); border-radius: 20px; padding: 4px 14px; margin: 4px; font-size: 0.9em; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .card { background: white; border-radius: 12px; padding: 30px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { color: #1F4E79; font-size: 1.3em; margin-bottom: 16px; border-bottom: 2px solid #e8f0fe; padding-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #2E75B6; color: white; padding: 10px 14px; text-align: left; }
        td { padding: 10px 14px; border-bottom: 1px solid #eee; }
        tr:last-child td { border-bottom: none; }
        tr:hover td { background: #f8f9ff; }
        .method { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
        .get  { background: #d4edda; color: #155724; }
        .post { background: #cce5ff; color: #004085; }
        pre { background: #1e1e1e; color: #f8f8f2; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 0.9em; line-height: 1.6; }
        .highlight { color: #66d9ef; }
        .string { color: #a6e22e; }
        .number { color: #ae81ff; }
        .risk { display: inline-block; padding: 3px 10px; border-radius: 4px; font-weight: bold; }
        .alto  { background: #f8d7da; color: #721c24; }
        .medio { background: #fff3cd; color: #856404; }
        .bajo  { background: #d4edda; color: #155724; }
        footer { text-align: center; padding: 30px; color: #888; font-size: 0.9em; }
        .url-box { background: #f8f9ff; border: 1px solid #2E75B6; border-radius: 8px; padding: 12px 16px; font-family: monospace; font-size: 0.85em; color: #1F4E79; word-break: break-all; margin-top: 10px; }
    </style>
</head>
<body>

<header>
    <h1>🏦 API de Predicción de Churn Bancario</h1>
    <p>Predice si un cliente bancario abandonará el banco</p>
    <br>
    <span class="badge">XGBoost optimizado</span>
    <span class="badge">ROC-AUC ~0.87</span>
    <span class="badge">Flask + Gunicorn</span>
    <span class="badge">Render</span>
    <br><br>
    <span class="badge">👥 Kelly Escalante · Jorge Martinez · Rebeca Prior</span>
</header>

<div class="container">

    <div class="card">
        <h2>📡 Endpoints disponibles</h2>
        <table>
            <tr><th>Endpoint</th><th>Método</th><th>Descripción</th></tr>
            <tr><td><code>/</code></td><td><span class="method get">GET</span></td><td>Documentación en JSON</td></tr>
            <tr><td><code>/docs</code></td><td><span class="method get">GET</span></td><td>Esta página — documentación visual</td></tr>
            <tr><td><code>/predict</code></td><td><span class="method get">GET</span></td><td>Predicción con parámetros en la URL</td></tr>
            <tr><td><code>/predict</code></td><td><span class="method post">POST</span></td><td>Predicción con body JSON</td></tr>
            <tr><td><code>/health</code></td><td><span class="method get">GET</span></td><td>Estado del servidor</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>📋 Parámetros requeridos</h2>
        <table>
            <tr><th>Parámetro</th><th>Tipo</th><th>Valores válidos</th></tr>
            <tr><td><code>CreditScore</code></td><td>int</td><td>300 - 850</td></tr>
            <tr><td><code>Geography</code></td><td>str</td><td>France, Germany, Spain</td></tr>
            <tr><td><code>Gender</code></td><td>str</td><td>Male, Female</td></tr>
            <tr><td><code>Age</code></td><td>int</td><td>18 - 92</td></tr>
            <tr><td><code>Tenure</code></td><td>int</td><td>0 - 10</td></tr>
            <tr><td><code>Balance</code></td><td>float</td><td>0.0 - 250000.0</td></tr>
            <tr><td><code>NumOfProducts</code></td><td>int</td><td>1 - 4</td></tr>
            <tr><td><code>HasCrCard</code></td><td>int</td><td>0 o 1</td></tr>
            <tr><td><code>IsActiveMember</code></td><td>int</td><td>0 o 1</td></tr>
            <tr><td><code>EstimatedSalary</code></td><td>float</td><td>0.0 - 200000.0</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>🚀 Ejemplo de uso desde Python</h2>
        <pre><span class="highlight">import</span> requests

BASE_URL = <span class="string">"https://team-challenge-final.onrender.com"</span>

params = {
    <span class="string">"CreditScore"</span>:     <span class="number">500</span>,
    <span class="string">"Geography"</span>:       <span class="string">"Germany"</span>,
    <span class="string">"Gender"</span>:          <span class="string">"Female"</span>,
    <span class="string">"Age"</span>:             <span class="number">55</span>,
    <span class="string">"Tenure"</span>:          <span class="number">2</span>,
    <span class="string">"Balance"</span>:         <span class="number">120000</span>,
    <span class="string">"NumOfProducts"</span>:   <span class="number">1</span>,
    <span class="string">"HasCrCard"</span>:       <span class="number">1</span>,
    <span class="string">"IsActiveMember"</span>:  <span class="number">0</span>,
    <span class="string">"EstimatedSalary"</span>: <span class="number">80000</span>
}
r = requests.get(<span class="string">f"{BASE_URL}/predict"</span>, params=params)
<span class="highlight">print</span>(r.json())</pre>
    </div>

    <div class="card">
        <h2>📊 Ejemplo de respuesta</h2>
        <pre>{
    <span class="string">"prediccion"</span>:         <span class="number">1</span>,
    <span class="string">"etiqueta"</span>:           <span class="string">"ABANDONA el banco"</span>,
    <span class="string">"probabilidad_churn"</span>: <span class="number">0.9637</span>,
    <span class="string">"nivel_riesgo"</span>:       <span class="string">"ALTO"</span>
}</pre>
        <br>
        <table>
            <tr><th>Nivel de riesgo</th><th>Probabilidad</th><th>Acción recomendada</th></tr>
            <tr><td><span class="risk bajo">🟢 BAJO</span></td><td>&lt; 40%</td><td>Seguimiento rutinario</td></tr>
            <tr><td><span class="risk medio">🟡 MEDIO</span></td><td>40% - 70%</td><td>Monitorizar y mejorar condiciones</td></tr>
            <tr><td><span class="risk alto">🔴 ALTO</span></td><td>&gt; 70%</td><td>Activar campaña de retención urgente</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>🔗 Prueba rápida desde el navegador</h2>
        <p>Copia esta URL y pégala en el navegador:</p>
        <div class="url-box">https://team-challenge-final.onrender.com/predict?CreditScore=500&amp;Geography=Germany&amp;Gender=Female&amp;Age=55&amp;Tenure=2&amp;Balance=120000&amp;NumOfProducts=1&amp;HasCrCard=1&amp;IsActiveMember=0&amp;EstimatedSalary=80000</div>
    </div>

</div>

<footer>
    🏦 API Churn Bancario · Kelly Escalante · Jorge Martinez · Rebeca Prior · The Bridge 2026
</footer>

</body>
</html>
"""
    return html


# ── ENDPOINT 3: Predicción (GET y POST) ───────────────────────────────────────
CAMPOS_REQUERIDOS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

def _hacer_prediccion(data: dict):
    """Lógica compartida entre GET y POST."""
    faltantes = [c for c in CAMPOS_REQUERIDOS if c not in data]
    if faltantes:
        return jsonify({"error": f"Faltan campos requeridos: {faltantes}"}), 400

    try:
        numericos = ["CreditScore", "Age", "Tenure", "Balance",
                     "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        for col in numericos:
            data[col] = float(data[col])
    except ValueError as e:
        return jsonify({"error": f"Error de tipo en los datos: {str(e)}"}), 400

    try:
        df_procesado = preprocesar(data)
        prob_churn   = float(modelo.predict_proba(df_procesado)[0][1])
        prediccion   = int(modelo.predict(df_procesado)[0])
        nivel_riesgo = (
            "ALTO"  if prob_churn > 0.7 else
            "MEDIO" if prob_churn > 0.4 else
            "BAJO"
        )
        return jsonify({
            "prediccion":         prediccion,
            "etiqueta":           "ABANDONA el banco" if prediccion == 1 else "PERMANECE en el banco",
            "probabilidad_churn": round(prob_churn, 4),
            "nivel_riesgo":       nivel_riesgo,
            "datos_entrada":      data
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500


@app.route("/predict", methods=["GET"])
def predict_get():
    data = request.args.to_dict()
    return _hacer_prediccion(data)


@app.route("/predict", methods=["POST"])
def predict_post():
    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 415
    data = request.get_json()
    return _hacer_prediccion(data)


# ── ENDPOINT 4: Health check ──────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "OK",
        "modelo_cargado": True,
        "features":       len(feature_names),
        "version":        "1.0.0"
    }), 200


# ── ENDPOINT EXTRA (descomentar para redespliegue en clase) ───────────────────
# @app.route("/stats", methods=["GET"])
# def stats():
#     """Devuelve las 10 features más importantes del modelo."""
#     importancias = dict(zip(feature_names, modelo.feature_importances_.tolist()))
#     top10 = dict(sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:10])
#     return jsonify({
#         "top10_features_importantes": top10,
#         "total_features": len(feature_names)
#     }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
