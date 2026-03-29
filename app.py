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

modelo        = joblib.load(os.path.join(MODELS_DIR, "modelo_churn_bancario.joblib"))
scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
capping_bounds = joblib.load(os.path.join(MODELS_DIR, "capping_bounds.joblib"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))


def preprocesar(data: dict) -> pd.DataFrame:
    """
    Aplica el mismo pipeline de preprocesado que en el notebook:
    1. Capping P1-P99 (con bounds aprendidos en train)
    2. Feature Engineering
    3. One-Hot Encoding
    4. Alineación de columnas
    5. StandardScaler (con parámetros aprendidos en train)
    """
    df = pd.DataFrame([data])

    # 1 ── Capping P1-P99
    cols_capping = ["Age", "Balance", "CreditScore", "EstimatedSalary"]
    for col in cols_capping:
        if col in df.columns and col in capping_bounds:
            p1, p99 = capping_bounds[col]
            df[col] = df[col].clip(lower=p1, upper=p99)

    # 2 ── Feature Engineering
    df["balance_per_product"] = df["Balance"] / (df["NumOfProducts"] + 1)
    df["HasBalance"]          = (df["Balance"] > 0).astype(int)
    df["EngagedCustomer"]     = (
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] > 1)
    ).astype(int)
    df["SalaryAgeRatio"]      = df["EstimatedSalary"] / (df["Age"] + 1)

    # 3 ── One-Hot Encoding
    cat_cols = ["Geography", "Gender"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4 ── Alinear columnas con las del modelo
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # 5 ── Escalar
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# ── ENDPOINT 1: Landing Page ──────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def landing():
    info = {
        "nombre": "API de Predicción de Churn Bancario",
        "descripcion": "Predice si un cliente bancario abandonará el banco (Exited=1) o permanecerá (Exited=0).",
        "modelo": "XGBoost optimizado con RandomizedSearchCV",
        "metrica_principal": "ROC-AUC > 0.86",
        "autores": ["Kelly Escalante", "Jorge Martinez", "Rebeca Prior"],
        "endpoints": {
            "GET /": "Esta página — información y guía de uso",
            "GET /predict": "Predicción de churn con parámetros en la URL (query params)",
            "POST /predict": "Predicción de churn con JSON en el body",
            "GET /health": "Estado del servidor y modelo cargado",
            # "GET /retrain": "EXTRA: Reentrenamiento del modelo (comentado)"
        },
        "ejemplo_get": (
            "/predict?CreditScore=600&Geography=France&Gender=Female"
            "&Age=40&Tenure=3&Balance=60000&NumOfProducts=2"
            "&HasCrCard=1&IsActiveMember=1&EstimatedSalary=50000"
        ),
        "ejemplo_post_body": {
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Female",
            "Age": 40,
            "Tenure": 3,
            "Balance": 60000,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000
        },
        "campos_requeridos": [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ],
        "valores_Geography": ["France", "Germany", "Spain"],
        "valores_Gender": ["Male", "Female"]
    }
    return jsonify(info), 200


# ── ENDPOINT 2: Predicción (GET y POST) ───────────────────────────────────────
CAMPOS_REQUERIDOS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

def _hacer_prediccion(data: dict):
    """Lógica compartida entre GET y POST."""
    # Validar campos
    faltantes = [c for c in CAMPOS_REQUERIDOS if c not in data]
    if faltantes:
        return jsonify({"error": f"Faltan campos requeridos: {faltantes}"}), 400

    # Convertir tipos numéricos
    try:
        numericos = ["CreditScore", "Age", "Tenure", "Balance",
                     "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        for col in numericos:
            data[col] = float(data[col])
    except ValueError as e:
        return jsonify({"error": f"Error de tipo en los datos: {str(e)}"}), 400

    # Preprocesar y predecir
    try:
        df_procesado = preprocesar(data)
        prob_churn   = float(modelo.predict_proba(df_procesado)[0][1])
        prediccion   = int(modelo.predict(df_procesado)[0])
        nivel_riesgo = (
            "ALTO"   if prob_churn > 0.7 else
            "MEDIO"  if prob_churn > 0.4 else
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
    """Predicción vía query params: /predict?CreditScore=600&Age=40&..."""
    data = request.args.to_dict()
    return _hacer_prediccion(data)


@app.route("/predict", methods=["POST"])
def predict_post():
    """Predicción vía JSON body."""
    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 415
    data = request.get_json()
    return _hacer_prediccion(data)


# ── ENDPOINT 3: Health check ──────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "OK",
        "modelo_cargado":  True,
        "features":        len(feature_names),
        "version":         "1.0.0"
    }), 200


# ── ENDPOINT EXTRA (comentado — descomentar para redespliegue en clase) ───────
# @app.route("/stats", methods=["GET"])
# def stats():
#     """Devuelve estadísticas del modelo: features más importantes."""
#     importancias = dict(zip(feature_names, modelo.feature_importances_.tolist()))
#     top10 = dict(sorted(importancias.items(), key=lambda x: x[1], reverse=True)[:10])
#     return jsonify({
#         "top10_features_importantes": top10,
#         "total_features": len(feature_names)
#     }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
