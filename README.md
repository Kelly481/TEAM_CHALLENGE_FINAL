# 🏦 API REST — Predicción de Churn Bancario

**Proyecto Final ML · The Bridge Data Science Bootcamp**  
Autores: Kelly Escalante · Jorge Martinez · Rebeca Prior

---

## ¿Qué hace esta API?

Predice si un cliente bancario abandonará el banco (`Exited=1`) o permanecerá (`Exited=0`) usando un modelo **XGBoost** optimizado con RandomizedSearchCV (ROC-AUC > 0.86).

---

## Estructura del repositorio

```
churn_api/
├── app.py                  ← API Flask (este archivo)
├── requirements.txt        ← Dependencias
├── README.md               ← Esta guía
└── src/
    ├── data_sample/
    │   └── churn.csv       ← Dataset original
    └── models/
        ├── modelo_churn_bancario.joblib
        ├── scaler.joblib
        ├── capping_bounds.joblib
        └── feature_names.joblib
```

---

## Endpoints

### `GET /`
Landing page — devuelve información completa de la API y ejemplos de uso.

### `GET /predict` — Predicción via query params
```
/predict?CreditScore=600&Geography=France&Gender=Female&Age=40&Tenure=3&Balance=60000&NumOfProducts=2&HasCrCard=1&IsActiveMember=1&EstimatedSalary=50000
```

### `POST /predict` — Predicción via JSON
```json
{
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
}
```

**Respuesta:**
```json
{
  "prediccion": 0,
  "etiqueta": "PERMANECE en el banco",
  "probabilidad_churn": 0.1823,
  "nivel_riesgo": "BAJO",
  "datos_entrada": { ... }
}
```

### `GET /health`
Devuelve el estado del servidor y confirma que el modelo está cargado.

---

## Cómo usar con Python `requests`

```python
import requests

# GET
url = "https://TU-APP.onrender.com/predict"
params = {
    "CreditScore": 600, "Geography": "France", "Gender": "Female",
    "Age": 40, "Tenure": 3, "Balance": 60000, "NumOfProducts": 2,
    "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000
}
response = requests.get(url, params=params)
print(response.json())

# POST
response = requests.post(url, json=params)
print(response.json())
```

---

## Despliegue en Render — Paso a paso

### 1. Crear el repositorio en GitHub
```bash
git init
git add .
git commit -m "Initial commit — Churn API"
git remote add origin https://github.com/TU_USUARIO/churn-api.git
git push -u origin main
```

### 2. Crear una cuenta en Render
Ve a [render.com](https://render.com) y regístrate con tu cuenta de GitHub.

### 3. Crear nuevo Web Service
- Clic en **New → Web Service**
- Conecta tu repositorio de GitHub
- Configura:
  - **Name:** churn-api (o el nombre que queráis)
  - **Runtime:** Python 3
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `gunicorn app:app`
  - **Instance Type:** Free

### 4. Deploy
- Clic en **Create Web Service**
- Espera 2-3 minutos a que termine el build
- Tu API estará disponible en: `https://churn-api.onrender.com`

---

## Redespliegue (para la exposición)
Para descomentar el endpoint `/stats` y redesplegar:
1. Descomenta el bloque `@app.route("/stats")` en `app.py`
2. `git add app.py && git commit -m "Add /stats endpoint" && git push`
3. Render detecta el push y redespliega automáticamente en ~2 min

---

## Campos requeridos

| Campo | Tipo | Valores |
|---|---|---|
| CreditScore | int | 300–850 |
| Geography | str | France, Germany, Spain |
| Gender | str | Male, Female |
| Age | int | 18–92 |
| Tenure | int | 0–10 |
| Balance | float | 0.0 – cualquier valor |
| NumOfProducts | int | 1–4 |
| HasCrCard | int | 0 o 1 |
| IsActiveMember | int | 0 o 1 |
| EstimatedSalary | float | cualquier valor |
