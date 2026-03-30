# 🏦 API REST — Predicción de Churn Bancario

**Autores:** Kelly Escalante · Jorge Martinez · Rebeca Prior  
**Bootcamp:** Data Science & AI — The Bridge · Abril 2026  
**Modelo:** XGBoost optimizado | **ROC-AUC:** ~0.87  
**URL pública:** https://team-challenge-final.onrender.com

---

## 📡 Endpoints

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Documentación de la API |
| `/predict` | GET | Predicción con parámetros en la URL |
| `/predict` | POST | Predicción con body JSON |
| `/health` | GET | Estado del servidor |
| `/stats` | GET | Top 5 features *(comentado — demo en clase)* |

---

## 🚀 Cómo usar la API

### Opción 1 — Desde el navegador
```
https://team-challenge-final.onrender.com/predict?CreditScore=500&Geography=Germany&Gender=Female&Age=55&Tenure=2&Balance=120000&NumOfProducts=1&HasCrCard=1&IsActiveMember=0&EstimatedSalary=80000
```

### Opción 2 — Desde Python
```python
import requests

BASE_URL = "https://team-challenge-final.onrender.com"

# GET
params = {
    "CreditScore": 500, "Geography": "Germany", "Gender": "Female",
    "Age": 55, "Tenure": 2, "Balance": 120000, "NumOfProducts": 1,
    "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 80000
}
r = requests.get(f"{BASE_URL}/predict", params=params)
print(r.json())

# POST
payload = {
    "CreditScore": 750, "Geography": "France", "Gender": "Male",
    "Age": 32, "Tenure": 7, "Balance": 0, "NumOfProducts": 2,
    "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000
}
r = requests.post(f"{BASE_URL}/predict", json=payload)
print(r.json())
```

---

## 📋 Parámetros requeridos

| Parámetro | Tipo | Valores válidos |
|---|---|---|
| `CreditScore` | int | 300 - 850 |
| `Geography` | str | France, Germany, Spain |
| `Gender` | str | Male, Female |
| `Age` | int | 18 - 92 |
| `Tenure` | int | 0 - 10 |
| `Balance` | float | 0.0 - 250000.0 |
| `NumOfProducts` | int | 1 - 4 |
| `HasCrCard` | int | 0 o 1 |
| `IsActiveMember` | int | 0 o 1 |
| `EstimatedSalary` | float | 0.0 - 200000.0 |

---

## 📊 Ejemplo de respuesta

```json
{
  "prediccion": 1,
  "etiqueta": "ABANDONA el banco",
  "probabilidad_churn": 0.9637,
  "nivel_riesgo": "ALTO",
  "datos_entrada": {}
}
```

### Niveles de riesgo

| Nivel | Probabilidad | Acción recomendada |
|---|---|---|
| 🟢 BAJO | < 40% | Seguimiento rutinario |
| 🟡 MEDIO | 40% - 70% | Monitorizar y mejorar condiciones |
| 🔴 ALTO | > 70% | Activar campaña de retención urgente |

---

## 📁 Estructura del repositorio

```
TEAM_CHALLENGE_FINAL/
├── app.py                  ← API principal (Flask + Gunicorn)
├── requirements.txt        ← Dependencias
├── runtime.txt             ← Versión Python (3.11.9)
├── test_api.py             ← Script de pruebas
├── README.md               ← Esta documentación
└── src/
    └── models/
        ├── modelo_churn_bancario.joblib
        ├── scaler.joblib
        ├── capping_bounds.joblib
        └── feature_names.joblib
```

---

## ⚠️ Nota sobre el plan gratuito

El servidor gratuito de Render se duerme tras 15 minutos de inactividad.  
El primer request puede tardar hasta 50 segundos en responder mientras arranca.
