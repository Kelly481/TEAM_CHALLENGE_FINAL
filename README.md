# 🏦 API de Predicción de Churn Bancario

API REST desarrollada en Python para predecir la probabilidad de abandono (churn) de clientes bancarios utilizando un modelo de Machine Learning basado en XGBoost optimizado.

---

## 🚀 Demo en producción

Puedes interactuar directamente con la API desplegada:

🔹 **API (respuesta en JSON)** 👉 https://team-challenge-final.onrender.com/  
🔹 **Documentación visual interactiva** 👉 https://team-challenge-final.onrender.com/docs  
🔹 **Health check del servicio** 👉 https://team-challenge-final.onrender.com/health  

> ⚠️ El plan gratuito de Render se duerme tras 15 min de inactividad. El primer request puede tardar hasta 50 segundos en responder.

---

## ⚡ Prueba rápida

Haz clic en el siguiente enlace para ejecutar una predicción real:

```
https://team-challenge-final.onrender.com/predict?CreditScore=500&Geography=Germany&Gender=Female&Age=55&Tenure=2&Balance=120000&NumOfProducts=1&HasCrCard=1&IsActiveMember=0&EstimatedSalary=80000
```

---

## 🎯 Objetivo del proyecto

Predecir si un cliente bancario abandonará el banco (`Exited = 1`) o permanecerá (`Exited = 0`), permitiendo:

- Identificar clientes en riesgo
- Optimizar estrategias de retención
- Tomar decisiones basadas en datos

---

## 🧠 Modelo de Machine Learning

- **Algoritmo:** XGBoost Classifier
- **Optimización:** RandomizedSearchCV
- **Métrica principal:** ROC-AUC ≈ 0.86 - 0.87
- **Dataset:** 10.000 clientes bancarios europeos

---

## ⚙️ Pipeline de datos

El modelo en producción replica exactamente el pipeline de entrenamiento:

1. **Capping (P1 - P99)** para control de outliers
2. **Feature Engineering:**
   - `balance_per_product`
   - `HasBalance`
   - `EngagedCustomer`
   - `SalaryAgeRatio`
3. **One-Hot Encoding manual**
4. **Alineación de features**
5. **Escalado con StandardScaler**

---

## 📡 Endpoints disponibles

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Información general de la API (JSON) |
| `/docs` | GET | Documentación visual interactiva |
| `/predict` | GET | Predicción mediante query params |
| `/predict` | POST | Predicción mediante JSON |
| `/health` | GET | Estado del servicio |
| `/stats` | GET | Top 10 features importantes *(comentado — demo en clase)* |

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

## 📥 Ejemplo de request (POST)

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

## 📤 Ejemplo de respuesta

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

## 🧱 Tecnologías utilizadas

- Python
- Flask + Gunicorn
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Joblib
- Render (deploy)

---

## 🏗️ Arquitectura

- API REST para inferencia en tiempo real
- Modelo serializado (`.joblib`)
- Pipeline de preprocesado integrado
- Despliegue en cloud (Render)

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

## 👥 Autores

- Kelly Escalante
- Jorge Martinez
- Rebeca Prior

**Bootcamp Data Science & AI — The Bridge · Abril 2026**

---

## 📌 Estado del proyecto

✅ Modelo entrenado y optimizado  
✅ API desplegada en producción  
✅ Documentación interactiva  
✅ Pipeline reproducible  

---

