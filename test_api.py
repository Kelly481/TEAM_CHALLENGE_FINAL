"""
Script de prueba de la API — Churn Bancario
Uso local:   python test_api.py
Uso Render:  python test_api.py https://tu-app.onrender.com
"""

import requests
import json
import sys

# URL base — cambia esto por la URL de Render cuando esté desplegada
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

print(f"\n{'='*55}")
print(f"  Pruebas de la API — Churn Bancario")
print(f"  URL: {BASE_URL}")
print(f"{'='*55}\n")


# ── TEST 1: Landing page ─────────────────────────────────────────────────────
print("▶ TEST 1: GET / — Landing page")
r = requests.get(f"{BASE_URL}/")
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Nombre: {data['nombre']}")
print(f"  Endpoints disponibles: {list(data['endpoints'].keys())}")
print()


# ── TEST 2: Health check ─────────────────────────────────────────────────────
print("▶ TEST 2: GET /health — Estado del servicio")
r = requests.get(f"{BASE_URL}/health")
print(f"  Status: {r.status_code}")
print(f"  Respuesta: {r.json()}")
print()


# ── TEST 3: Predicción GET — cliente con riesgo ALTO ────────────────────────
print("▶ TEST 3: GET /predict — Cliente riesgo ALTO")
params = {
    "CreditScore": 500, "Geography": "Germany", "Gender": "Female",
    "Age": 55, "Tenure": 2, "Balance": 120000, "NumOfProducts": 1,
    "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 80000
}
r = requests.get(f"{BASE_URL}/predict", params=params)
print(f"  Status: {r.status_code}")
res = r.json()
print(f"  Predicción:  {res['etiqueta']}")
print(f"  Prob. churn: {res['probabilidad_churn']}")
print(f"  Riesgo:      {res['nivel_riesgo']}")
print(f"  Acción:      {res['recomendacion']}")
print()


# ── TEST 4: Predicción POST — cliente con riesgo BAJO ───────────────────────
print("▶ TEST 4: POST /predict — Cliente riesgo BAJO")
payload = {
    "CreditScore": 750, "Geography": "France", "Gender": "Male",
    "Age": 32, "Tenure": 7, "Balance": 0, "NumOfProducts": 2,
    "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000
}
r = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"  Status: {r.status_code}")
res = r.json()
print(f"  Predicción:  {res['etiqueta']}")
print(f"  Prob. churn: {res['probabilidad_churn']}")
print(f"  Riesgo:      {res['nivel_riesgo']}")
print()


# ── TEST 5: Error — campos faltantes ────────────────────────────────────────
print("▶ TEST 5: GET /predict — Error: campos faltantes")
r = requests.get(f"{BASE_URL}/predict", params={"CreditScore": 700})
print(f"  Status: {r.status_code}")
print(f"  Error: {r.json()['error']}")
print(f"  Faltan: {r.json()['campos_faltantes']}")
print()

print("✅ Todas las pruebas completadas")