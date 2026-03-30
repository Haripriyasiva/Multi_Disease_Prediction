import pytest
from fastapi.testclient import TestClient
from server import app 

client = TestClient(app)

def test_invalid_input():
    response = client.post("/api/v1/clinical-visits", json={
        "patient_id": "REQ-9942",
        "visit_date": "2026-03-22T08:30:00Z",
        "vitals": {
            "fasting_glucose": 600,
            "post_prandial_glucose": None,
            "systolic_bp": 128,
            "diastolic_bp": 82,
            "bmi": 12
        }
    })
    assert response.status_code == 201
    assert response.json()["risk_score"] == 0.999

def test_missing_data():
    response = client.post("/api/v1/clinical-visits", json={
        "patient_id": "REQ-9942",
        "visit_date": "2026-03-22T08:30:00Z",
        "vitals": {
            "fasting_glucose": 115,
            "post_prandial_glucose": None,
            "systolic_bp": 128,
            "diastolic_bp": 82,
            "bmi": 26.5
        }
    })
    assert response.status_code == 400
