
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_prediction_endpoint():
    payload = {
        "type": "L",
        "air_temperature_k": 303.0,
        "process_temperature_k": 313.0,
        "rotational_speed_rpm": 1400,
        "torque_nm": 60.0,
        "tool_wear_min": 250
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Normal", "Failure"]
    assert 0 <= result["failure_probability_pct"] <= 100


def test_invalid_input():
    payload = {
        "type": "L",
        "air_temperature_k": 303.0,
        "process_temperature_k": 313.0,
        "rotational_speed_rpm": 1400,
        "torque_nm": -10.0,
        "tool_wear_min": 250
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
