import requests
import json

BASE_URL = "http://localhost:5000"

def get_token():
    """Logs in and returns a JWT token."""
    login_data = {
        "username": "doctor1",
        "password": "secure123"
    }
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    response.raise_for_status()
    return response.json()['token']

def test_predict(token):
    """Tests the /predict endpoint for Heart and Kidney diseases."""
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Test Heart disease
    heart_vitals = {
        "thal": 3,
        "cp": 3,
        "ca": 0,
        "exang": 0,
        "thalach": 150,
        "chol": 233,
        "slope": 0,
        "oldpeak": 2.3
    }
    heart_data = {
        "type": "Heart",
        "vitals": heart_vitals
    }
    print("Testing Heart model...")
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=heart_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 20)

    # Test Kidney disease
    kidney_vitals = {
        "id": 0,
        "hemo": 15.4,
        "pcv": 44,
        "rc": 5.2,
        "al": 1,
        "htn": 1,
        "bu": 36,
        "bgr": 121
    }
    kidney_data = {
        "type": "Kidney",
        "vitals": kidney_vitals
    }
    print("Testing Kidney model...")
    response = requests.post(f"{BASE_URL}/predict", headers=headers, json=kidney_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 20)

if __name__ == "__main__":
    try:
        jwt_token = get_token()
        test_predict(jwt_token)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
