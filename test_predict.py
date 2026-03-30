import requests

def run_test():
    url = "http://127.0.0.1:5000/demo_login"
    try:
        login_resp = requests.post(url, json={"role": "DOCTOR"}, timeout=10)
        print("Login:", login_resp.json())
        token = login_resp.json()["token"]
    except Exception as e:
        print("Login failed:", e)
        return

    pred_url = "http://127.0.0.1:5000/predict"
    body = {
        "type": "Diabetes",
        "vitals": {
            "fasting_glucose": "140",
            "post_prandial_glucose": "215",
            "BMI": "29.5",
            "BloodPressure": "90",
            "Age": "58",
            "DiabetesPedigreeFunction": "0.52",
            "SkinThickness": "28",
            "Insulin": "120",
            "Pregnancies": "2"
        },
        "patientName": "patient1"
    }

    try:
        r = requests.post(pred_url, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        print("Predict Code:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print("Prob:", data.get("prob"))
            print("Risk Level:", data.get("disease"))
            print("SHAP DATA structure:", data.get("shap_data"))
            print("CHART Image:", "Yes" if data.get("chart") else "No")
        else:
            print("Error Response:", r.text)
    except Exception as e:
        print("Predict failed:", e)

if __name__ == "__main__":
    run_test()
