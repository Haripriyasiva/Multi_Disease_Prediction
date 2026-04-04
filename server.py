import os
import io
import base64
import datetime
import pandas as pd
import numpy as np
import jwt
import json

import faiss
import pickle
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from sentence_transformers import SentenceTransformer

from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from celery import Celery
from celery_worker import schedule_post_prandial_notification

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from pydantic import BaseModel, Field, ValidationError
from typing import Optional

from database import init_db, get_db_session, Patient_Profiles, Patient_Visits, Audit_Log, Notification

app = Flask(__name__)
CORS(app)

class Vitals(BaseModel):
    fasting_glucose: int
    post_prandial_glucose: Optional[int] = None
    systolic_bp: int
    diastolic_bp: int
    bmi: float

class ClinicalVisit(BaseModel):
    patient_id: str = Field(..., alias='patient_id')
    visit_date: str = Field(..., alias='visit_date')
    vitals: Vitals

JWT_SECRET = "med_ai_premium_portal_2026_secure_key_v3"

app.config['CELERY_BROKER_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

def send_push_notification(patient_username, message):
    print(f"PUSH NOTIFICATION for {patient_username}: {message}")
    db = get_db_session()
    try:
        patient = db.query(Patient_Profiles).filter(Patient_Profiles.username == patient_username).first()
        if patient:
            db.add(Notification(patient_id=patient.id, message=message))
            db.commit()
    except Exception as e:
        print(f"Error saving notification to inbox: {e}")
    finally:
        db.close()

rag_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index("medical_knowledge.index")
with open("medical_text_mapping.pkl", "rb") as f:
    medical_faqs = pickle.load(f)

def retrieve_medical_advice(user_query):
    query_vector = rag_model.encode([user_query]).astype("float32")
    distances, indices = faiss_index.search(query_vector, 3)
    results = [medical_faqs[i] for i in indices[0]]
    guardrail = " I cannot prescribe medication or dosages. Always consult your doctor."
    return " ".join(results) + guardrail

def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def calculate_egfr(creatinine, age, sex='male'):
    if sex == 'female':
        kappa, alpha, sex_factor = 0.7, -0.329, 1.018
    else:
        kappa, alpha, sex_factor = 0.9, -0.411, 1.0
    return 141 * min(creatinine/kappa,1)**alpha * max(creatinine/kappa,1)**-1.209 * 0.993**age * sex_factor

# ─── MODULE 3: Strict 8-Feature Enforcement for Diabetes ─────────────
DISEASE_CONFIG = {
    "Heart": {
        "model_path": "Datasets/Heart/Heart_model.keras",
        "features": ['thal','cp','ca','exang','thalach','chol','slope','oldpeak'],
        "dataset_path": "Datasets/Heart/Heart_cleaned.csv",
        "threshold": 0.8
    },
    "Kidney": {
        "model_path": "Datasets/Kidney/Kidney_model.keras",
        "features": ['sc','hemo','pcv','rc','al','htn','bu','bgr'],
        "dataset_path": "Datasets/Kidney/Kidney_cleaned.csv",
        "threshold": 0.9
    },
    "Diabetes": {
        "model_path": "Datasets/Diabetes/Diabetes_model.keras",
        "features": ['Glucose','BMI','Pregnancies','Age','DiabetesPedigreeFunction','Insulin','SkinThickness','BloodPressure'],
        "dataset_path": "Datasets/Diabetes/Diabetes_cleaned.csv",
        "threshold": 0.7
    }
}

def get_clinical_advice(disease, prob, threshold):
    if prob < 0.1:
        return {"level":"No Risk","meds":"No medication required.","treatment":"Annual wellness check-ups.","diet":"Balanced diet rich in whole foods."}
    high_risk_boundary = threshold
    medium_risk_boundary = high_risk_boundary * 0.5 

    if prob < medium_risk_boundary:
        return {"level":"Low Risk","meds":"Preventive care advised.","treatment":"Regular monitoring.","diet":"Balanced diet; reduce processed foods."}
    
    if prob < high_risk_boundary:
        advice = {
            "Diabetes": {"meds":"Consult endocrinologist for low-dose medication.","treatment":"Monitor blood glucose regularly.","diet":"Strictly monitor carbs."},
            "Heart": {"meds":"Consult cardiologist. Lifestyle changes critical.","treatment":"Consider stress test.","diet":"Heart-healthy diet."},
            "Kidney": {"meds":"Consult nephrologist. BP control crucial.","treatment":"Monitor eGFR. Avoid NSAIDs.","diet":"Kidney-friendly diet."}
        }
        r = advice.get(disease, {"meds":"Consult specialist","treatment":"Tests needed","diet":"Consult Nutritionist"})
        r["level"] = "Medium Risk"
        return r

    advice = {
        "Diabetes": {"meds":"Metformin (500mg), SGLT2 inhibitors.","treatment":"HbA1c test every 3 months.","diet":"Low carb, high fiber."},
        "Heart": {"meds":"Atorvastatin (20mg), Low-dose Aspirin (75mg).","treatment":"Stress Echo, Cardiac MRI.","diet":"Mediterranean diet."},
        "Kidney": {"meds":"ACE Inhibitors (Enalapril), Phosphate binders.","treatment":"GFR Monitoring.","diet":"Low sodium, controlled protein."}
    }
    r = advice.get(disease, {"meds":"Consult specialist","treatment":"Lab tests required","diet":"Consult Nutritionist"})
    r["level"] = "High Risk"
    return r

# ─── MODULE 3: Fast Permutation-Based Feature Importance (< 2s) ──────
# Replaces KernelExplainer which hangs for minutes on LSTM wrappers.
# Permutation importance: shuffle each feature N times, measure prediction drop.
def create_explanation_chart(model, training_features, scaled_patient_vitals, feature_names):
    try:
        feature_names = list(feature_names)
        n_features = len(feature_names)
        
        # Get baseline prediction
        base_input = scaled_patient_vitals.reshape(1, 1, n_features)
        base_pred = float(model.predict(base_input, verbose=0)[0][0])
        
        # Fast permutation importance — 20 shuffles per feature
        N_REPEATS = 20
        importances = []
        rng = np.random.default_rng(42)
        
        # FIX: scale the reference training data, otherwise we are swapping scaled patient features with unscaled raw data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(training_features)
        ref_data_scaled = scaler.transform(training_features)
        
        # Use up to 30 training samples as reference for permutation
        ref_data = ref_data_scaled[:30] if len(ref_data_scaled) > 30 else ref_data_scaled
        
        for i in range(n_features):
            # Batch permutations for instant parallel prediction
            perturbed_batch = np.repeat(scaled_patient_vitals, N_REPEATS, axis=0)
            random_rows = rng.integers(0, len(ref_data), size=N_REPEATS)
            perturbed_batch[:, i] = ref_data[random_rows, i]
            
            perturbed_batch_input = perturbed_batch.reshape(N_REPEATS, 1, n_features)
            perturbed_preds = model.predict(perturbed_batch_input, verbose=0)
            
            deltas = np.abs(base_pred - perturbed_preds[:, 0])
            mean_impact = float(np.mean(deltas))
            
            # Clinical Color Logic: if patient's value is higher than the dataset mean, it's considered 'High/Abnormal' (Positive/Red)
            is_high_value = scaled_patient_vitals[0, i] > np.mean(ref_data_scaled[:, i])
            
            # Assign positive mathematical sign for High values (Red), negative for Normal/Low (Green)
            clinical_shap_value = mean_impact if is_high_value else -mean_impact
            importances.append(clinical_shap_value)

        # Sort by absolute importance magnitude
        sorted_idx = np.argsort(np.abs(importances))[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = [importances[i] for i in sorted_idx]

        # Build structured data for frontend Chart.js
        shap_data = {
            "features": sorted_features,
            "values": sorted_values
        }

        # Build waterfall-style matplotlib chart
        colors = ['#DC2626' if v >= 0 else '#10B981' for v in sorted_values]
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = range(len(sorted_features))
        bars = ax.barh(list(y_pos), sorted_values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sorted_features, fontsize=10, fontweight='bold')
        ax.axvline(0, color='#334155', linewidth=1.2)
        ax.set_xlabel('Clinical Value Impact (Red = High/Warning, Green = Normal/Safe)', fontsize=10, color='#64748b')
        ax.set_title('Feature Importance & Patient Vitals', fontsize=13, fontweight='bold', color='#0f172a', pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        
        # Value labels on bars
        for bar, val in zip(bars, sorted_values):
            ax.text(val + (0.001 if val >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                    f'{abs(val):.3f}', va='center', ha='left' if val >= 0 else 'right',
                    fontsize=8, color='#334155')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        print(f"[SHAP] Feature importance computed for {n_features} features. Base pred: {base_pred:.3f}")
        return chart_b64, shap_data
    except Exception as e:
        print(f"Feature importance error: {e}")
        import traceback; traceback.print_exc()
        return None, None

def run_diagnostic(disease_type, vitals_dict):
    try:
        if disease_type not in DISEASE_CONFIG:
            return 0.0, "", None, {"meds":"Invalid config","treatment":"N/A","diet":"N/A"}

        config = DISEASE_CONFIG[disease_type]
        model_path = os.path.join(os.getcwd(), config["model_path"])
        dataset_path = os.path.join(os.getcwd(), config["dataset_path"])
        if not os.path.exists(model_path) or not os.path.exists(dataset_path):
            return 0.0, "", None, {"meds":"Missing Data","treatment":"N/A","diet":"N/A"}
            
        model = load_model(model_path)
        training_data = pd.read_csv(dataset_path)
        for feature in config["features"]:
            if feature in training_data.columns:
                training_data[feature] = pd.to_numeric(training_data[feature], errors='coerce')
        training_data.dropna(subset=config["features"], inplace=True)
        
        scaler = MinMaxScaler()
        scaler.fit(training_data[config["features"]])
        
        # MODULE 3: Strictly use only the configured features
        ordered_vitals = [to_float(vitals_dict.get(feature, 0.0)) for feature in config["features"]]
        patient_df = pd.DataFrame([ordered_vitals], columns=config["features"])
        scaled_vitals = scaler.transform(patient_df)
        
        # CRITICAL FIX: Clip input values [0, 1] so extreme out-of-bounds inputs (e.g., BP 200) 
        # do not cause neural network saturation and invert the predictions.
        scaled_vitals = np.clip(scaled_vitals, 0.0, 1.0)
        
        # Validate tensor shape matches model expectation
        expected_features = len(config["features"])
        actual_features = scaled_vitals.shape[1]
        if actual_features != expected_features:
            print(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
            return 0.0, "", None, {"meds":"Feature mismatch","treatment":"Check config","diet":"N/A"}
        
        final_input = scaled_vitals.reshape(1, 1, expected_features)
        
        # Check model input shape compatibility
        model_input_shape = model.input_shape
        if model_input_shape[2] != expected_features:
            print(f"MODEL SHAPE MISMATCH: model expects {model_input_shape[2]} features, config has {expected_features}")
            print(f"Attempting inference anyway with {expected_features} features...")
            # If model was trained with extra features, pad with zeros
            if model_input_shape[2] > expected_features:
                pad_width = model_input_shape[2] - expected_features
                padded = np.pad(scaled_vitals, ((0,0),(0,pad_width)), mode='constant', constant_values=0)
                final_input = padded.reshape(1, 1, model_input_shape[2])
                print(f"Padded input from {expected_features} to {model_input_shape[2]} features")
        
        prob = float(model.predict(final_input, verbose=0)[0][0])
        
        # Critical Clinical Safety Net (Overrides raw dataset bias)
        if disease_type == 'Kidney':
            hemo = to_float(vitals_dict.get('hemo'))
            if hemo > 0 and hemo < 10.0:
                prob = max(prob, 0.95)  # Severe anemia guarantees high risk
        
        clinical_plan = get_clinical_advice(disease_type, prob, config['threshold'])
        chart_base64, shap_data = create_explanation_chart(model, training_data[config["features"]], scaled_vitals, config["features"])

        return prob, chart_base64, shap_data, clinical_plan
    except Exception as e:
        print(f"Backend Diagnostic Error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, "", None, {"meds":"Internal Error","treatment":"Check Logs","diet":"N/A"}

@app.route('/patient_login', methods=['POST'])
def patient_login():
    data = request.get_json()
    name = data.get('name')
    age = data.get('age')
    if not name or not age:
        return jsonify({'error': 'Name and Age are required'}), 400
    
    # Create a consistent username from the name and age
    username = name.lower().replace(" ", "") + str(age)
    
    token = jwt.encode({
        'user': name, 'role': 'PATIENT', 'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }, JWT_SECRET, algorithm="HS256")
    
    db = get_db_session()
    patient = db.query(Patient_Profiles).filter(Patient_Profiles.username == username).first()
    if not patient:
        db.add(Patient_Profiles(username=username, name=name, family_history=''))
        db.commit()
    db.close()
    
    return jsonify({'token': token, 'user': name, 'role': 'PATIENT', 'username': username}), 200

def load_users():
    with open('users.json', 'r') as f:
        return json.load(f)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    users = load_users()
    for user in users:
        if data.get('username') == user['username'] and data.get('password') == user['password']:
            token = jwt.encode({
                'user': user['name'], 'role': user['role'], 'username': user['username'],
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
            }, JWT_SECRET, algorithm="HS256")
            db = get_db_session()
            patient = db.query(Patient_Profiles).filter(Patient_Profiles.username == user['username']).first()
            if not patient:
                db.add(Patient_Profiles(username=user['username'], name=user['name'], family_history=''))
                db.commit()
            db.close()
            return jsonify({'token': token, 'user': user['name'], 'role': user['role']}), 200
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/demo_login', methods=['GET', 'POST'])
def demo_login():
    data = request.get_json(silent=True) or {}
    requested_role = data.get('role', 'DOCTOR')
    users = load_users()
    demo_user = next((u for u in users if u['role'] == requested_role), users[0])
    token = jwt.encode({
        'user': demo_user['name'], 'role': demo_user['role'], 'username': demo_user['username'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }, JWT_SECRET, algorithm="HS256")
    db = get_db_session()
    patient = db.query(Patient_Profiles).filter(Patient_Profiles.username == demo_user['username']).first()
    if not patient:
        db.add(Patient_Profiles(username=demo_user['username'], name=demo_user['name'], family_history=''))
        db.commit()
    db.close()
    return jsonify({'token':token,'user':demo_user['name'],'role':demo_user['role'],'username':demo_user['username']}), 200

@app.route('/predict', methods=['POST'])
def predict():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Unauthorized'}), 401
    token = token.replace('Bearer ', '').strip()
    
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        patient_username = decoded_token.get('username', 'demo')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        # MODULE 1: Allow demo tokens to pass through gracefully
        patient_username = 'doctor1'
    
    try:
        data = request.json
        disease_type = data.get('type')
        vitals = data.get('vitals', {})
        
        # Override token user if Doctor explicitly mapped a target patient Name/ID
        explicit_patient = data.get('patientName')
        if explicit_patient:
            patient_username = explicit_patient.strip().lower()

        # MODULE 3: Tensor Validation — check disease type exists
        if disease_type not in DISEASE_CONFIG:
            return jsonify({'error': f'Unknown disease type: {disease_type}'}), 400

        model_vitals = vitals.copy()
        visit_date_str = data.get('visitDate')
        visit_date_obj = datetime.datetime.utcnow()
        if visit_date_str:
            try:
                visit_date_obj = datetime.datetime.strptime(visit_date_str, '%Y-%m-%d')
            except ValueError:
                pass

        if disease_type == 'Kidney':
            creatinine = to_float(vitals.get('sc'))
            age = to_float(vitals.get('Age'))
            if creatinine and age:
                model_vitals['eGFR'] = calculate_egfr(creatinine, age)
            if 'Creatinine' in model_vitals: del model_vitals['Creatinine']
            if 'Albumin' in model_vitals: model_vitals['al'] = model_vitals.pop('Albumin')
            if 'Sugar' in model_vitals: model_vitals['bgr'] = model_vitals.pop('Sugar')
            if 'BP' in model_vitals: model_vitals['htn'] = model_vitals.pop('BP')
            if 'Hemoglobin' in model_vitals: model_vitals['hemo'] = model_vitals.pop('Hemoglobin')

        elif disease_type == 'Diabetes':
            # MODULE 3: Map frontend field names to model feature names
            if 'Pedigree' in model_vitals:
                model_vitals['DiabetesPedigreeFunction'] = model_vitals.pop('Pedigree')
            # Map fasting_glucose → Glucose (the model feature)
            if 'fasting_glucose' in model_vitals:
                model_vitals['Glucose'] = model_vitals['fasting_glucose']

        # MODULE 3: Enforce exactly N features — validate before inference
        required_features = DISEASE_CONFIG[disease_type]['features']
        n_required = len(required_features)
        missing = [f for f in required_features if f not in model_vitals or model_vitals.get(f) is None]
        if missing:
            return jsonify({'error': f'Missing {len(missing)} required features: {missing}', 'missing_features': missing}), 400

        # Strip to only the required features for validation
        feature_count = sum(1 for f in required_features if f in model_vitals)
        print(f"[PREDICT] {disease_type}: {feature_count}/{n_required} features present")
        
        prob, chart, shap_data, plan = run_diagnostic(disease_type, model_vitals)
        
        if plan.get('level') == 'No Risk':
            prob = 0.0

        if prob > 0.7:
            send_push_notification(patient_username, f'High risk detected for {disease_type}.')
        elif prob < 0.3:
            send_push_notification(patient_username, 'Risk levels are low.')

        rag_query = f"Diet plan specifically for {disease_type} disease at {plan.get('level')} risk."
        diet_tips = retrieve_medical_advice(rag_query)
        
        base_diet = plan.get('diet', '')
        plan['diet'] = f"{base_diet} — {diet_tips}"

        # Save to database
        try:
            db = get_db_session()
            
            # MODULE 4: Smart Patient Matching Logic (consistent with get_patient_record)
            patient = db.query(Patient_Profiles).filter(Patient_Profiles.username == patient_username).first()
            
            if not patient:
                base_name = ''.join([c for c in patient_username if not c.isdigit()])
                if base_name:
                    patient = db.query(Patient_Profiles).filter(Patient_Profiles.username == base_name).first()
            
            if not patient:
                patient = db.query(Patient_Profiles).filter(Patient_Profiles.username.ilike(f"%{patient_username}%")).first()

            if not patient:
                patient = Patient_Profiles(username=patient_username, name=patient_username.capitalize(), family_history='')
                db.add(patient)
                db.commit()
                
            if patient:
                fasting_glucose = vitals.get('fasting_glucose', vitals.get('Glucose'))
                post_prandial_glucose = vitals.get('post_prandial_glucose')
                hba1c = vitals.get('hba1c')
                glucose_delta = None
                if fasting_glucose and post_prandial_glucose:
                    glucose_delta = str(float(post_prandial_glucose) - float(fasting_glucose))
                new_visit = Patient_Visits(
                    patient_id=patient.id, disease_type=disease_type, vitals=json.dumps(vitals),
                    prediction_prob=float(prob), clinical_plan=json.dumps(plan), chart_image=chart,
                    fasting_glucose=str(fasting_glucose) if fasting_glucose else None,
                    post_prandial_glucose=str(post_prandial_glucose) if post_prandial_glucose else None,
                    glucose_delta=glucose_delta, hba1c=str(hba1c) if hba1c else None,
                    visit_date=visit_date_obj
                )
                db.add(new_visit)
                db.commit()
                
                if disease_type == 'Diabetes' and not post_prandial_glucose:
                    try:
                        schedule_post_prandial_notification.apply_async(args=[new_visit.id])
                        print(f"[CELERY] Scheduled 2-hour post-prandial reminder for Visit ID: {new_visit.id}")
                    except Exception as celery_err:
                        print(f"[CELERY] Failed to schedule task: {celery_err}")
                        
            db.close()
        except Exception as db_err:
            print(f"DB save error (non-fatal): {db_err}")

        response_data = {
            'prob': float(prob),
            'chart': chart,
            'shap_data': shap_data,
            'plan': plan,
            'disease': plan.get('level'),
            'patient_name': patient_username,
            'vitals': vitals,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Auth/Predict Route Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/what_if_simulation', methods=['POST'])
def what_if_simulation():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json
        prob, chart, shap_data, _ = run_diagnostic(data.get('type'), data.get('vitals', {}))
        return jsonify({'prob': float(prob), 'chart': chart, 'shap_data': shap_data}), 200
    except Exception as e:
        return jsonify({'error': 'Simulation error'}), 400

@app.route('/patient/record', methods=['POST'])
def get_patient_record():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    token = token.replace('Bearer ', '').strip()
    try:
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            doctor_id = decoded.get('username')
        except jwt.InvalidTokenError:
            doctor_id = 'doctor1'
        
        data = request.get_json()
        raw_username = data.get('username')
        if not raw_username:
            return jsonify([]), 200
            
        patient_username = raw_username.strip().lower().replace(" ", "")
        db = get_db_session()
        
        # MODULE 4: Aggregate results from all related profiles (handling varsha vs varsha60)
        base_name = ''.join([c for c in patient_username if not c.isdigit()])
        patient_ids = db.query(Patient_Profiles.id).filter(
            (Patient_Profiles.username == patient_username) |
            (Patient_Profiles.username == base_name) |
            (Patient_Profiles.username.ilike(f"%{patient_username}%"))
        ).all()
        
        all_ids = [pid[0] for pid in patient_ids]
        
        if all_ids:
            visits = db.query(Patient_Visits).filter(Patient_Visits.patient_id.in_(all_ids)).order_by(Patient_Visits.visit_date.desc()).all()
            
            if visits:
                # Log audit for the first matched patient
                audit = Audit_Log(doctor_id=doctor_id, action='ACCESS_PATIENT_RECORD', patient_id=all_ids[0])
                db.add(audit)
                db.commit()
                
                records = []
                for visit in visits:
                    records.append({
                        'id': visit.id, 'visit_date': visit.visit_date.isoformat() if visit.visit_date else None,
                        'disease_type': visit.disease_type, 'vitals': json.loads(visit.vitals) if visit.vitals else {},
                        'prediction_prob': visit.prediction_prob, 'clinical_plan': json.loads(visit.clinical_plan) if visit.clinical_plan else {},
                        'chart_image': visit.chart_image, 'fasting_glucose': visit.fasting_glucose,
                        'post_prandial_glucose': visit.post_prandial_glucose, 'glucose_delta': visit.glucose_delta, 'hba1c': visit.hba1c
                    })
                db.close()
                return jsonify(records), 200
        
        db.close()
        return jsonify([]), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Invalid Session'}), 401

@app.route('/patient/risk_trend', methods=['POST'])
def get_patient_risk_trend():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    token = token.replace('Bearer ', '').strip()
    try:
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            decoded = {'username': 'doctor1'}
        doctor_id = decoded.get('username')
        data = request.json
        raw_username = data.get('username')
        
        if not raw_username:
            return jsonify([]), 200

        patient_username = raw_username.strip().lower().replace(" ", "")
        db = get_db_session()

        base_name = ''.join([c for c in patient_username if not c.isdigit()])
        patient_ids = db.query(Patient_Profiles.id).filter(
            (Patient_Profiles.username == patient_username) |
            (Patient_Profiles.username == base_name) |
            (Patient_Profiles.username.ilike(f"%{patient_username}%"))
        ).all()
        
        all_ids = [pid[0] for pid in patient_ids]
        
        if all_ids:
            visits = db.query(Patient_Visits).filter(Patient_Visits.patient_id.in_(all_ids)).order_by(Patient_Visits.visit_date.desc()).limit(5).all()
            
            if visits:
                audit = Audit_Log(doctor_id=doctor_id, action='ACCESS_RISK_TREND', patient_id=all_ids[0])
                db.add(audit)
                db.commit()
                records = [{'prediction_prob': v.prediction_prob, 'visit_date': v.visit_date.isoformat() if v.visit_date else None} for v in visits]
                db.close()
                return jsonify(records), 200
        
        db.close()
        return jsonify([]), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Invalid Session'}), 401

@app.route('/patient/notifications', methods=['GET'])
def get_notifications():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    
    token = token.replace('Bearer ', '').strip()
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        patient_username = decoded_token.get('username')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        patient_username = 'doctor1'
        
    db = get_db_session()
    
    # Aggregated Fuzzy Patient Matching for Notifications
    base_name = ''.join([c for c in patient_username if not c.isdigit()])
    patient_ids = db.query(Patient_Profiles.id).filter(
        (Patient_Profiles.username == patient_username) |
        (Patient_Profiles.username == base_name) |
        (Patient_Profiles.username.ilike(f"%{patient_username}%"))
    ).all()
    
    all_ids = [pid[0] for pid in patient_ids]
    
    if not all_ids:
        db.close()
        return jsonify([])
        
    notifications = db.query(Notification).filter(Notification.patient_id.in_(all_ids), Notification.is_read == 0).order_by(Notification.timestamp.desc()).all()
    results = [{"id": n.id, "message": n.message, "timestamp": n.timestamp.isoformat() if n.timestamp else None} for n in notifications]
    db.close()
    return jsonify(results), 200

@app.route('/patient/notifications/<int:notif_id>/read', methods=['POST'])
def mark_notification_read(notif_id):
    db = get_db_session()
    notif = db.query(Notification).filter(Notification.id == notif_id).first()
    if notif:
        notif.is_read = 1
        db.commit()
    db.close()
    return jsonify({"success": True}), 200

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
