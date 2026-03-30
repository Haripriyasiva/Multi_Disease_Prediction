import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. UNIFIED CONFIGURATION DICTIONARY
# ==========================================
# Holds the Top 8 features selected via Phase 2 Hybrid Optimization
DISEASE_CONFIG = {
    "Heart": {
        "model_path": "Datasets/Heart/Heart_model.keras",
        "features": ['thal', 'cp', 'ca', 'exang', 'thalach', 'chol', 'slope', 'oldpeak']
    },
    "Kidney": {
        "model_path": "Datasets/Kidney/Kidney_model.keras",
        "features": ['id', 'hemo', 'pcv', 'rc', 'al', 'htn', 'bgr', 'bu']
    },
    "Diabetes": {
        "model_path": "Datasets/Diabetes/Diabetes_model.keras",
        "features": ['Glucose', 'BMI', 'Pregnancies', 'Age', 'DiabetesPedigreeFunction', 'Insulin', 'SkinThickness', 'BloodPressure']
    }
}

scaler = MinMaxScaler()

# ==========================================
# 2. LOAD OFFLINE MEDICAL ASSISTANT (RAG)
# ==========================================
print("Booting up the Offline Medical Assistant (BERT + FAISS)...")
# Uses a local BERT model to ensure privacy and hallucination-free guidance
rag_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index("medical_knowledge.index")

with open("medical_text_mapping.pkl", "rb") as f:
    medical_faqs = pickle.load(f)

def retrieve_medical_advice(user_query):
    """Semantic search to find verified medical guidance."""
    query_vector = rag_model.encode([user_query]).astype("float32")
    distances, indices = faiss_index.search(query_vector, 1)
    return medical_faqs[indices[0][0]]

# ==========================================
# 3. THE DIAGNOSTIC ENGINE
# ==========================================
def execute_diagnostic_pipeline(disease_type, raw_patient_data, background_training_data):
    
    if disease_type not in DISEASE_CONFIG:
        return "Error: Invalid disease type selected."
        
    config = DISEASE_CONFIG[disease_type]
    optimal_features = config["features"]
    
    print(f"\n[{disease_type.upper()} DISEASE PREDICTION INITIATED]")
    
    # --- PREPROCESSING ---
    # Median Imputation and Min-Max Scaling for data stability
    model = load_model(config["model_path"])
    raw_patient_data.fillna(raw_patient_data.median(), inplace=True)
    
    scaled_data = scaler.fit_transform(raw_patient_data)
    scaled_df = pd.DataFrame(scaled_data, columns=raw_patient_data.columns)
    
    # --- HYBRID FEATURE SELECTION & LSTM RESHAPING ---
    patient_optimal_vector = scaled_df[optimal_features].values
    lstm_input = np.reshape(patient_optimal_vector, (patient_optimal_vector.shape[0], 1, patient_optimal_vector.shape[1]))
    
    # --- DEEP LEARNING PREDICTION ---
    # Using Adam Optimizer and Sigmoid Activation for classification
    probability_score = model.predict(lstm_input)[0][0]
    
    print("--- DIAGNOSTIC RESULTS ---")
    
    if probability_score > 0.5:
        print(f"Status: DISEASE DETECTED")
        print(f"Probability Score: {probability_score * 100:.2f}%")
        print("Risk Level: HIGH RISK" if probability_score >= 0.75 else "Risk Level: MODERATE RISK")
            
        print("\n--- TRIGGERING SHAP EXPLAINER ---")
        # GradientExplainer provides interpretability for the LSTM layers
        background_lstm = np.reshape(background_training_data, (background_training_data.shape[0], 1, background_training_data.shape[1]))
        explainer = shap.GradientExplainer(model, background_lstm)
        shap_values = explainer.shap_values(lstm_input)
        
        print(f"SHAP values computed. Rendering clinical waterfall plot...")
        
        # Flattening matrix output to 1D for individual patient waterfall plot
        if isinstance(shap_values, list):
            shap_values_2d = shap_values[0][:, 0, :]
        else:
            shap_values_2d = shap_values[:, 0, :]
            
        shap_values_1d = shap_values_2d[0].flatten()
        patient_vector_1d = lstm_input[0, 0, :].flatten()
        
        # Robustly fetch the expected baseline value
        try:
            base_val = explainer.expected_value
        except AttributeError:
            base_val = 0.5 
            
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

        # Generate a professional Waterfall Plot for the single patient checkup
        explanation = shap.Explanation(
            values=shap_values_1d,
            base_values=float(base_val),
            data=patient_vector_1d,
            feature_names=optimal_features
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"Clinical Interpretation: {disease_type} Risk Factors")
        plt.tight_layout()
        plt.show()
        
    else:
        print(f"Status: NO DISEASE DETECTED")
        print(f"Probability Score: {probability_score * 100:.2f}%")
        
    # --- SMART OUTPUT & SUPPORT ---
    # Retrieves relevant maintenance tips from the offline RAG system
    print("\n--- PATIENT SUPPORT (OFFLINE ASSISTANT) ---")
    automated_query = f"Management tips for {disease_type} disease"
    retrieved_advice = retrieve_medical_advice(automated_query)
    
    print(f"System Query: '{automated_query}'")
    print(f"Verified Guidance: {retrieved_advice}")

# ==========================================
# 4. EXAMPLE EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Simulated background data for SHAP baseline
    mock_background = np.random.rand(100, 8) 
    
    # Heart Test (Positive Risk Case)
    mock_heart = pd.DataFrame([{
        'age': 58, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250, 
        'fbs': 0, 'restecg': 1, 'thalach': 140, 'exang': 0, 
        'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 2
    }])
    execute_diagnostic_pipeline("Heart", mock_heart, mock_background)

    # Kidney Test
    mock_kidney = pd.DataFrame([{
        'id': 1, 'age': 48, 'bp': 80, 'sg': 1.02, 'al': 1, 'su': 0, 'bgr': 121, 
        'bu': 36, 'sc': 1.2, 'sod': 135, 'pot': 4.5, 'hemo': 15.4, 'pcv': 44, 
        'wc': 7800, 'rc': 5.2, 'htn': 1, 'dm': 1, 'cad': 0, 'appet': 0, 'pe': 0, 'ane': 0
    }])
    execute_diagnostic_pipeline("Kidney", mock_kidney, mock_background)

    # Diabetes Test
    mock_diabetes = pd.DataFrame([{
        'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35, 
        'Insulin': 0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50
    }])
    execute_diagnostic_pipeline("Diabetes", mock_diabetes, mock_background)