import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURATION ---
FILES_TO_TRAIN = {
    "Heart":    {"file": "Datasets/Heart/Heart_final.csv",       "target": "target"},
    "Kidney":   {"file": "Datasets/Kidney/Kidney_final.csv",     "target": "classification"},
    "Diabetes": {"file": "Datasets/Diabetes/Diabetes_final.csv", "target": "Outcome"}
}

def build_multitier_lstm(input_shape):
    """
    Builds the Multi-Tier Deep Learning Model defined in Phase 3.
    """
    model = Sequential()
    
    # Tier 1: LSTM Sequence Learning
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    
    # Tier 2: Dropout (Regularization)
    model.add(Dropout(0.2))
    
    # Tier 3: Classification
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def calculate_egfr(creatinine, age, sex='male'):
    """Calculates eGFR using the CKD-EPI formula."""
    if sex == 'female':
        kappa = 0.7
        alpha = -0.329
        sex_factor = 1.018
    else: # male
        kappa = 0.9
        alpha = -0.411
        sex_factor = 1.0

    return 141 * min(creatinine / kappa, 1)**alpha * max(creatinine / kappa, 1)**-1.209 * 0.993**age * sex_factor


def train_and_save():
    print("--- STARTING MULTI-TIER LSTM TRAINING (SAVING TO SUBFOLDERS) ---")
    
    for disease, config in FILES_TO_TRAIN.items():
        filename = config['file']
        target_col = config['target']
        
        # 1. Validation
        if not os.path.exists(filename):
            print(f"\n[SKIP] Could not find {filename}. Please run part2_feature_selection.py first.")
            continue
            
        print(f"\n==========================================")
        print(f" TRAINING MODEL: {disease.upper()}")
        print(f"==========================================")
        
        # 2. Load Data
        df = pd.read_csv(filename)

        # MODULE 3: Diabetes model uses exactly 8 features from the cleaned dataset
        # Do NOT add hba1c or glucose_delta_trend — they cause tensor shape mismatch
        if disease == 'Diabetes':
            pass  # Strict 8-feature enforcement
        
        if disease == 'Kidney':
            # Assuming 'age' and 'creatinine' columns exist in the dataset
            if 'age' in df.columns and 'creatinine' in df.columns:
                 df['eGFR'] = df.apply(lambda row: calculate_egfr(row['creatinine'], row['age']), axis=1)

        
        # 3. Separate X (Features) and y (Target)
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # 4. Reshape for LSTM [Samples, TimeSteps, Features]
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # 5. Split Data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)
        
        # 6. Configure Training Parameters
        if disease.lower() == "diabetes":
            epochs = 100
            batch_size = 16
            print(f"   -> Mode: High-Intensity Training (Diabetes Fix)")
        else:
            epochs = 30
            batch_size = 32
            print(f"   -> Mode: Standard Training")

        # 7. Build Model
        input_shape = (1, X_train.shape[2])
        model = build_multitier_lstm(input_shape)
        
        # 8. Train
        print(f"   -> Training on {X_train.shape[0]} samples with {X_train.shape[2]} features...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 9. Evaluate Performance
        print("\n   [EVALUATION REPORT]")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"   ------------------------------------------")
        print(f"   FINAL TEST ACCURACY: {accuracy:.2%}") 
        print(f"   ------------------------------------------")
        
        # 10. Save Model to Specific Folder
        # Get the folder from the input filename (e.g., Datasets/Heart)
        folder_path = os.path.dirname(filename)
        
        model_name = f"{disease.capitalize()}_model.keras"
        save_path = os.path.join(folder_path, model_name)
        
        model.save(save_path)
        print(f"   [SAVED] Model saved to {save_path}")

if __name__ == "__main__":
    train_and_save()
