import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
# The output will be saved inside these paths
DATASET_FOLDERS = {
    "Heart":    {"path": "Datasets/Heart",    "target": "target"},
    "Kidney":   {"path": "Datasets/Kidney",   "target": "classification"},
    "Diabetes": {"path": "Datasets/Diabetes", "target": "Outcome"}
}

def load_and_merge(folder_path):
    """Loads all CSVs in a folder and merges them."""
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Filter out files that already contain "_cleaned" or "_final" to avoid re-reading outputs
    files = [f for f in files if "_cleaned" not in f and "_final" not in f]

    if not files:
        print(f"Warning: No raw CSV files found in {folder_path}")
        return None
    
    df_list = []
    for f in files:
        try:
            temp = pd.read_csv(f)
            df_list.append(temp)
            print(f"   -> Loaded {os.path.basename(f)}: {temp.shape}")
        except Exception as e:
            print(f"   Skipping {f}: {e}")
            
    if df_list:
        return pd.concat(df_list, axis=0, ignore_index=True)
    return None

def preprocess_data(df, target_col, disease_name):
    print(f"   Original Shape: {df.shape}")
    
    # --- 1. SPECIAL FIX FOR DIABETES (The "Zero" Problem) ---
    if disease_name.lower() == "diabetes":
        print("   [DIABETES FIX] Replacing 0 with Median for biological columns...")
        cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    # --- 2. SPECIAL FIX FOR KIDNEY (Dirty Text) ---
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        df[target_col] = df[target_col].replace({'ckd\t': 'ckd'})

    # --- 3. ENCODE TARGET ---
    if target_col in df.columns:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        
        if len(df[target_col].unique()) < 2:
            print(f"   CRITICAL WARNING: Only one class found in target! Model cannot learn.")
    
    # --- 4. CLEANING ---
    df = df.replace('?', np.nan)
    df = df.dropna(subset=[target_col])
    
    # --- 5. FILL MISSING VALUES ---
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if not X[col].mode().empty:
            X[col] = X[col].fillna(X[col].mode()[0])

    # --- 6. ENCODE FEATURES ---
    for col in X.columns:
        if X[col].dtype == 'object':
            le_feat = LabelEncoder()
            X[col] = le_feat.fit_transform(X[col].astype(str))

    # --- 7. REMOVE DUPLICATES ---
    df_clean = pd.concat([X, y], axis=1)
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < before_dedup:
        print(f"   [FIX] Removed {before_dedup - len(df_clean)} duplicate rows.")
        
    return df_clean

if __name__ == "__main__":
    print("--- PART 1: DATA PREPROCESSING (SAVING TO SUBFOLDERS) ---")
    
    for disease, config in DATASET_FOLDERS.items():
        print(f"\nProcessing {disease.upper()}...")
        
        raw_df = load_and_merge(config['path'])
        
        if raw_df is not None:
            # Handle Target Column Mismatches
            current_target = config['target']
            if current_target not in raw_df.columns:
                 if 'classification' in raw_df.columns: current_target = 'classification'
                 elif 'class' in raw_df.columns: current_target = 'class'
                 elif 'Outcome' in raw_df.columns: current_target = 'Outcome'

            if current_target in raw_df.columns:
                # Run Preprocessing
                clean_df = preprocess_data(raw_df, current_target, disease)
                
                # --- SAVE LOGIC MODIFIED HERE ---
                # Saves to: Datasets/Heart/Heart_cleaned.csv
                output_filename = f"{disease}_cleaned.csv"
                save_path = os.path.join(config['path'], output_filename)
                
                clean_df.to_csv(save_path, index=False)
                print(f"   -> SUCCESS: Saved cleaned data to: {save_path}")
            else:
                print(f"   ERROR: Target column '{config['target']}' not found!")
    
    print("\n[DONE] Now proceed to Feature selection.")