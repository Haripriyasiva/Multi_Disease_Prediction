import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# --- CONFIGURATION ---
# These match the output locations from your updated Part 1 script
FILES_TO_PROCESS = {
    "Heart":    {"input": "Datasets/Heart/Heart_cleaned.csv",       "target": "target"},
    "Kidney":   {"input": "Datasets/Kidney/Kidney_cleaned.csv",     "target": "classification"}, 
    "Diabetes": {"input": "Datasets/Diabetes/Diabetes_cleaned.csv", "target": "Outcome"}
}

def hybrid_optimization(df, target_col, k=8):
    """
    Implements the 4-Stage Hybrid Pipeline:
    1. Correlation Matrix (Removes Redundant Data)
    2. ANOVA (Selects Best Numerical Features)
    3. Chi-Square (Selects Best Categorical Features)
    4. Mutual Information (Final Non-Linear Selection)
    """
    print(f"   Starting Hybrid Selection (Target: {k} features)...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify types dynamically
    cat_features = [c for c in X.columns if X[c].nunique() < 10]
    num_features = [c for c in X.columns if c not in cat_features]

    # --- STAGE 1: CORRELATION FILTER ---
    if len(num_features) > 1:
        corr = X[num_features].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        
        if to_drop:
            print(f"   [1] Correlation Matrix dropped redundant: {to_drop}")
            X = X.drop(columns=to_drop)
            # Update lists
            cat_features = [c for c in X.columns if c in cat_features]
            num_features = [c for c in X.columns if c not in cat_features]

    # --- STAGE 2: STATISTICAL FILTER (ANOVA + Chi-Square) ---
    candidates = set()
    
    # A. ANOVA (Numerical)
    if len(num_features) > 0:
        try:
            k_num = max(1, len(num_features) // 2)
            anova = SelectKBest(score_func=f_classif, k=k_num)
            anova.fit(X[num_features], y)
            selected_num = list(X[num_features].columns[anova.get_support()])
            candidates.update(selected_num)
            print(f"   [2] ANOVA selected: {selected_num}")
        except:
            candidates.update(num_features)

    # B. Chi-Square (Categorical)
    if len(cat_features) > 0:
        try:
            k_cat = max(1, len(cat_features) // 2)
            chi_sq = SelectKBest(score_func=chi2, k=k_cat)
            chi_sq.fit(X[cat_features], y)
            selected_cat = list(X[cat_features].columns[chi_sq.get_support()])
            candidates.update(selected_cat)
            print(f"   [3] Chi-Square selected: {selected_cat}")
        except:
            candidates.update(cat_features)

    # Fallback if filters failed
    if len(candidates) < k:
        candidates = list(X.columns)
    else:
        candidates = list(candidates)

    X_subset = X[candidates]

    # --- STAGE 3: MUTUAL INFORMATION (Final Selection) ---
    print("   [4] Calculating Mutual Information (Final Stage)...")
    
    discrete_mask = [col in cat_features for col in X_subset.columns]
    mi_scores = mutual_info_classif(X_subset, y, discrete_features=discrete_mask, random_state=42)
    
    final_scores = dict(zip(X_subset.columns, mi_scores))
    sorted_feats = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_selection = [x[0] for x in sorted_feats[:k]]
    
    print(f"   -> FINAL HYBRID SELECTION: {final_selection}")
    return df[final_selection + [target_col]]

def normalize(df, target_col):
    """
    Normalizes data to 0-1 range (Required for LSTM Accuracy)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    final_df = pd.DataFrame(X_scaled, columns=X.columns)
    final_df[target_col] = y.values
    return final_df

if __name__ == "__main__":
    print("--- PART 2: FEATURE SELECTION (SAVING TO SUBFOLDERS) ---")
    
    for disease, config in FILES_TO_PROCESS.items():
        try:
            # 1. Load cleaned data
            df = pd.read_csv(config['input'])
            print(f"\nProcessing {disease.upper()}...")
            
            # 2. Run Optimization
            df_opt = hybrid_optimization(df, config['target'], k=8)
            
            # 3. Normalize
            df_final = normalize(df_opt, config['target'])
            
            # 4. Save to Specific Folder
            # Extract the folder path from the input (e.g., Datasets/Heart)
            folder_path = os.path.dirname(config['input'])
            
            output_filename = f"{disease}_final.csv"
            save_path = os.path.join(folder_path, output_filename)
            
            df_final.to_csv(save_path, index=False)
            print(f"   -> SUCCESS: Saved final data to: {save_path}")
            
        except FileNotFoundError:
            print(f"Error: {config['input']} not found. Please run part1_preprocessing.py first.")