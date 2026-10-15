"""
preprocessing.py
Data preprocessing, cleaning, outlier handling, encoding, scaling,
and train/validation/test stratified splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Removes zero-variance / uninformative columns (EmployeeCount, Over18, StandardHours, EmployeeNumber)
    - Deduplicates rows
    - Converts target variable 'Attrition' to binary integer (Yes=1, No=0)
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Drop constant and identifier columns
    cols_to_drop = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    existing_drops = [c for c in cols_to_drop if c in cleaned_df.columns]
    if existing_drops:
        cleaned_df = cleaned_df.drop(columns=existing_drops)
        print(f"[Preprocessing] Dropped uninformative columns: {existing_drops}")
        
    # Map target variable
    if "Attrition" in cleaned_df.columns and cleaned_df["Attrition"].dtype == object:
        cleaned_df["Attrition"] = cleaned_df["Attrition"].map({"Yes": 1, "No": 0})
        print("[Preprocessing] Encoded Attrition: {'Yes': 1, 'No': 0}")
        
    return cleaned_df


def detect_and_handle_outliers_iqr(df: pd.DataFrame, continuous_cols: list, factor: float = 1.5) -> tuple[pd.DataFrame, dict]:
    """
    Detects and treats outliers using IQR capping (Winsorization) to preserve data volume
    while eliminating extreme distribution distortions.
    """
    treated_df = df.copy()
    outlier_stats = {}
    
    for col in continuous_cols:
        if col in treated_df.columns:
            q25 = treated_df[col].quantile(0.25)
            q75 = treated_df[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (factor * iqr)
            upper_bound = q75 + (factor * iqr)
            
            num_outliers = int(((treated_df[col] < lower_bound) | (treated_df[col] > upper_bound)).sum())
            outlier_stats[col] = {
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_count": num_outliers,
                "outlier_pct": round((num_outliers / len(treated_df)) * 100, 2)
            }
            # Winsorize / Cap boundaries
            treated_df[col] = treated_df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
            
    print(f"[Preprocessing] Outlier treatment applied to {len(continuous_cols)} continuous features.")
    return treated_df, outlier_stats


def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    output_dir: str = "data/processed"
) -> dict:
    """
    Comprehensive preprocessing pipeline:
    1. Clean raw dataset
    2. Outlier treatment
    3. One-hot encoding for nominal categoricals
    4. Stratified splitting: Train (70%), Validation (15%), Test (15%)
    5. StandardScaler fitted on Train and applied to Val & Test
    6. Save processed datasets to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clean
    cleaned = clean_dataset(df)
    
    continuous_features = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
        "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    
    # 2. Outlier treatment
    treated, outlier_stats = detect_and_handle_outliers_iqr(cleaned, continuous_features)
    
    # 3. Categorical encoding
    categorical_cols = treated.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        encoded_df = pd.get_dummies(treated, columns=categorical_cols, drop_first=True, dtype=float)
        print(f"[Preprocessing] One-Hot Encoded {len(categorical_cols)} categorical columns: {categorical_cols}")
    else:
        encoded_df = treated.copy()
        
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]
    feature_names = X.columns.tolist()
    
    # 4. Stratified Split: Train (70%), Temp (30%) -> Val (15%), Test (15%)
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / temp_size  # 0.15 / 0.30 = 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio), random_state=random_state, stratify=y_temp
    )
    
    print(f"[Preprocessing] Split shapes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    print(f"[Preprocessing] Feature count after encoding: {X_train.shape[1]}")
    
    # 5. Standardize Numerical Features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_names, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    
    # 6. Save splits
    train_df = pd.concat([X_train_scaled, y_train], axis=1)
    val_df = pd.concat([X_val_scaled, y_val], axis=1)
    test_df = pd.concat([X_test_scaled, y_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"[Preprocessing] Saved train.csv, val.csv, test.csv to {output_dir}")
    
    return {
        "cleaned_df": cleaned,
        "treated_df": treated,
        "outlier_stats": outlier_stats,
        "feature_names": feature_names,
        "scaler": scaler,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test
    }


if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    splits = preprocess_and_split(df)
