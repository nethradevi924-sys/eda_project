"""
data_loader.py
Module for ingesting and validating the IBM HR Employee Attrition dataset.
Part of the U21ADP05 Exploratory Data Analysis & Visualization Project.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str = "data/raw/employee_attrition.csv") -> pd.DataFrame:
    """
    Loads raw CSV dataset and performs initial structural validations.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")

    df = pd.read_csv(filepath)
    print(f"[DataLoader] Successfully loaded dataset from {filepath}")
    print(f"[DataLoader] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """
    Performs comprehensive data auditing:
    - Missing values count & percentage
    - Duplicate rows
    - Data types
    - Constant / zero-variance columns
    - Unique values per column
    """
    audit_results = {}
    
    # Missing values
    missing = df.isnull().sum()
    audit_results["missing_values"] = missing[missing > 0].to_dict()
    audit_results["total_missing"] = int(missing.sum())
    
    # Duplicates
    audit_results["duplicate_rows"] = int(df.duplicated().sum())
    
    # Constant columns (zero-variance)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    audit_results["constant_columns"] = constant_cols
    
    # Data type summary
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    audit_results["numerical_columns"] = numerical_cols
    audit_results["categorical_columns"] = categorical_cols
    audit_results["num_features_count"] = len(df.columns)
    
    # Target distribution
    if "Attrition" in df.columns:
        target_counts = df["Attrition"].value_counts().to_dict()
        target_pct = (df["Attrition"].value_counts(normalize=True) * 100).round(2).to_dict()
        audit_results["target_distribution"] = target_counts
        audit_results["target_percentage"] = target_pct
        
    return audit_results


def print_audit_summary(audit: dict):
    """Prints a structured summary of data audit results."""
    print("=" * 60)
    print("           DATA AUDIT & INTEGRITY SUMMARY")
    print("=" * 60)
    print(f"Total Features Analyzed:   {audit['num_features_count']}")
    print(f"Total Missing Values:      {audit['total_missing']}")
    print(f"Duplicate Rows Detected:   {audit['duplicate_rows']}")
    print(f"Zero-Variance Features:    {audit['constant_columns']}")
    print(f"Numerical Features Count:  {len(audit['numerical_columns'])}")
    print(f"Categorical Features Count:{len(audit['categorical_columns'])}")
    if "target_distribution" in audit:
        print(f"Target Distribution:       {audit['target_distribution']}")
        print(f"Target Proportions (%):    {audit['target_percentage']}")
    print("=" * 60)


if __name__ == "__main__":
    df = load_raw_data()
    audit = audit_dataset(df)
    print_audit_summary(audit)
