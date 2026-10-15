"""
main.py
End-to-End Orchestrator for the U21ADP05 EDA, Visualization, and Deep Learning Case Study.
Runs the complete data pipeline from ingestion to report generation.
"""

import os
import sys

# Ensure src/ is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_raw_data, audit_dataset, print_audit_summary
from preprocessing import preprocess_and_split
from visualizations import generate_all_eda_visualizations
from train import train_attrition_model, perform_hyperparameter_tuning
from evaluate import evaluate_model_on_test, generate_all_evaluation_visualizations
from generate_report_pdf import generate_full_reports


def run_pipeline():
    print("=" * 70)
    print("   U21ADP05: EXPLORATORY DATA ANALYSIS, VISUALIZATION & DEEP LEARNING")
    print("   Case Study: IBM HR Analytics Employee Attrition & Retention Risk")
    print("=" * 70)
    
    # 1. Ingestion & Audit
    print("\n>>> STEP 1: Data Acquisition & Integrity Auditing...")
    raw_df = load_raw_data("data/raw/employee_attrition.csv")
    audit = audit_dataset(raw_df)
    print_audit_summary(audit)
    
    # 2. Preprocessing & Splitting
    print("\n>>> STEP 2: Preprocessing, Outlier Treatment & Stratified Splitting...")
    splits = preprocess_and_split(raw_df)
    
    # 3. Exploratory Data Analysis & Visualizations (CO3 & CO4)
    print("\n>>> STEP 3: Ben Fry's 7 Stages & Statistical Distribution Visualizations...")
    eda_catalog = generate_all_eda_visualizations(splits["cleaned_df"])
    
    # 4. Hyperparameter Tuning Exploration (Step 4 requirement)
    print("\n>>> STEP 4: Deep Learning Hyperparameter Exploration...")
    tuning_df = perform_hyperparameter_tuning(
        splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]
    )
    
    # 5. Deep Learning Model Training
    print("\n>>> STEP 5: Training Multi-Layer Perceptron (AttritionMLP)...")
    model, history = train_attrition_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        epochs=80, batch_size=32, lr=0.001
    )
    
    # 6. Evaluation on Unseen Test Partition
    print("\n>>> STEP 6: Evaluating Model Performance on Unseen Test Partition...")
    metrics = evaluate_model_on_test(model, splits["X_test"], splits["y_test"])
    
    # 7. Model Evaluation Visualizations (Step 5 requirement)
    print("\n>>> STEP 7: Generating Model Evaluation Visualizations...")
    eval_catalog = generate_all_evaluation_visualizations(
        model, history, metrics, splits["X_val"], splits["y_val"]
    )
    
    # 8. Report Generation (Markdown & PDF)
    print("\n>>> STEP 8: Compiling Academic Project Report (PDF & Markdown)...")
    generate_full_reports(eda_catalog, eval_catalog, metrics, tuning_df)
    
    print("\n" + "=" * 70)
    print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"   • Figures saved:        outputs/figures/ (11 figures)")
    print(f"   • Model weights:        outputs/models/best_attrition_mlp.pt")
    print(f"   • Processed datasets:   data/processed/ (train, val, test)")
    print(f"   • Academic PDF Report:  reports/Project_Report.pdf")
    print(f"   • Markdown Report:      reports/Project_Report.md")
    print("=" * 70)
    return {
        "metrics": metrics,
        "eda_catalog": eda_catalog,
        "eval_catalog": eval_catalog,
        "tuning_df": tuning_df
    }


if __name__ == "__main__":
    run_pipeline()
