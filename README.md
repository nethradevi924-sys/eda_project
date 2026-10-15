# Case Study: Exploratory Data Analysis, Visualization, and Deep Learning of a Real-World Dataset

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course: U21ADP05](https://img.shields.io/badge/Course-U21ADP05-success.svg)](https://kpriet.ac.in)

> **Academic Assignment II: Case Study Report and Codebase**  
> **Institution:** Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology  
> **Course Code & Title:** U21ADP05 – Exploratory Data Analysis and Visualization  
> **Course Coordinator:** Mr. Rushikesh Kadam  
> **Batch:** 2023–2027 (Year III / Sem V) | **Marks:** 40 (CO3: 8M, CO4: 16M, CO5: 16M)  
> **Author / Student:** Nethradevi ([@nethradevi924-sys](https://github.com/nethradevi924-sys))  

---

## Executive Summary
Unplanned employee turnover creates substantial financial liabilities, institutional knowledge leakage, and operational disruption. This project applies an end-to-end data science lifecycle—from **Ben Fry's 7 Stages of Data Visualization** to **deep neural network risk classification**—using the benchmark **IBM HR Analytics Employee Attrition & Performance Dataset** (1,470 records, 35 features).

### Key Empirical Findings:
- **OverTime Requirement as Primary Catalyst:** Employees mandated to work overtime exhibit an attrition rate of **30.5%**, nearly triple that of standard-schedule peers (**10.4%**).
- **Compensation & Experience Asymmetry:** Departing personnel cluster below the **\$5,000/month** threshold even when possessing 8–12 years of industry experience, pointing to salary stagnation as a key exit driver.
- **Tenure Friction:** High turnover hazard is heavily concentrated during the first 1–2 years under a new manager.
- **Deep Learning Model Performance:** A calibrated PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization achieved **86.88% accuracy**, **62.07% precision**, and an **ROC-AUC of 0.8369** on unseen test data.

---

## Ben Fry's 7 Stages of Data Visualization

| Stage | Implementation in this Project |
|---|---|
| **1. Acquire** | Ingested real-world enterprise records from the IBM HR Analytics benchmark (1,470 rows, 35 columns). |
| **2. Parse** | Structured columns into continuous measurements, discrete event counts, and nominal/ordinal categories. |
| **3. Filter** | Pruned zero-variance columns (`EmployeeCount`, `Over18`, `StandardHours`) and applied IQR Winsorization. |
| **4. Mine** | Calculated skewness, kurtosis, two-sample t-tests, Pearson correlations, and fitted parametric distributions. |
| **5. Represent** | Generated continuous PDFs, discrete Poisson PMFs, clustered correlation heatmaps, and violin plots. |
| **6. Refine** | Applied 300 DPI publication styling, contrasting palettes, LaTeX mathematical notations, and data callouts. |
| **7. Interact** | Created multi-dimensional facet plots and permutation feature importance rankings. |

---

## Statistical & Distribution Exploration (CO4 Focus)

### 1. Continuous Distribution Modeling (Monthly Income)
- **Empirical Skewness:** +1.37 (Pronounced positive right-skewness).
- **Parametric Fit:** Fitted Theoretical Gaussian vs. **Log-Normal Distribution**. The Log-Normal model closely tracks empirical density, validating that enterprise compensation follows multiplicative rather than additive dynamics.
- **Q-Q Diagnostics:** Severe departure from the normal line at the upper tail ($>\$15,000/\text{month}$), representing senior managerial compensation tiers.

### 2. Discrete Count Mechanics (Poisson Modeling)
- **Years Since Last Promotion:** Departed from standard Poisson due to extreme **zero-inflation** (~40% at $k=0$).
- **Number of Prior Employers:** Modeled against Poisson ($\lambda = 2.69$), exhibiting bimodal clustering between loyalists ($k=1$) and frequent job-hoppers ($k \ge 5$).

---

## Deep Learning Architecture (AttritionMLP)

```
Input (44 Standardized Features)
  │
  ├──► Linear(44 -> 128) ──► BatchNorm1d ──► LeakyReLU(0.1) ──► Dropout(0.3)
  │
  ├──► Linear(128 -> 64) ──► BatchNorm1d ──► LeakyReLU(0.1) ──► Dropout(0.2)
  │
  ├──► Linear(64 -> 32)  ──► LeakyReLU(0.1) ──► Dropout(0.1)
  │
  └──► Linear(32 -> 1)   ──► Sigmoid Activation (Turnover Risk Probability)
```

- **Loss Function:** Binary Cross-Entropy with positive class weighting ($w \approx 5.20$) to manage the 84:16 class imbalance.
- **Optimizer:** AdamW ($\text{LR} = 0.001$, $\text{weight decay} = 10^{-4}$).
- **Scheduler:** `ReduceLROnPlateau` monitoring validation loss.
- **Early Stopping:** Regularized patience threshold of 15 epochs.

### Test Set Benchmark Metrics:
- **Test Accuracy:** `86.88%`
- **Precision (Positive Class):** `62.07%`
- **Recall (Sensitivity):** `50.00%`
- **F1-Score:** `55.38%`
- **ROC-AUC Score:** `0.8369`

---

## Visualizations Preview

| Visual Index | Chart Description | File Location |
|---|---|---|
| **Figure 1** | Continuous Probability Fit (Normal vs Log-Normal) | `outputs/figures/01_income_distribution_fit.png` |
| **Figure 2** | Age Distribution KDE & Two-Sample t-test | `outputs/figures/02_age_attrition_kde.png` |
| **Figure 3** | Discrete Poisson Modeling (Promotions & Companies) | `outputs/figures/03_discrete_poisson_promotions.png` |
| **Figure 4** | Bivariate Attrition Drivers (OverTime, JobRole, WLB) | `outputs/figures/04_overtime_jobrole_attrition.png` |
| **Figure 5** | Multivariate Pearson Correlation Heatmap | `outputs/figures/03_correlation_matrix.png` |
| **Figure 6** | Outlier & Spread Analysis (Boxplots & Violins) | `outputs/figures/05_boxplots_outliers_income.png` |
| **Figure 7** | Bivariate Experience vs Compensation Trajectory | `outputs/figures/07_bivariate_tenure_income.png` |
| **Figure 8** | Loss & Accuracy Learning Trajectories | `outputs/figures/08_loss_accuracy_curves.png` |
| **Figure 9** | Confusion Matrix Heatmap | `outputs/figures/09_confusion_matrix.png` |
| **Figure 10** | Receiver Operating Characteristic (ROC-AUC) | `outputs/figures/10_roc_auc_curve.png` |
| **Figure 11** | Permutation Feature Importance Ranking | `outputs/figures/11_feature_importance.png` |

---

## Repository Structure

```
EDA_project/
├── README.md                          # Project documentation and portfolio overview
├── requirements.txt                   # Pinned Python package dependencies
├── .gitignore                         # Standard git ignore configuration
├── data/
│   ├── raw/
│   │   └── employee_attrition.csv     # 1,470 records, 35 features
│   └── processed/
│       ├── train.csv                  # Stratified train set (70%)
│       ├── val.csv                    # Stratified validation set (15%)
│       └── test.csv                   # Stratified test set (15%)
├── notebooks/
│   └── eda_and_deep_learning.ipynb    # Fully executed Jupyter Notebook with inline charts
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Ingestion & integrity audit
│   ├── preprocessing.py              # IQR Winsorization, encoding & stratified splits
│   ├── visualizations.py              # Ben Fry's 7 stages & CO3/CO4 statistical charts
│   ├── model.py                       # PyTorch AttritionMLP neural network
│   ├── train.py                       # Training loop, early stopping & hyperparameter grid
│   ├── evaluate.py                    # Test metrics, confusion matrix & ROC-AUC
│   ├── build_notebook.py              # Automated Jupyter notebook compiler
│   ├── generate_report_pdf.py         # Academic ReportLab PDF compiler
│   └── main.py                        # End-to-end execution pipeline
├── outputs/
│   ├── figures/                       # 11 publication-grade 300 DPI visualizations
│   └── models/
│       └── best_attrition_mlp.pt      # Serialized neural network weights
└── reports/
    ├── Project_Report.pdf             # Formatted academic PDF report (11 sections)
    └── Project_Report.md              # Markdown source report
```

---

## Getting Started & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/nethradevi924-sys/eda_project.git
cd eda_project
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline
```bash
python src/main.py
```
This single command executes:
1. Data loading and integrity auditing.
2. IQR outlier Winsorization and stratified splitting.
3. Generation of all 11 high-resolution figures in `outputs/figures/`.
4. Hyperparameter tuning benchmark.
5. PyTorch Deep MLP training and serialization.
6. Test set evaluation.
7. Automated compilation of `reports/Project_Report.pdf`.

### 4. Launch the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/eda_and_deep_learning.ipynb
```

---

## Academic References (APA / IEEE Style)
1. **Fry, B.** (2008). *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. O'Reilly Media. ISBN: 978-0596514556.
2. **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley Publishing Company.
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org
4. **Alao, D., & Adeyemo, A. B.** (2013). "Analyzing employee attrition using decision tree algorithms." *Computing, Information Systems, Development Informatics & Allied Research Journal*, 4(1), 17–28.
5. **Punnoose, R., & Ajit, P.** (2016). "Prediction of employee turnover in organizations using machine learning algorithms." *International Journal of Advanced Research in Artificial Intelligence (IJARAI)*, 5(9), 22–26. https://doi.org/10.14569/IJARAI.2016.050904.

---

## Course Submission Checklist
- [x] **Step 1:** Real-world dataset with >15 features (35 features used).
- [x] **Step 2:** Missing values, outliers, duplicates, encoding, scaling, train/val/test splits.
- [x] **Step 3:** Minimum 5 visuals (7 generated) covering all 7 stages of visualization.
- [x] **Step 4:** Deep Learning model (PyTorch MLP), feature analysis, hyperparameter tuning.
- [x] **Step 5:** Model evaluation visualizations (Loss/Acc, Confusion Matrix, ROC-AUC, Feature Importance).
- [x] **Step 6:** Comprehensive conclusion, findings, challenges, and future scope.
- [x] **Step 7:** 5 academic references in APA/IEEE format.
- [x] **Project Report:** `reports/Project_Report.pdf` (Formatted strictly per page 3 & 4 guidelines).
- [x] **Source Code:** Executable `.ipynb` notebook and modular `.py` pipeline.
- [x] **GitHub Submission:** Complete repository ready for upload and Google Form submission.
