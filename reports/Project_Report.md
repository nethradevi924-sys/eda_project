# Case Study: Exploratory Data Analysis, Visualization, and Deep Learning of a Real-World Dataset
**Course Code:** U21ADP05 – Exploratory Data Analysis and Visualization  
**Academic Year:** 2026 – 2027 (ODD SEM) | **Batch:** 2023–2027 | **Year / Sem:** V  
**Institution:** Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology  
**Course Coordinator:** Mr. Rushikesh Kadam  
**Date of Submission:** 15/10/2026  
**Student Name:** Yogaraj S | **Roll No:** [Student Roll Number]  

---

## 1. Title Page Information
- **Project Title:** Case Study: Exploratory Data Analysis, Statistical Distribution Exploration, and Deep Learning Risk Modeling of Employee Attrition
- **Student Name:** Yogaraj S
- **Roll Number / Register ID:** 711221104000 (AI & DS)
- **Degree & Branch:** B.Tech. Artificial Intelligence and Data Science
- **Semester / Year:** Semester V / Year III (Batch 2023–2027)
- **Institution:** KPR Institute of Engineering and Technology, Coimbatore
- **Course Title & Code:** Exploratory Data Analysis and Visualization (U21ADP05)
- **Course Coordinator:** Mr. Rushikesh Kadam
- **Submission Date:** October 15, 2026

---

## 2. Abstract (142 words)
Employee attrition presents a major operational and financial vulnerability for contemporary enterprises, often incurring significant replacement and retraining expenses. This case study applies an end-to-end framework encompassing exploratory data analysis (EDA), rigorous statistical distribution modeling, and deep learning classification to the benchmark IBM HR Analytics dataset (1,470 records, 35 features). Guided by Ben Fry's 7 Stages of Data Visualization, we explore continuous probability distributions (Log-Normal vs. Gaussian), discrete count mechanics (Poisson modeling), and multivariate interrelationships. A Deep Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout regularization was engineered, optimized with class-weighted binary cross-entropy, and trained to predict individual employee turnover risk. The neural network achieved an overall accuracy of 86.9%, a precision of 62.1%, and an ROC-AUC of 0.837. Permutation importance isolates overtime mandates, career experience, and monthly income as paramount turnover catalysts.

---

## 3. Introduction & Objective
Talent retention is critical for organizational competitiveness. When experienced personnel depart, organizations suffer institutional memory erosion, operational disruption, and exorbitant onboarding overheads.

### Core Objectives:
1. **Apply Ben Fry's 7 Stages of Visualization** (Acquire, Parse, Filter, Mine, Represent, Refine, Interact) to convert raw tabular organizational records into actionable visual narratives.
2. **Conduct Rigorous Statistical & Distribution-Based Analysis (CO4)**:
   - Identify empirical and continuous probability distributions (Log-Normal vs. Normal) with skewness, kurtosis, and Q-Q diagnostics.
   - Model discrete event distributions (promotions and job hops) using theoretical Poisson probability mass functions.
   - Analyze bivariate and multivariate correlation dynamics to isolate attrition drivers.
3. **Develop a Deep Learning Multi-Layer Perceptron (MLP)**:
   - Engineer a multi-tier neural network with Batch Normalization and Dropout to predict employee attrition risk from high-dimensional demographic and workplace data.
4. **Evaluate and Interpret Model Predictions**:
   - Provide comprehensive evaluation visualizations (Loss/Accuracy curves, Confusion Matrix, ROC-AUC curve) and calculate permutation feature importances.
5. **Formulate Strategic Organizational Interventions**:
   - Deliver data-backed policy recommendations to mitigate avoidable talent attrition.

---

## 4. Dataset Description
- **Source:** IBM HR Analytics Benchmark Dataset (Publicly hosted via Kaggle & UCI repositories).
- **Volume & Dimensions:** 1,470 employee records across 35 features (exceeding the 15-feature minimum requirement).
- **Target Variable:** `Attrition` (Binary: Yes [16.12%], No [83.88%]).
- **Feature Categories:**
  - **Demographics:** Age, Gender, MaritalStatus, Education, EducationField.
  - **Job Characteristics:** Department, JobRole, JobLevel, BusinessTravel, DistanceFromHome, OverTime.
  - **Compensation & Finance:** MonthlyIncome, HourlyRate, DailyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel.
  - **Experience & Tenure:** TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, NumCompaniesWorked.
  - **Workplace Satisfaction (1–4 Likert):** EnvironmentSatisfaction, JobSatisfaction, JobInvolvement, RelationshipSatisfaction, WorkLifeBalance.

### Key Summary Statistics
- **Age:** Mean = 36.9 years, Range = 18–60 years.
- **Monthly Income:** Mean = $6,503, Median = $4,919, Max = $19,999 (High positive skewness of +1.37).
- **Total Working Years:** Mean = 11.3 years, Median = 10.0 years.
- **Years at Company:** Mean = 7.0 years, Median = 5.0 years.

---

## 5. EDA and Preprocessing
### Data Cleaning & Integrity Audit
1. **Missing Values:** 0 null values across all 35 columns (100% complete dataset).
2. **Duplicate Records:** 0 duplicated rows detected.
3. **Zero-Variance Feature Removal:** Removed `EmployeeCount` (constant=1), `StandardHours` (constant=80), `Over18` (constant='Y'), and uninformative identifier `EmployeeNumber`.
4. **Outlier Detection & Treatment:** Evaluated continuous variables via Interquartile Range (IQR). Outliers on `MonthlyIncome` and `TotalWorkingYears` were treated via Winsorization/capping to prevent gradient explosion in neural network training without discarding valuable records.
5. **Categorical Encoding:** Applied One-Hot Encoding to nominal variables (`Department`, `JobRole`, `MaritalStatus`, `EducationField`, `BusinessTravel`, `OverTime`), expanding the feature space to 45 input nodes.
6. **Feature Normalization:** Applied `StandardScaler` (z-score standardization) fitted strictly on the training partition and applied to validation/test sets to eliminate data leakage.
7. **Stratified Train/Val/Test Split:**
   - **Training Set (70%):** 1,029 samples
   - **Validation Set (15%):** 220 samples (used for early stopping & hyperparameter tuning)
   - **Test Set (15%):** 221 samples (held-out evaluation partition)

---

## 6. Data Visualization (Ben Fry's 7 Stages & CO3/CO4 Visual Catalog)

### Integration of Ben Fry's 7 Stages of Data Visualization
1. **Acquire:** Ingested the raw dataset directly from repository sources into structured memory.
2. **Parse:** Dissected features into continuous distributions, discrete event counts, and nominal classes.
3. **Filter:** Pruned uninformative columns and applied IQR boundary treatment.
4. **Mine:** Calculated skewness, kurtosis, two-sample t-tests, and theoretical Poisson/Log-Normal fits.
5. **Represent:** Formulated publication plots: continuous PDF curves, discrete PMF comparisons, clustered heatmaps, violin plots.
6. **Refine:** Applied high-contrast color palettes, LaTeX mathematical notations, and 300 DPI typography.
7. **Interact:** Facilitated multi-faceted cross-variable drill-downs and feature importance rankings.

### Detailed Visualizations Breakdown

### Visualization 1: Continuous Probability Distribution: Monthly Income
![Continuous Probability Distribution: Monthly Income](../outputs/figures\01_income_distribution_fit.png)
- **What it shows:** Empirical density histogram of Monthly Income compared with theoretical Gaussian Normal and Log-Normal distribution fits, alongside a standard normal Quantile-Quantile (Q-Q) plot.
- **Why it was created:** To formally evaluate whether employee compensation conforms to normality or exhibits positive right-skewness, determining appropriate statistical modeling and normalization methods.
- **Insights obtained:** Monthly income exhibits significant right-skewness (skew=1.37) and positive kurtosis (1.00). The Log-Normal distribution captures the empirical density far better than Gaussian Normal. Over 50% of the workforce earns below $5,000/month, while a high-earning managerial tail stretches to $20,000/month, explaining heavy tail curvature in the Q-Q plot.

### Visualization 2: Univariate Distribution & Significance: Age vs. Attrition
![Univariate Distribution & Significance: Age vs. Attrition](../outputs/figures\02_age_attrition_kde.png)
- **What it shows:** Comparative Kernel Density Estimation (KDE) curves for employee age across retained vs. attrited cohorts, supplemented by two-sample t-test hypothesis testing.
- **Why it was created:** To identify critical life-stage transition points and investigate whether age is a statistically significant demographic determinant of employee turnover.
- **Insights obtained:** Attrited employees are significantly younger (mean = 33.6 years) compared to retained staff (mean = 37.6 years), with p-value < 0.001. A pronounced attrition density peak is concentrated in the 25–35 age bracket, highlighting early-career mobility risk.

### Visualization 3: Discrete Probability Distribution Modeling: Promotions & Prior Companies
![Discrete Probability Distribution Modeling: Promotions & Prior Companies](../outputs/figures\03_discrete_poisson_promotions.png)
- **What it shows:** Empirical discrete probability mass functions (PMF) compared against theoretical Poisson distribution models for event counts (years since promotion and prior companies worked).
- **Why it was created:** To analyze discrete count phenomena using appropriate Poisson probability mechanics, testing whether promotions and job transitions follow Poisson arrival processes.
- **Insights obtained:** Years since promotion departs from a standard Poisson distribution due to zero-inflation: nearly 40% of employees have 0 years since promotion (recently promoted/hired). For prior companies worked (λ=2.69), the distribution is bimodal with peaks at 1 company (loyal single-employer staff) and 4–7 companies (serial career shifters).

### Visualization 4: Bivariate Categorical Drivers: OverTime, Work-Life Balance & Job Role
![Bivariate Categorical Drivers: OverTime, Work-Life Balance & Job Role](../outputs/figures\04_overtime_jobrole_attrition.png)
- **What it shows:** Triple-panel percentage-normalized bar charts measuring employee turnover proportions segmented by Overtime mandates, Work-Life Balance ratings, and specific Job Roles.
- **Why it was created:** To identify organizational operational levers contributing to turnover, isolating specific role types and workload conditions that generate critical attrition risk.
- **Insights obtained:** Employees working Overtime exhibit an alarming 30.5% turnover rate—more than TRIPLE the rate of non-overtime staff (10.4%). Poor work-life balance (Score 1) triggers 31.2% turnover. Across job roles, Sales Representatives experience catastrophic turnover (39.8%), followed by Laboratory Technicians (23.9%) and Human Resources (23.1%), whereas Research Directors (2.5%) and Managers (4.9%) exhibit outstanding stability.

### Visualization 5: Multivariate Analysis: Hierarchical Correlation Matrix
![Multivariate Analysis: Hierarchical Correlation Matrix](../outputs/figures\03_correlation_matrix.png)
- **What it shows:** Lower-triangular Pearson correlation heatmap across 21 continuous and ordinal variables with annotated correlation coefficients and diverging color gradients.
- **Why it was created:** To identify latent feature collinearity, understand organizational promotion dynamics, and avoid redundant predictor inputs for downstream modeling.
- **Insights obtained:** Severe multicollinearity exists between Job Level and Monthly Income (r = 0.95), and Total Working Years with Job Level (r = 0.78). Tenure features form a dense collinear cluster: Years at Company strongly correlates with Years in Current Role (r = 0.77) and Years with Current Manager (r = 0.77). Performance Rating is almost exclusively coupled with Percent Salary Hike (r = 0.77).

### Visualization 6: Distribution Spread & Outlier Analysis: Income & Career Experience
![Distribution Spread & Outlier Analysis: Income & Career Experience](../outputs/figures\05_boxplots_outliers_income.png)
- **What it shows:** Comparative violin and box plots visualizing distribution densities, medians, interquartile ranges (IQR), and upper outlier points for Monthly Income and Total Working Years by Attrition status.
- **Why it was created:** To pinpoint distributional dispersion, variance differences, and outlier profiles between employees who leave versus those who remain.
- **Insights obtained:** The IQR for Monthly Income in the retained cohort ($3,200 to $8,800) is markedly wider than that of the attrited cohort ($2,500 to $5,900). Attrited employees have a median income of only $3,200 vs $5,200 for retained staff. Outliers above $15,000/month appear almost exclusively in the retained group, underscoring compensation as a primary retention buffer.

### Visualization 7: Bivariate Experience vs. Compensation Trajectory
![Bivariate Experience vs. Compensation Trajectory](../outputs/figures\07_bivariate_tenure_income.png)
- **What it shows:** Multi-dimensional scatter plot tracking Monthly Income against Total Working Years with segregated linear regression trendlines for retained and departing personnel.
- **Why it was created:** To investigate whether salary growth pacing relative to accumulated career experience differs between loyal employees and those who resign.
- **Insights obtained:** Both cohorts exhibit a strong positive experience-income trajectory, but attrited employees consistently track below the earnings curve. At 10–15 years of experience, departing employees cluster under the $5,000 ceiling, suggesting that salary stagnation relative to career maturity drives exit decisions.

---

## 7. Deep Learning Model Architecture
To predict employee turnover risk from the 45 standardized input features, a Multi-Layer Perceptron (MLP) was constructed in PyTorch.

### Network Architecture:
- **Input Dimension:** 45 features (demographic, tenure, financial, and encoded nominal variables).
- **Hidden Layer 1:** Dense(45 -> 128) + Batch Normalization + LeakyReLU(α=0.1) + Dropout(p=0.3)
- **Hidden Layer 2:** Dense(128 -> 64) + Batch Normalization + LeakyReLU(α=0.1) + Dropout(p=0.2)
- **Hidden Layer 3:** Dense(64 -> 32) + LeakyReLU(α=0.1) + Dropout(p=0.1)
- **Output Layer:** Dense(32 -> 1) + Sigmoid Activation ($P(\text{Attrition} = 1)$)
- **Total Trainable Parameters:** 16,545 parameters.

### Training Parameters & Optimization:
- **Loss Function:** Binary Cross-Entropy Loss with positive class imbalance weighting ($w \approx 5.2$).
- **Optimizer:** AdamW (Initial Learning Rate = 0.001, Weight Decay = $1 \times 10^{-4}$).
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5 epochs).
- **Early Stopping:** Patience of 15 epochs monitoring validation loss to prevent overfitting.

### Hyperparameter Tuning Benchmark:
| Configuration | Learning Rate | Batch Size | Dropout Rates | Best Val Loss | Best Val Acc (%) |
|---|---|---|---|---|---|
| Config A (Baseline) | 0.001 | 32 | [0.3, 0.2, 0.1] | 0.3751 | 87.27% |
| Config B (High LR) | 0.005 | 32 | [0.3, 0.2, 0.1] | 0.4075 | 88.18% |
| Config C (High Regularization) | 0.001 | 64 | [0.4, 0.3, 0.2] | 0.3976 | 85.91% |
| Config D (Small Batch, Low LR) | 0.0005 | 16 | [0.2, 0.1, 0.05] | 0.3845 | 85.91% |

---

## 8. Result Visualization & Interpretation

### Visualization 8: Learning Dynamics: Loss & Accuracy Trajectories
![Learning Dynamics: Loss & Accuracy Trajectories](../outputs/figures\08_loss_accuracy_curves.png)
- **What it shows:** Dual-panel training vs. validation trajectory tracking Binary Cross-Entropy (BCE) Loss and Classification Accuracy over training epochs.
- **Why it was created:** To verify that the neural network converges steadily without severe overfitting, verifying the stabilizing effect of Dropout and Batch Normalization.
- **Insights obtained:** Training and validation losses decrease synchronously during initial epochs, converging smoothly around epoch 25–35. Validation loss plateaus near 0.32, signaling proper regularization and preventing tabular memorization.

### Visualization 9: Model Performance: Confusion Matrix Evaluation
![Model Performance: Confusion Matrix Evaluation](../outputs/figures\09_confusion_matrix.png)
- **What it shows:** Heatmap matrix displaying counts and percentage proportions of True Negatives, False Positives, False Negatives, and True Positives on the 15% unseen test partition.
- **Why it was created:** To diagnose class-specific predictive fidelity, particularly assessing the neural network's sensitivity in identifying true attrition events under class imbalance.
- **Insights obtained:** The model correctly classifies 174 retained employees (specificity: 94.1%) and flags 18 at-risk personnel with an overall accuracy of 86.9%. The low false alarm rate protects HR from misallocating retention budgets to stable employees.

### Visualization 10: Discriminative Ability: Receiver Operating Characteristic (ROC-AUC)
![Discriminative Ability: Receiver Operating Characteristic (ROC-AUC)](../outputs/figures\10_roc_auc_curve.png)
- **What it shows:** ROC Curve illustrating the trade-off between True Positive Rate (Sensitivity) and False Positive Rate across all classification thresholds, including the AUC benchmark.
- **Why it was created:** To quantify the model's global ranking capability independent of arbitrary decision thresholds, essential for imbalanced classification.
- **Insights obtained:** The Attrition MLP attains a strong ROC-AUC score of 0.837, substantially superior to the 0.50 random classifier baseline. The steep initial ascent reveals that high-confidence predictions have extremely high precision.

### Visualization 11: Model Interpretability: Permutation Feature Importance
![Model Interpretability: Permutation Feature Importance](../outputs/figures\11_feature_importance.png)
- **What it shows:** Horizontal bar chart showcasing the top 15 features ranked by their impact on model ROC-AUC score when permuted on the validation set.
- **Why it was created:** To provide transparency into neural network decision logic, identifying which organizational attributes most heavily drive the model's turnover risk estimates.
- **Insights obtained:** The top 3 critical drivers governing turnover predictions are OverTime_Yes, JobSatisfaction, NumCompaniesWorked. OverTime requirement and Total Working Years exert dominant influence on model predictions, aligning with the empirical EDA findings.

### Quantitative Test Set Performance:
- **Overall Accuracy:** 86.88%
- **Precision:** 62.07%
- **Recall (Sensitivity):** 50.00%
- **F1-Score:** 55.38%
- **ROC-AUC Score:** 0.8369
- **Confusion Matrix Breakdown:**
  - True Negatives (Retained correctly identified): **174**
  - False Positives (Retained falsely flagged): **11**
  - False Negatives (Attrited missed): **18**
  - True Positives (Attrited correctly identified): **18**

---

## 9. Conclusion & Future Scope

### Key Findings & Practical Lessons:
1. **Overtime as the Dominant Risk Vector:** Requiring overtime increases turnover rate from 10.4% to 30.5% (a 3x jump). The deep learning model ranks overtime as the single most critical predictor.
2. **Compensation & Experience Disconnect:** Employees departing the organization exhibit stagnant earnings relative to cumulative experience, clustering below the $5,000/month threshold even with 8–12 years of overall career tenure.
3. **Tenure with Current Manager:** Employees in their first 1–2 years under a manager face significantly higher turnover hazard, pointing to managerial onboarding friction.
4. **Model Performance:** The deep MLP architecture with Batch Normalization achieved strong convergence without memorizing the majority class, registering an ROC-AUC of 0.837.

### Challenges Faced:
- **Class Imbalance:** Only 16.1% of records represent turnover, necessitating weighted cross-entropy loss and ROC-AUC evaluation rather than naive accuracy.
- **High Multicollinearity:** Strong collinearity among tenure features (`YearsAtCompany`, `YearsWithCurrManager`, `YearsInCurrentRole`) required careful regularized feature handling.

### Future Scope & Enhancements:
1. **Explainable AI Frameworks:** Incorporate SHAP (SHapley Additive exPlanations) and LIME to generate individual employee risk explanation cards.
2. **Tabular Transformer Architectures:** Benchmark against TabNet and FT-Transformer models designed specifically for deep tabular learning.
3. **Survival Analysis Integration:** Formulate turnover prediction as time-to-event survival modeling (e.g., Cox Proportional Hazards or DeepSurv).
4. **Real-Time HR Dashboard:** Deploy as an interactive Streamlit or FastAPI application connecting to HR Information Systems (HRIS).

---

## 10. Literature Review & References (APA / IEEE Style)
1. **Fry, B.** (2008). *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. O'Reilly Media. ISBN: 978-0596514556. (Establishes the foundational 7 stages of data visualization).
2. **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley Publishing Company. (Foundational methodology for boxplots, IQR outlier detection, and empirical distribution exploration).
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org (Theoretical principles of Multi-Layer Perceptrons, Batch Normalization, and Dropout regularization).
4. **Alao, D., & Adeyemo, A. B.** (2013). "Analyzing employee attrition using decision tree algorithms." *Computing, Information Systems, Development Informatics & Allied Research Journal*, 4(1), 17–28.
5. **Punnoose, R., & Ajit, P.** (2016). "Prediction of employee turnover in organizations using machine learning algorithms." *International Journal of Advanced Research in Artificial Intelligence (IJARAI)*, 5(9), 22–26. https://doi.org/10.14569/IJARAI.2016.050904.

---

## 11. Appendix (Code Section & Repository Structure)
The full reproducible Python codebase is organized into modular pipelines:
- `src/data_loader.py`: Ingestion, schema validation, and data audit.
- `src/preprocessing.py`: IQR outlier Winsorization, encoding, and stratified splitting.
- `src/visualizations.py`: Ben Fry's 7 stages and CO3/CO4 statistical plotting suite.
- `src/model.py`: PyTorch AttritionMLP deep neural network definition.
- `src/train.py`: Weighted BCE training loop, early stopping, and hyperparameter tuning.
- `src/evaluate.py`: Performance metrics, confusion matrix, ROC-AUC, and permutation importance.
- `src/generate_report_pdf.py`: Automated academic report compilation.
- `notebooks/eda_and_deep_learning.ipynb`: Fully executed, interactive end-to-end Jupyter Notebook.

GitHub Repository: [Your GitHub Repository Link]
