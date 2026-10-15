"""
generate_report_pdf.py
Publication-quality academic Project Report PDF and Markdown generator
conforming to U21ADP05 specifications for student Nethra (Roll No: 24AD073).
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp total page count, header, and footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#6c757d"))
        
        # Header on pages > 1
        if self._pageNumber > 1:
            self.drawString(40, 762, "KPRIET | Dept. of AI & DS — U21ADP05: EDA & Visualization")
            self.drawRightString(572, 762, "Case Study: Employee Attrition & Deep Learning")
            self.setStrokeColor(colors.HexColor("#ced4da"))
            self.setLineWidth(0.5)
            self.line(40, 756, 572, 756)
            
        # Running Footer on all pages
        self.setStrokeColor(colors.HexColor("#ced4da"))
        self.setLineWidth(0.5)
        self.line(40, 34, 572, 34)
        self.drawString(40, 23, "Student: Nethra (Roll No: 24AD073) | GitHub: nethradevi924-sys/eda_project")
        self.drawRightString(572, 23, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_markdown_report(eda_catalog: list[dict], eval_catalog: list[dict], metrics: dict, tuning_df: pd.DataFrame) -> str:
    """Builds a comprehensive academic project report in Markdown format."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    md_content = f"""# Case Study: Exploratory Data Analysis, Visualization, and Deep Learning of a Real-World Dataset
**Course Code:** U21ADP05 – Exploratory Data Analysis and Visualization  
**Academic Year:** 2026 – 2027 (ODD SEM) | **Batch:** 2023–2027 | **Year / Sem:** V  
**Institution:** Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology  
**Date of Submission:** 15/10/2026  
**Student Name:** Nethra | **Roll No:** 24AD073  

---

## 1. Title Page Information
- **Project Title:** Case Study: Exploratory Data Analysis, Statistical Distribution Exploration, and Deep Learning Risk Modeling of Employee Attrition
- **Student Name:** Nethra
- **Roll Number / Register ID:** 24AD073
- **Degree & Branch:** B.Tech. Artificial Intelligence and Data Science
- **Semester / Year:** Semester V / Year III (Batch 2023–2027)
- **Institution:** KPR Institute of Engineering and Technology, Coimbatore
- **Course Title & Code:** Exploratory Data Analysis and Visualization (U21ADP05)
- **Submission Date:** October 15, 2026
- **Public GitHub Repository:** https://github.com/nethradevi924-sys/eda_project

---

## 2. Abstract
Employee attrition presents a major operational and financial vulnerability for contemporary enterprises, often incurring significant replacement and retraining expenses. This case study applies an end-to-end framework encompassing exploratory data analysis (EDA), rigorous statistical distribution modeling, and deep learning classification to the benchmark IBM HR Analytics dataset (1,470 records, 35 features). Guided by Ben Fry's 7 Stages of Data Visualization, we explore continuous probability distributions (Log-Normal vs. Gaussian), discrete count mechanics (Poisson modeling), and multivariate interrelationships. A Deep Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout regularization was engineered, optimized with class-weighted binary cross-entropy, and trained to predict individual employee turnover risk. The neural network achieved an overall accuracy of {metrics['accuracy']*100:.1f}%, a precision of {metrics['precision']*100:.1f}%, and an ROC-AUC of {metrics['roc_auc']:.3f}. Permutation importance isolates overtime mandates, career experience, and monthly income as paramount turnover catalysts.

---

## 3. Introduction & Objectives
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
1. **Data Sanitization:** Removed uninformative constants (`EmployeeCount`, `StandardHours`, `Over18`) and arbitrary identifier `EmployeeNumber`.
2. **Outlier Winsorization:** Applied IQR boundary capping to extreme upper-tail values in `MonthlyIncome` and `TotalWorkingYears` to stabilize neural gradient propagation without dropping observations.
3. **Categorical Encoding:** Encoded nominal features (`JobRole`, `Department`, `EducationField`, `MaritalStatus`, `OverTime`, `BusinessTravel`), expanding the feature space to 44 input nodes.
4. **Feature Normalization:** Standardized continuous variables using `StandardScaler` fitted strictly on the training partition to prevent data leakage.
5. **Stratified Partitioning:** Partitioned data into 70% Train (1,029), 15% Validation (220), and 15% Test (221) while strictly preserving the 16.1% target attrition proportion.

---

## 6. Data Visualization (Ben Fry's 7 Stages & CO3/CO4 Visual Catalog)
"""
    for idx, vis in enumerate(eda_catalog, start=1):
        md_content += f"""
### Visualization {idx}: {vis['title']}
![{vis['title']}](../{vis['file']})
- **What it shows:** {vis['shows']}
- **Why it was created:** {vis['why']}
- **Insights obtained:** {vis['insights']}
"""

    md_content += f"""
---

## 7. Deep Learning Model Architecture
To predict employee turnover risk from the 44 standardized input features, a Multi-Layer Perceptron (MLP) was constructed in PyTorch.

### Network Architecture:
- **Input Dimension:** 44 features.
- **Hidden Layer 1:** Dense(44 -> 128) + Batch Normalization + LeakyReLU(α=0.1) + Dropout(p=0.3)
- **Hidden Layer 2:** Dense(128 -> 64) + Batch Normalization + LeakyReLU(α=0.1) + Dropout(p=0.2)
- **Hidden Layer 3:** Dense(64 -> 32) + LeakyReLU(α=0.1) + Dropout(p=0.1)
- **Output Layer:** Dense(32 -> 1) + Sigmoid Activation ($P(\\text{{Attrition}} = 1)$)
- **Total Trainable Parameters:** 16,513 parameters.

### Training Parameters & Optimization:
- **Loss Function:** Binary Cross-Entropy Loss with positive class imbalance weighting ($w \\approx 5.2$).
- **Optimizer:** AdamW (Initial Learning Rate = 0.001, Weight Decay = $1 \\times 10^{{-4}}$).
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5 epochs).
- **Early Stopping:** Patience of 15 epochs monitoring validation loss to prevent overfitting.

### Hyperparameter Tuning Benchmark:
| Configuration | Learning Rate | Batch Size | Dropout Rates | Best Val Loss | Best Val Acc (%) |
|---|---|---|---|---|---|
"""
    for _, row in tuning_df.iterrows():
        md_content += f"| {row['Configuration']} | {row['Learning Rate']} | {row['Batch Size']} | {row['Dropout Rates']} | {row['Best Val Loss']} | {row['Best Val Accuracy (%)']}% |\n"

    md_content += f"""
---

## 8. Result Visualization & Interpretation
"""
    for idx, vis in enumerate(eval_catalog, start=8):
        md_content += f"""
### Visualization {idx}: {vis['title']}
![{vis['title']}](../{vis['file']})
- **What it shows:** {vis['shows']}
- **Why it was created:** {vis['why']}
- **Insights obtained:** {vis['insights']}
"""

    md_content += f"""
### Quantitative Test Set Performance:
- **Overall Accuracy:** {metrics['accuracy']*100:.2f}%
- **Precision:** {metrics['precision']*100:.2f}%
- **Recall (Sensitivity):** {metrics['recall']*100:.2f}%
- **F1-Score:** {metrics['f1']*100:.2f}%
- **ROC-AUC Score:** {metrics['roc_auc']:.4f}
- **Confusion Matrix Breakdown:**
  - True Negatives (Retained correctly identified): **{tn}**
  - False Positives (Retained falsely flagged): **{fp}**
  - False Negatives (Attrited missed): **{fn}**
  - True Positives (Attrited correctly identified): **{tp}**

---

## 9. Conclusion & Future Scope
- **Overtime as the Dominant Risk Vector:** Requiring overtime increases turnover rate from 10.4% to 30.5% (a 3x jump). The deep learning model ranks overtime as the single most critical predictor.
- **Compensation & Experience Disconnect:** Employees departing the organization exhibit stagnant earnings relative to cumulative experience, clustering below the $5,000/month threshold even with 8–12 years of overall career tenure.
- **Tenure with Current Manager:** Employees in their first 1–2 years under a manager face significantly higher turnover hazard, pointing to managerial onboarding friction.
- **Model Performance:** The deep MLP architecture with Batch Normalization achieved strong convergence without memorizing the majority class, registering an ROC-AUC of {metrics['roc_auc']:.3f}.

---

## 10. Literature Review & References (APA / IEEE Style)
1. **Fry, B.** (2008). *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. O'Reilly Media. ISBN: 978-0596514556.
2. **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley Publishing Company.
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. http://www.deeplearningbook.org
4. **Alao, D., & Adeyemo, A. B.** (2013). "Analyzing employee attrition using decision tree algorithms." *Computing, Information Systems, Development Informatics & Allied Research Journal*, 4(1), 17–28.
5. **Punnoose, R., & Ajit, P.** (2016). "Prediction of employee turnover in organizations using machine learning algorithms." *International Journal of Advanced Research in Artificial Intelligence (IJARAI)*, 5(9), 22–26. https://doi.org/10.14569/IJARAI.2016.050904.

---

## 11. Appendix (Code Section & Repository Structure)
GitHub Repository: https://github.com/nethradevi924-sys/eda_project
"""
    return md_content


def generate_pdf_report(eda_catalog: list[dict], eval_catalog: list[dict], metrics: dict, tuning_df: pd.DataFrame, output_path: str = "reports/Project_Report.pdf"):
    """Compiles a tight, publication-grade academic PDF report without large white spaces."""
    print(f"[PDF Generator] Compiling academic PDF report to '{output_path}'...")
    
    # 36 pt (0.5 inch) margins for professional publication density
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36, rightMargin=36,
        topMargin=42, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Clean, professional typography styles
    title_style = ParagraphStyle("RepTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor("#1d3557"), alignment=1, spaceAfter=8)
    h1_style = ParagraphStyle("RepH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=colors.HexColor("#1d3557"), spaceBefore=10, spaceAfter=4, keepWithNext=True)
    h2_style = ParagraphStyle("RepH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13.5, textColor=colors.HexColor("#e63946"), spaceBefore=6, spaceAfter=2, keepWithNext=True)
    body_style = ParagraphStyle("RepBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#2b2d42"), spaceBefore=2, spaceAfter=3)
    bullet_style = ParagraphStyle("RepBullet", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#2b2d42"), leftIndent=10, spaceBefore=1, spaceAfter=1)
    meta_box_style = ParagraphStyle("RepMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#1d3557"))
    
    story = []
    
    # ==================== HEADER / TITLE BLOCK ====================
    story.append(Paragraph("Department of Artificial Intelligence and Data Science", ParagraphStyle("DepHead", fontName="Helvetica", fontSize=11, leading=14, textColor=colors.HexColor("#1d3557"), alignment=1)))
    story.append(Paragraph("KPR Institute of Engineering and Technology, Coimbatore", ParagraphStyle("InstHead", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#e63946"), alignment=1)))
    story.append(Spacer(1, 4))
    story.append(Paragraph("COURSE ASSIGNMENT II — ACADEMIC YEAR 2026 – 2027 (ODD SEM) | BATCH: 2023–2027", ParagraphStyle("AssignHead", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor("#457b9d"), alignment=1)))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1d3557"), spaceAfter=8))
    
    story.append(Paragraph("Case Study: Exploratory Data Analysis, Statistical Distribution Exploration, and Deep Learning Risk Modeling of Employee Attrition", title_style))
    
    # Metadata Table (Without Course Coordinator or Total Marks rows)
    meta_data = [
        [Paragraph("<b>Course Code & Title:</b>", meta_box_style), Paragraph("U21ADP05 – Exploratory Data Analysis and Visualization", meta_box_style),
         Paragraph("<b>Student Name:</b>", meta_box_style), Paragraph("<b>Nethra</b>", meta_box_style)],
        [Paragraph("<b>Degree & Branch:</b>", meta_box_style), Paragraph("B.Tech. Artificial Intelligence and Data Science", meta_box_style),
         Paragraph("<b>Roll / Register No:</b>", meta_box_style), Paragraph("<b>24AD073</b>", meta_box_style)],
        [Paragraph("<b>GitHub Repository:</b>", meta_box_style), Paragraph("<font color='#0066cc'><u>https://github.com/nethradevi924-sys/eda_project</u></font>", meta_box_style),
         Paragraph("<b>Submission Date:</b>", meta_box_style), Paragraph("15/10/2026", meta_box_style)],
    ]
    meta_table = Table(meta_data, colWidths=[120, 180, 100, 140])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#ced4da")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#e9ecef")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))
    
    # Abstract callout box
    abstract_text = (
        "<b>Abstract—</b> Employee attrition represents an acute organizational vulnerability causing severe financial loss and operational disruption. "
        "This case study presents a rigorous end-to-end framework combining Exploratory Data Analysis (EDA), parametric distribution modeling, "
        "and deep learning classification on the benchmark IBM HR Analytics dataset (1,470 records, 35 features). Structured under Ben Fry's 7 Stages "
        "of Data Visualization, we examine continuous probability density (Log-Normal vs. Gaussian), discrete Poisson mechanics, and multivariate correlation structures. "
        f"A Deep Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout was trained to forecast turnover risk, yielding {metrics['accuracy']*100:.1f}% accuracy, "
        f"{metrics['precision']*100:.1f}% precision, and an ROC-AUC of {metrics['roc_auc']:.3f}. Permutation importance identifies overtime demands, total career tenure, "
        "and compensation ceilings as primary catalysts driving attrition."
    )
    abstract_table = Table([[Paragraph(abstract_text, body_style)]], colWidths=[540])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eef4f8")),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor("#b8d0e0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 6))
    
    # ==================== BODY SECTIONS ====================
    # Section 3: Introduction & Objectives
    story.append(Paragraph("3. Introduction & Objectives", h1_style))
    story.append(Paragraph(
        "Talent retention is critical for organizational competitiveness. When experienced personnel depart, organizations incur steep replacement costs, "
        "institutional memory erosion, and project disruption. This case study applies exploratory data analysis, statistical distribution fitting, and deep learning "
        "to diagnose turnover risk factors and deploy early warning models.",
        body_style
    ))
    story.append(Paragraph("• <b>Ben Fry's 7 Stages:</b> Structured execution across Acquire, Parse, Filter, Mine, Represent, Refine, and Interact.", bullet_style))
    story.append(Paragraph("• <b>Distribution Exploration (CO4):</b> Fit continuous (Gaussian vs Log-Normal) and discrete (Poisson) probability distributions.", bullet_style))
    story.append(Paragraph("• <b>Multivariate Diagnostics:</b> Evaluate multicollinearity and hierarchical relationship clustering.", bullet_style))
    story.append(Paragraph("• <b>Deep Learning (Step 4):</b> Design and tune a deep PyTorch Multi-Layer Perceptron with Batch Normalization.", bullet_style))
    story.append(Spacer(1, 4))
    
    # Section 4: Dataset Description
    story.append(Paragraph("4. Dataset Description & Structural Properties", h1_style))
    story.append(Paragraph(
        "The project utilizes the benchmark <b>IBM HR Analytics Employee Attrition & Performance</b> dataset comprising 1,470 records across 35 features. "
        "The features include continuous metrics (Monthly Income, Age, Tenure), discrete event counts (Promotions, Previous Employers), ordinal ratings (Work-Life Balance, Satisfaction), "
        "and nominal attributes (Job Role, Department, Overtime). The target variable is <i>Attrition</i> ('Yes': 16.1%, 'No': 83.9%), presenting a real-world imbalanced classification domain.",
        body_style
    ))
    
    desc_data = [
        [Paragraph("<b>Metric / Dimension</b>", meta_box_style), Paragraph("<b>Specification / Value</b>", meta_box_style),
         Paragraph("<b>Metric / Dimension</b>", meta_box_style), Paragraph("<b>Specification / Value</b>", meta_box_style)],
        [Paragraph("Total Observations", meta_box_style), Paragraph("1,470 Employees", meta_box_style),
         Paragraph("Total Original Features", meta_box_style), Paragraph("35 Features (≥ 15 required)", meta_box_style)],
        [Paragraph("Target Class Balance", meta_box_style), Paragraph("Retained: 1,233 (83.9%) | Attrited: 237 (16.1%)", meta_box_style),
         Paragraph("Missing / Null Values", meta_box_style), Paragraph("0 (100% complete dataset)", meta_box_style)],
        [Paragraph("Zero-Variance Columns", meta_box_style), Paragraph("EmployeeCount, StandardHours, Over18 (Removed)", meta_box_style),
         Paragraph("Processed Features", meta_box_style), Paragraph("44 Input Features after One-Hot Encoding", meta_box_style)],
    ]
    t_desc = Table(desc_data, colWidths=[120, 150, 120, 150])
    t_desc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#ced4da")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#e9ecef")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_desc)
    story.append(Spacer(1, 4))
    
    # Section 5: Preprocessing
    story.append(Paragraph("5. EDA & Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "<b>1. Sanitization:</b> Removed constant features (<i>EmployeeCount, StandardHours, Over18</i>) and uninformative identifier <i>EmployeeNumber</i>.<br/>"
        "<b>2. Outlier Treatment:</b> Applied IQR boundary capping (Winsorization) to extreme upper tails of <i>MonthlyIncome</i> and <i>TotalWorkingYears</i> to stabilize gradients.<br/>"
        "<b>3. Categorical Encoding:</b> One-Hot Encoded nominal attributes (<i>JobRole, Department, EducationField, MaritalStatus, OverTime, BusinessTravel</i>) to 44 features.<br/>"
        "<b>4. Normalization:</b> Scaled features using <code>StandardScaler</code> fitted strictly on training data to prevent leakage.<br/>"
        "<b>5. Stratified Partitioning:</b> Partitioned data into 70% Train (1,029), 15% Validation (220), and 15% Test (221) while preserving class proportions.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    # Section 6: Data Visualization
    story.append(Paragraph("6. Exploratory Data Visualization & Statistical Findings (Ben Fry's 7 Stages & CO3/CO4)", h1_style))
    story.append(Paragraph(
        "Each visual below incorporates Ben Fry's 7 Stages (Acquire, Parse, Filter, Mine, Represent, Refine, Interact) with complete interpretations:",
        body_style
    ))
    
    # Loop over EDA visuals with KeepTogether to prevent large gaps
    for idx, vis in enumerate(eda_catalog, start=1):
        fig_elements = []
        fig_elements.append(Paragraph(f"<b>Figure {idx}: {vis['title']}</b>", h2_style))
        if os.path.exists(vis["file"]):
            fig_elements.append(Image(vis["file"], width=5.8 * inch, height=2.1 * inch))
        fig_elements.append(Paragraph(f"<b>• What it shows:</b> {vis['shows']}", body_style))
        fig_elements.append(Paragraph(f"<b>• Why created:</b> {vis['why']}", body_style))
        fig_elements.append(Paragraph(f"<b>• Insights obtained:</b> {vis['insights']}", body_style))
        fig_elements.append(Spacer(1, 4))
        story.append(KeepTogether(fig_elements))
        
    # Section 7: Deep Learning Model Architecture
    story.append(Paragraph("7. Deep Learning Model Architecture & Hyperparameter Tuning", h1_style))
    story.append(Paragraph(
        "A Deep Multi-Layer Perceptron (MLP) was constructed in PyTorch. The architecture uses Batch Normalization after linear transformations "
        "to prevent internal covariate shift, paired with Dropout regularization and LeakyReLU activations to prevent neuron saturation on tabular inputs.",
        body_style
    ))
    
    arch_data = [
        [Paragraph("<b>Layer</b>", meta_box_style), Paragraph("<b>Components & Dimensions</b>", meta_box_style), Paragraph("<b>Regularization & Activation</b>", meta_box_style)],
        [Paragraph("Input Layer", meta_box_style), Paragraph("44 Standardized Features", meta_box_style), Paragraph("StandardScaler Normalization", meta_box_style)],
        [Paragraph("Hidden Layer 1", meta_box_style), Paragraph("Linear(44 -> 128)", meta_box_style), Paragraph("BatchNorm1d + LeakyReLU(0.1) + Dropout(0.3)", meta_box_style)],
        [Paragraph("Hidden Layer 2", meta_box_style), Paragraph("Linear(128 -> 64)", meta_box_style), Paragraph("BatchNorm1d + LeakyReLU(0.1) + Dropout(0.2)", meta_box_style)],
        [Paragraph("Hidden Layer 3", meta_box_style), Paragraph("Linear(64 -> 32)", meta_box_style), Paragraph("LeakyReLU(0.1) + Dropout(0.1)", meta_box_style)],
        [Paragraph("Output Layer", meta_box_style), Paragraph("Linear(32 -> 1)", meta_box_style), Paragraph("Sigmoid Activation ($P(\\text{Attrition}=1)$)", meta_box_style)],
    ]
    t_arch = Table(arch_data, colWidths=[100, 200, 240])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#ced4da")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#e9ecef")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Hyperparameter Tuning Benchmark Table:</b>", body_style))
    tune_table_data = [[
        Paragraph("<b>Config</b>", meta_box_style),
        Paragraph("<b>Learning Rate</b>", meta_box_style),
        Paragraph("<b>Batch Size</b>", meta_box_style),
        Paragraph("<b>Dropout Rates</b>", meta_box_style),
        Paragraph("<b>Best Val Loss</b>", meta_box_style),
        Paragraph("<b>Best Val Acc (%)</b>", meta_box_style)
    ]]
    for _, row in tuning_df.iterrows():
        tune_table_data.append([
            Paragraph(str(row["Configuration"]), meta_box_style),
            Paragraph(str(row["Learning Rate"]), meta_box_style),
            Paragraph(str(row["Batch Size"]), meta_box_style),
            Paragraph(str(row["Dropout Rates"]), meta_box_style),
            Paragraph(str(row["Best Val Loss"]), meta_box_style),
            Paragraph(f"{row['Best Val Accuracy (%)']}%", meta_box_style),
        ])
    t_tune = Table(tune_table_data, colWidths=[130, 70, 60, 140, 70, 70])
    t_tune.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#ced4da")),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#e9ecef")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_tune)
    story.append(Spacer(1, 6))
    
    # Section 8: Model Evaluation
    story.append(Paragraph("8. Result Visualization & Performance Interpretation", h1_style))
    for idx, vis in enumerate(eval_catalog, start=8):
        eval_elements = []
        eval_elements.append(Paragraph(f"<b>Figure {idx}: {vis['title']}</b>", h2_style))
        if os.path.exists(vis["file"]):
            img_h = 2.0 * inch if idx != 9 else 1.9 * inch
            eval_elements.append(Image(vis["file"], width=5.6 * inch, height=img_h))
        eval_elements.append(Paragraph(f"<b>• What it shows:</b> {vis['shows']}", body_style))
        eval_elements.append(Paragraph(f"<b>• Why created:</b> {vis['why']}", body_style))
        eval_elements.append(Paragraph(f"<b>• Insights obtained:</b> {vis['insights']}", body_style))
        eval_elements.append(Spacer(1, 4))
        story.append(KeepTogether(eval_elements))
        
    # Section 9: Conclusion & Future Scope
    conc_elements = []
    conc_elements.append(Paragraph("9. Conclusion & Future Scope", h1_style))
    conc_elements.append(Paragraph(
        "<b>Summary of Findings:</b> This investigation establishes that workforce turnover is driven by identifiable operational catalysts. "
        "Mandatory overtime exerts the largest single multiplier effect (30.5% attrition vs 10.4% without overtime). Young tenure employees and entry-level sales roles "
        f"experience the highest attrition hazard. The PyTorch Multi-Layer Perceptron delivered {metrics['accuracy']*100:.1f}% accuracy and {metrics['roc_auc']:.3f} ROC-AUC, "
        "while permutation feature importance confirms overtime mandates, tenure, and monthly income as dominant predictors.<br/><br/>"
        "<b>Challenges Encountered:</b> Addressed severe class imbalance (16.1% positive class) via class-weighted binary cross-entropy loss, and handled multicollinear career tenure features.<br/><br/>"
        "<b>Future Enhancements:</b><br/>"
        "• Integrate Explainable AI (XAI) frameworks (SHAP and LIME) for individualized employee risk scorecards.<br/>"
        "• Benchmark tabular transformer architectures (TabNet, FT-Transformer) on broader datasets.<br/>"
        "• Formulate continuous-time survival hazard modeling using DeepSurv to project expected tenure intervals.",
        body_style
    ))
    conc_elements.append(Spacer(1, 4))
    story.append(KeepTogether(conc_elements))
    
    # Section 10: References
    ref_elements = []
    ref_elements.append(Paragraph("10. Literature Review & References (APA / IEEE Style)", h1_style))
    refs = [
        "1. Fry, B. (2008). <i>Visualizing Data: Exploring and Explaining Data with the Processing Environment</i>. O'Reilly Media. ISBN: 978-0596514556.",
        "2. Tukey, J. W. (1977). <i>Exploratory Data Analysis</i>. Addison-Wesley Publishing Company.",
        "3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). <i>Deep Learning</i>. MIT Press. http://www.deeplearningbook.org",
        "4. Alao, D., & Adeyemo, A. B. (2013). 'Analyzing employee attrition using decision tree algorithms.' <i>Computing, Information Systems, Development Informatics & Allied Research Journal</i>, 4(1), 17–28.",
        "5. Punnoose, R., & Ajit, P. (2016). 'Prediction of employee turnover in organizations using machine learning algorithms.' <i>International Journal of Advanced Research in Artificial Intelligence (IJARAI)</i>, 5(9), 22–26. https://doi.org/10.14569/IJARAI.2016.050904."
    ]
    for r in refs:
        ref_elements.append(Paragraph(r, bullet_style))
    ref_elements.append(Spacer(1, 4))
    story.append(KeepTogether(ref_elements))
    
    # Section 11: Appendix
    app_elements = []
    app_elements.append(Paragraph("11. Appendix: Modular Codebase & Repository Structure", h1_style))
    app_elements.append(Paragraph(
        "The project is structured into self-contained, reproducible Python modules:<br/>"
        "• <code>src/data_loader.py</code>: Data ingestion and integrity auditing.<br/>"
        "• <code>src/preprocessing.py</code>: IQR Winsorization, encoding, and stratified splitting.<br/>"
        "• <code>src/visualizations.py</code>: Ben Fry 7 stages & CO3/CO4 statistical visualization suite.<br/>"
        "• <code>src/model.py</code>: PyTorch AttritionMLP deep architecture.<br/>"
        "• <code>src/train.py</code>: Weighted BCE training loop, early stopping, and hyperparameter tuning.<br/>"
        "• <code>src/evaluate.py</code>: Test set metrics, confusion matrix, ROC-AUC, and permutation importance.<br/>"
        "• <code>src/generate_report_pdf.py</code>: Academic report compiler.<br/>"
        "• <code>notebooks/eda_and_deep_learning.ipynb</code>: Executed interactive Jupyter Notebook.<br/><br/>"
        "<b>Public GitHub Repository:</b> <font color='#0066cc'><u>https://github.com/nethradevi924-sys/eda_project</u></font>",
        body_style
    ))
    story.append(KeepTogether(app_elements))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Generator] Successfully generated professional compact PDF report at '{output_path}'.")


def generate_full_reports(eda_catalog: list[dict], eval_catalog: list[dict], metrics: dict, tuning_df: pd.DataFrame):
    """Generates both Markdown and PDF versions of the Project Report."""
    md_content = build_markdown_report(eda_catalog, eval_catalog, metrics, tuning_df)
    md_path = os.path.join(REPORT_DIR, "Project_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Report] Markdown project report saved to '{md_path}'.")
    
    pdf_path = os.path.join(REPORT_DIR, "Project_Report.pdf")
    generate_pdf_report(eda_catalog, eval_catalog, metrics, tuning_df, pdf_path)
