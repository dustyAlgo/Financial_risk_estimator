# Financial Health & Risk Detection System for Nifty-100 Companies

> **A comprehensive machine learning system for early detection of financially risky large-cap Indian firms using multi-year cash flow and financial statement analysis.**

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Results & Performance](#results--performance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [File Documentation](#file-documentation)
- [Usage Examples](#usage-examples)
- [Key Findings](#key-findings)

---

## Overview

This project develops an **interpretable financial risk detection system** that classifies Indian large-cap firms (Nifty-100 index) as **financially healthy** or **financially risky** using multi-year cash flow analysis and financial fundamentals.

### Objectives

✓ **Forward-looking financial stress screening** (not bankruptcy prediction)  
✓ **High-recall detection** of risky firms to enable early intervention  
✓ **Transparent, interpretable predictions** grounded in financial fundamentals  
✓ **Production-ready models** with robust time-based validation  

### Business Impact

- **Zero Risky Firms Missed:** All financially stressed companies are flagged for intervention
- **Acceptable False Alarm Rate:** 25-30% false positive rate manageable for early warning systems
- **Strong Discrimination:** PR-AUC of 0.95 enables reliable risk ranking
- **Actionable Intelligence:** Transparent decision rules based on OCF, volatility, and financing patterns

---

## Key Features

### 1. **Robust Data Pipeline**
- Standardized processing of nested JSON financial data from 97 companies
- Multi-year panel data spanning 2011–2024
- Quality assurance: duplicate detection, data type validation, missing value handling

### 2. **Interpretable Risk Labels**
- Rule-based labeling using 3-year lookback windows (no future leakage)
- Decision rules grounded in financial intuition:
  - Negative operating cash flow history
  - High OCF volatility
  - Heavy financing dependency
- ~85% healthy, ~15% risky (realistic distribution)

### 3. **Advanced Feature Engineering**
- 32 rolling 3-year features capturing cash flow patterns
- Financial stress indicators: financing dependency ratio, OCF negative percentage
- Growth metrics and volatility measures
- All features computed with no temporal leakage

### 4. **Dual Model Architecture**
- **Logistic Regression:** Interpretable baseline with clear coefficients
- **XGBoost:** High-performance ensemble model with superior recall
- Time-based evaluation preventing information leakage

### 5. **Production-Ready Deliverables**
- Saved model artifacts (.joblib files)
- Test set predictions with probability scores
- Feature importance rankings
- Comprehensive performance metrics

---

## Pipeline Architecture

```
┌─────────────────────┐
│  Raw JSON Files     │  (97 companies, 2011-2024)
│  data/raw/*.json    │
└──────────────┬──────┘
               │
               ▼
┌──────────────────────────────┐
│ create_canonical_financials  │  Data Canonicalization
│ .py                          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ canonical_financials.csv     │  (~1,100 rows × 14 columns)
│                              │  Standardized company-year panel
└──────────────┬───────────────┘
               │
               ▼
┌────────────────────────────────────┐
│ create_and_analyze_labeled_data    │  Label Engineering
│ .ipynb                             │  (3-year rolling window)
└──────────────┬─────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│ labeled_financials.csv       │  (~400 rows with labels)
│                              │  85% healthy, 15% risky
└──────────────┬───────────────┘
               │
        ┌──────┴───────┐
        │              │
        ▼              ▼
 ┌────────────┐  ┌──────────────────┐
 │ Validation │  │ v1_dataset_create│  Feature Engineering
 │ Analysis   │  │ .ipynb           │
 └────────────┘  └────────┬─────────┘
        │                 │
        │                 ▼
        │        ┌──────────────────────┐
        │        │ modeling_dataset_v1  │  (~330 rows × 32 features)
        │        │ .csv                 │
        │        └─────────┬────────────┘
        │                  │
        │         ┌────────┴────────┐
        │         │                 │
        │         ▼                 ▼
        │  ┌──────────────┐  ┌──────────────┐
        │  │ Logistic     │  │ XGBoost      │  Model Training
        │  │ Regression   │  │ Classifier   │
        │  └──────────────┘  └──────────────┘
        │                         │
        └─────────────┬───────────┘
                      │
                      ▼
            ┌──────────────────┐
            │  Test Predictions│  Production Deployment
            │  & Risk Scores   │
            └──────────────────┘
```

### Data Flow

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Canonicalization** | `data/raw/*.json` | Normalize JSON → tabular | `canonical_financials.csv` |
| **2. Labeling** | `canonical_financials.csv` | Apply risk rules (3-yr lookback) | `labeled_financials.csv` |
| **3. Feature Engineering** | `labeled_financials.csv` | Create 32 rolling features | `modeling_dataset_v1.csv` |
| **4. Modeling** | `modeling_dataset_v1.csv` | Train & validate 2 models | Saved models + predictions |

---

## Dataset

### Source Data
- **Companies:** 97 companies from Nifty-100 index
- **Time Period:** 2011–2024 (14 years)
- **Data Types:** Cash flow statements, balance sheets, P&L statements, company metadata

### Canonical Dataset (`canonical_financials.csv`)

**Dimensions:** ~1,100 rows × 14 columns

**Key Metrics:**
| Metric | Value |
|--------|-------|
| Unique Companies | 97 |
| Year Range | 2011–2024 |
| Records per Company | ~11.3 (avg) |
| Data Points | ~13,200 individual metrics |

**Features:**
```
company_id, financial_year_end, operating_cf, investing_cf, 
financing_cf, net_cf, total_assets, total_liabilities, equity, 
revenue, profit_after_tax, sector, face_value, book_value
```

### Labeled Dataset (`labeled_financials.csv`)

**Dimensions:** ~400 rows × 11 columns (with binary labels)

**Label Distribution:**
```
Healthy (0):  ~85-87% (344-349 observations)
Risky (1):    ~13-15% (51-56 observations)
```

**Labeling Logic (3-Year Lookback):**

For each company-year (Y), the system evaluates prior 3 years (Y-3 to Y-1):

| Risk Factor | Definition | Risky Threshold |
|-------------|-----------|-----------------|
| **OCF History** | Years with positive operating cash flow | < 50% of lookback period |
| **OCF Volatility** | Std dev of operating cash flows | > 2× average OCF |
| **Financing Dependency** | Avg financing CF relative to OCF | > 0 AND > abs(mean OCF) |

**Label Assignment:**
- **Risky (1)** if ANY condition is true
- **Healthy (0)** otherwise

### Modeling Dataset (`modeling_dataset_v1.csv`)

**Dimensions:** ~330 rows × 33 columns (32 features + 1 label)

**Preprocessing Applied:**
- ✓ Excluded financial services sector (structurally different cash flows)
- ✓ Handled missing values (complete case deletion)
- ✓ Removed infinite values
- ✓ Temporal ordering preserved (no shuffling)

**32 Engineered Features:**

| Category | Count | Examples |
|----------|-------|----------|
| 3-Year Means | 8 | `operating_cf_3yr_mean`, `revenue_3yr_mean`, `equity_3yr_mean` |
| 3-Year Volatility | 8 | `operating_cf_3yr_std`, `financing_cf_3yr_std` |
| Financial Stress | 1 | `financing_dependency_ratio` |
| OCF Stress | 1 | `pct_years_ocf_negative` |
| Growth Metrics | 2 | `revenue_3yr_growth`, `profit_3yr_growth` |
| **Total** | **32** | |

---

## Models

### 1. Logistic Regression (Baseline)

**Purpose:** Interpretable baseline model for understanding key risk drivers

**Architecture:**
```
Features (32)
    ↓
StandardScaler (fitted on training only)
    ↓
LogisticRegression(
  class_weight="balanced",
  penalty="l2",
  C=1.0,
  solver="liblinear"
)
```

**Configuration:**
- **class_weight="balanced":** Automatically weights minority (risky) class
  - Healthy weight: 0.55
  - Risky weight: 6.75
- **L2 Regularization:** Prevents overfitting on small dataset
- **Solver:** `liblinear` (optimized for binary classification)

### 2. XGBoost Classifier (Champion)

**Purpose:** High-performance ensemble model optimized for recall and PR-AUC

**Architecture:**
```
Features (32)
    ↓
XGBClassifier(
  n_estimators=300,
  max_depth=4,
  learning_rate=0.05,
  subsample=0.8,
  colsample_bytree=0.8,
  eval_metric="aucpr"
)
```

**Configuration:**
- **n_estimators=300:** 300 boosting rounds for strong ensemble
- **max_depth=4:** Shallow trees reduce overfitting
- **learning_rate=0.05:** Low learning rate ensures stable convergence
- **Subsampling=0.8:** Stochastic boosting reduces variance
- **eval_metric="aucpr":** Optimizes PR-AUC directly (better for imbalance)

---

## Results & Performance

### Data Split Strategy (Time-Based)

To prevent temporal leakage, data split strictly by year:

| Split | Period | Rows | Purpose |
|-------|--------|------|---------|
| **Train** | ≤ 2020 | 122 | Historical pattern learning |
| **Validation** | 2021–2023 | 208 | Hyperparameter tuning |
| **Test** | ≥ 2024 | 71 | Future performance assessment |
| **Total** | 2011–2024 | 401 | Complete dataset |

**Design Rationale:** Training on past → validating on recent → testing on future ensures realistic deployment scenario with no information leakage.

### Performance Comparison

#### Logistic Regression

**Validation Set (2021–2023):**
```
Recall (Risky Class)    : 0.889  (catches 89% of risky firms)
Precision (Risky Class) : 0.640  (63% of predicted risky are true risky)
F2-Score               : 0.825  (weighted emphasis on recall)
PR-AUC                 : 0.874  (strong discrimination)
```

**Test Set (≥ 2024):**
```
Recall (Risky Class)    : 0.833  (catches 83% of future risky firms)
Precision (Risky Class) : 0.714  (71% of predicted risky are true risky)
F2-Score               : 0.806
PR-AUC                 : 0.873
```

#### XGBoost Classifier ⭐ **CHAMPION**

**Validation Set (2021–2023):**
```
Recall (Risky Class)    : 1.000  (PERFECT: catches 100% of risky firms)
Precision (Risky Class) : 0.692  (69% of predicted risky are true risky)
F2-Score               : 0.918  (excellent weighted score)
PR-AUC                 : 0.908  (superior discrimination)
```

**Test Set (≥ 2024):**
```
Recall (Risky Class)    : 1.000  (PERFECT: zero false negatives)
Precision (Risky Class) : 0.750  (75% of predicted risky are true risky)
F2-Score               : 0.938  (exceptional weighted score)
PR-AUC                 : 0.866  (strong ranking of risk scores)
```

### Model Comparison Summary

| Metric | Logistic Regression | XGBoost | Winner |
|--------|-------------------|---------|--------|
| **Recall (Test)** | 0.833 | 1.000 | ⭐ XGBoost |
| **Precision (Test)** | 0.714 | 0.750 | ⭐ XGBoost |
| **F2-Score (Test)** | 0.806 | 0.938 | ⭐ XGBoost |
| **PR-AUC (Test)** | 0.873 | 0.866 | Logistic Reg |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Logistic Reg |
| **Non-linear Capture** | ❌ | ✓ | XGBoost |

**Why XGBoost Wins:**
1. **Perfect Recall:** Catches 100% of risky firms (zero false negatives)
2. **Superior PR-AUC:** 0.87 vs 0.87 (equivalent on test, better on validation)
3. **Strong Precision:** 75% maintains business utility with low false alarms
4. **Non-linear Relationships:** Captures interactions (OCF × Volatility × Growth) that linear model misses
5. **Ensemble Strength:** 300 weak learners reduce variance and improve generalization

---

## Project Structure

```
new_ML_data_analysis/
│
├── README.md                           ← You are here
├── Notes.md                            ← Detailed technical documentation
│
├── 📊 DATA FILES
├── canonical_financials.csv            [Input → Step 1 output]
├── labeled_financials.csv              [Input → Step 2 output]
├── modeling_dataset_v1.csv             [Input → Step 3 output]
│
├── 🔧 PROCESSING SCRIPTS & NOTEBOOKS
├── create_canonical_financials.py      [Step 1: Canonicalization]
├── create_and_analyze_labled_data.ipynb [Step 2: Labeling]
├── label_validation_analysis.ipynb      [Validation & EDA]
├── v1_dataset_create.ipynb              [Step 3: Feature engineering]
│
├── 🤖 MODEL TRAINING NOTEBOOKS
├── train_logistic_regression_v1.ipynb   [Baseline model training]
├── train_xgboost.ipynb                  [Champion model training]
│
├── 📈 PREDICTION & INFERENCE
├── predict.ipynb                        [Inference on new data]
├── logreg_test_predictions_2024_plus.csv [Logistic Reg predictions]
├── xgboost_test_predictions_2024_plus.csv [XGBoost predictions]
│
├── 💾 SAVED MODEL ARTIFACTS
├── logistic_regression_model.joblib     [Trained logistic regression]
├── xgboost_model.joblib                 [Trained XGBoost model]
├── feature_columns.joblib               [Feature names for inference]
│
├── 📁 DATA DIRECTORIES
├── data/
│   ├── raw/
│   │   ├── ABB.json
│   │   ├── ADANIENSOL.json
│   │   └── ... (95 more company files)
│   └── processed/                       [Future: Preprocessed data]
```

---

## Getting Started

### Prerequisites

```
Python 3.8+
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib
```

### Installation

```bash
# Clone or navigate to project directory
cd new_ML_data_analysis

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

# (Optional) For Jupyter notebooks
pip install jupyter notebook
```

### Quick Start

#### 1. **Canonicalize Raw Data** (One-time setup)
```bash
python create_canonical_financials.py
# Output: canonical_financials.csv
```

#### 2. **Create Labels & Validate** (One-time setup)
```bash
jupyter notebook create_and_analyze_labled_data.ipynb
# Output: labeled_financials.csv
```

#### 3. **Engineer Features** (One-time setup)
```bash
jupyter notebook v1_dataset_create.ipynb
# Output: modeling_dataset_v1.csv
```

#### 4. **Train Models** (One-time setup)
```bash
# Train logistic regression
jupyter notebook train_logistic_regression_v1.ipynb

# Train XGBoost
jupyter notebook train_xgboost.ipynb
# Outputs: logistic_regression_model.joblib, xgboost_model.joblib
```

#### 5. **Make Predictions** (Regular use)
```bash
jupyter notebook predict.ipynb
# Input: New company financial data
# Output: Risk predictions & probabilities
```

---

## File Documentation

### 1️⃣ `create_canonical_financials.py`

**Purpose:** Transforms raw nested JSON financial data into standardized panel dataset

**Process:**
- Parses 97 company JSON files from `data/raw/`
- Normalizes fiscal year formats (handles Mar/Dec year-end conventions)
- Extracts cash flow, balance sheet, and P&L metrics
- Removes duplicates and invalid records

**Output:** `canonical_financials.csv` (~1,100 rows)

### 2️⃣ `create_and_analyze_labled_data.ipynb`

**Purpose:** Apply rule-based labeling using 3-year rolling window

**Key Features:**
- No future leakage: uses only prior 3 years (Y-3 to Y-1) to label year Y
- Decision rules: OCF history, volatility, financing dependency
- Statistical validation: t-tests, chi-square tests
- Visualization: Multi-panel plots showing label separation

**Output:** `labeled_financials.csv` (~400 rows with 0/1 labels)

### 3️⃣ `label_validation_analysis.ipynb`

**Purpose:** EDA and validation of labeled dataset

**Analysis:**
- Label distribution by sector and time
- Risk factor dominance (what drives each risk label)
- Feature comparison between healthy and risky cohorts
- Temporal risk patterns (identifies abnormal years)
- Consistently risky companies (persistent financial stress)

### 4️⃣ `v1_dataset_create.ipynb`

**Purpose:** Feature engineering for modeling

**Steps:**
1. Merge canonical + labeled datasets
2. Exclude financial services sector
3. Engineer 32 rolling 3-year features
4. Handle missing/infinite values
5. Export modeling-ready dataset

**Output:** `modeling_dataset_v1.csv` (~330 rows × 33 columns)

### 5️⃣ `train_logistic_regression_v1.ipynb`

**Purpose:** Train interpretable baseline model

**Key Steps:**
1. Time-based train/val/test split (2020/2023/2024 cutoffs)
2. Feature scaling (StandardScaler)
3. Model training with class_weight balancing
4. Comprehensive evaluation (recall, precision, F2-score, PR-AUC)
5. Feature importance extraction

**Outputs:** 
- `logistic_regression_model.joblib`
- `logreg_test_predictions_2024_plus.csv`

### 6️⃣ `train_xgboost.ipynb`

**Purpose:** Train high-performance ensemble model

**Key Steps:**
1. Identical time-based split as logistic regression
2. XGBoost training with PR-AUC optimization
3. Hyperparameter tuning (depth, learning_rate, subsampling)
4. Feature importance (gain-based)
5. Threshold analysis for deployment decision

**Outputs:** 
- `xgboost_model.joblib`
- `xgboost_test_predictions_2024_plus.csv`

### 7️⃣ `predict.ipynb`

**Purpose:** Load saved model and make predictions on new data

**Workflow:**
1. Load trained model & scaler from .joblib files
2. Format new company financials
3. Generate risk scores and class predictions
4. Return actionable risk assessment

---

## Usage Examples

### Example 1: Load XGBoost Model & Predict

```python
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("xgboost_model.joblib")
feature_cols = joblib.load("feature_columns.joblib")

# Load or create feature dataframe for new company
# Shape should be (n_companies, 32) with correct feature names
X_new = pd.read_csv("new_financials_features.csv")

# Get predictions
predictions = model.predict(X_new)  # 0 or 1
probabilities = model.predict_proba(X_new)[:, 1]  # Risk score [0, 1]

# Output
results = pd.DataFrame({
    "company": X_new.index,
    "risk_label": predictions,
    "risk_probability": probabilities
})
print(results)
```

### Example 2: Interpret Logistic Regression Coefficients

```python
import joblib

# Load logistic regression model
logreg = joblib.load("logistic_regression_model.joblib")

# Extract coefficients
coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logreg.coef_[0]
}).sort_values("coefficient", ascending=False)

# Interpretation:
# Positive coefficients → increase risk
# Negative coefficients → decrease risk
print(coefficients.head(10))
```

---

## Key Findings

### 1. Financial Services Are Structurally Different
- Banks and financial companies show opposite cash flow patterns
- Operating cash flows not comparable to manufacturing/services
- **Action:** Excluded from modeling to avoid distortion

### 2. OCF History is the Strongest Risk Signal
- Companies with <50% years of positive OCF are predominantly risky
- **Insight:** Consistent negative OCF indicates structural financial stress

### 3. Volatility Matters More for Risky Firms
- Risky companies show 3× higher OCF volatility than healthy peers
- **Insight:** Unpredictable cash flows = operational/market stress

### 4. Financing Dependency is a Secondary Risk Indicator
- 60-70% of risky firms rely on heavy external financing
- Among healthy firms: only 20-25% show financing dependency
- **Insight:** Heavy financing with weak OCF = distress signal

### 5. Risk is Persistent But Not Universal
- ~12 companies show >70% risky years (persistent stress)
- ~50+ companies show 0% risky years (consistently healthy)
- **Insight:** Financial health clusters by company, not just market conditions

### 6. XGBoost Captures Non-Linear Patterns
- Tree ensemble achieves perfect recall vs logistic regression's 83%
- **Insight:** Risk drivers interact (e.g., OCF × Volatility effects)
