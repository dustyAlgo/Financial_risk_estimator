📌 Project: Financial Health & Risk Detection for Nifty-100 Companies
Problem Definition & Framing

Designed a company-level financial risk detection system to classify Indian large-cap firms as financially healthy vs financially risky, using multi-year cash-flow and financial statement data.

Framed the problem as a forward-looking financial stress screening task, not bankruptcy prediction, prioritizing high recall of risky firms.

Defined interpretability as a first-class requirement, ensuring all predictions could be explained in financial terms.

Dataset Construction & Canonicalization

Built a canonical dataset from nested JSON files (one per company) containing:

Cash flow statements

Balance sheets

Profit & loss statements

Company metadata and expert pros/cons

Flattened data to a company-year level panel dataset (≈1,100+ rows, 97 companies, 2011–2024).

Enforced strict time integrity checks:

No duplicate (company, year) rows

Correct handling of missing years and mixed fiscal/calendar formats

Excluded companies with insufficient history (<5 years) from modeling to prevent unstable labels.

Label Engineering (Rule-Based Financial Health)

Designed a transparent, rule-based labeling framework using prior 3-year financial behavior, avoiding future leakage:

Persistent negative operating cash flow

High cash-flow volatility

Heavy dependence on financing activities

Generated year-wise risk labels, enabling early-warning detection rather than static classification.

Resulting labeled dataset:

~8–9% risky observations

71 companies with sufficient rolling history

Time span: 2016–2024

Exploratory Data Analysis (EDA with Intent)

Conducted target-aware EDA focused on learnability rather than visualization:

Verified strong signal separation between healthy and risky firms

Identified financing dependency and operating cash-flow instability as dominant risk drivers

Performed sector-level risk analysis, discovering:

Financial Services exhibited structurally different dynamics → excluded to avoid model distortion

Validated that extreme values reflected genuine business behavior, not data corruption.

Feature Engineering

Engineered rolling 3-year statistical features to capture financial stability and trends:

Means and volatility of operating, investing, and financing cash flows

Revenue and profit growth metrics

Balance sheet stability indicators

Added domain-specific risk indicators:

Financing dependency ratio

Percentage of years with negative operating cash flow

Final modeling dataset:

401 company-year rows

23 fully non-null, leakage-safe features

Evaluation Strategy

Used strict time-based splits to simulate real-world deployment:

Train: ≤ 2020

Validation: 2021–2023

Test: ≥ 2024

Selected metrics aligned with business risk:

Recall (Risky class) as primary

Precision, PR-AUC, and F2-score as supporting metrics

Avoided accuracy and random splits to prevent misleading performance.

Baseline Model — Logistic Regression

Trained an interpretable logistic regression with:

Standardized features

Class-imbalance handling

Probability-based outputs

Achieved strong generalization:

Validation Recall ≈ 0.89

Test Recall ≈ 0.83 with high precision

Coefficient analysis confirmed financially intuitive behavior:

Negative OCF frequency and financing dependency as strongest risk drivers

Revenue and cash-flow stability as protective factors

Challenger Model — XGBoost

Built an XGBoost challenger under strict constraints:

Same features, same time splits, no feature leakage

Shallow trees for stability and interpretability

Delivered consistent out-of-time improvements over logistic regression:

Validation Recall: 1.00, Precision ≈ 0.69

Test Recall: 1.00, Precision ≈ 0.75

Feature importance aligned with domain knowledge, confirming model trustworthiness.

Model Selection & Validation

Selected XGBoost as the champion model based on:

Higher recall and precision

Stable PR-AUC across time

Zero false negatives on unseen future data

Retained logistic regression as:

Benchmark model

Interpretability reference

Fallback option

Key Outcomes

Built a production-grade financial risk scoring engine with:

Leakage-free temporal validation

Strong generalization to future years

Clear, finance-aligned explanations

Demonstrated how disciplined problem framing, labeling, and evaluation outperform brute-force modeling.