# Credit-Wise Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or
rejected, based on applicant financial, credit, and demographic data.

## Overview
This project applies and compares multiple classification models to predict loan approval
outcomes (`Loan_Approved`), with a focus on precision — since in a banking context, incorrectly
approving a bad loan (false positive) is more costly than incorrectly rejecting a good one.

## Dataset
- **loan_approval_data.csv** — Applicant records including:
  - Demographics: `Gender`, `Marital_Status`, `Education_Level`, `Employment_Status`, `Employer_Category`
  - Financials: `Applicant_Income`, `Coapplicant_Income`, `Credit_Score`, `DTI_Ratio`
  - Loan details: `Loan_Purpose`, `Property_Area`
  - Target: `Loan_Approved`

## Workflow
1. **Data cleaning** — handled missing values (mean imputation for numerical columns, mode
   imputation for categorical columns)
2. **Exploratory Data Analysis (EDA)** — class balance, income/credit score distributions,
   outlier detection via box plots, feature relationships with the approval outcome
3. **Encoding** — Label Encoding for ordinal features (`Education_Level`), One-Hot Encoding for
   nominal categorical features (`Employment_Status`, `Marital_Status`, `Loan_Purpose`,
   `Property_Area`, `Gender`, `Employer_Category`)
4. **Correlation analysis** — heatmap to identify which features most strongly relate to
   loan approval
5. **Train/test split & feature scaling** — 80/20 split with `StandardScaler`
6. **Model training & evaluation** — Logistic Regression, K-Nearest Neighbors, and Naive Bayes
7. **Feature engineering** — added squared terms (`DTI_Ratio_sq`, `Credit_Score_sq`) and
   re-evaluated all three models

## Results

**Before feature engineering:**
| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.783 | 0.770 | 0.777 | 0.865 |
| KNN (k=5) | 0.627 | 0.525 | 0.571 | 0.760 |
| Naive Bayes | 0.804 | 0.738 | 0.769 | 0.865 |

**After feature engineering:**
| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.785 | 0.836 | 0.810 | 0.880 |
| KNN (k=5) | 0.673 | 0.574 | 0.619 | 0.785 |
| Naive Bayes | 0.811 | 0.705 | 0.754 | 0.860 |

**Best model: Naive Bayes** — chosen for having the highest precision, which is the priority
metric for this use case (minimizing false approvals matters more than catching every rejection
in a banking/lending context).

## Files
- `credit_wise_loan_system.ipynb` — Full notebook: EDA, preprocessing, model training, and evaluation
- `loan_approval_data.csv` — Source dataset

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (LogisticRegression, KNeighborsClassifier, GaussianNB, StandardScaler,
  LabelEncoder, OneHotEncoder, SimpleImputer)
- Matplotlib, Seaborn (visualization)
