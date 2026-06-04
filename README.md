# Ethical AI-Based Employee Performance Evaluation System

An end-to-end machine learning pipeline for predicting employee **Performance Ratings** (2, 3, or 4) in an unbiased and ethical manner. The project tackles class imbalance, multicollinearity, and ordinal target structure through a series of rigorously designed experiments.

---

## Project Overview

Employee performance appraisals are high-stakes decisions. Biased or poorly calibrated models can lead to unfair outcomes. This project builds a transparent, statistically validated classification system with the following goals:

- **Predict** ordinal performance ratings (2 = Below Average, 3 = Average, 4 = Above Average)
- **Minimise bias** through ethical feature selection (VIF-based multicollinearity removal, Variance Inflation Factor)
- **Handle class imbalance** using ADASYN oversampling
- **Validate** model superiority via bootstrap-based statistical testing

---

## Dataset

**File:** `Employee_Performance_Dataset.csv`

The dataset contains HR attributes for employees across multiple departments. The **target variable** is `Performance Rating` with three ordinal classes: **2**, **3**, and **4**.

**Final shape:** 23 selected features after VIF-based reduction (from ~36 engineered features).

---

## Methodology

### 1. Data Preprocessing & Encoding

Raw categorical variables are encoded using appropriate strategies based on their nature:
- **Nominal binary** (2 levels): direct 0/1 mapping
- **Ordinal** (`Business_Travel_Frequency`): scikit-learn `LabelEncoder`
- **Nominal multi-level**: one-hot encoding with `drop_first=True` to avoid the dummy variable trap
- **High-cardinality nominal** (`Employee Job Role`): normalized frequency encoding to avoid dimensionality explosion

### 2. Feature Engineering

**Sanity Checks:**
- Verified logical constraints: `Total_Work_Experience_In_Years ≥ Experience_Years_At_This_Company ≥ Experience_Years_In_Current_Role`
- Confirmed zero violations in the dataset

**Variance Inflation Factor (VIF) Analysis:**

VIF measures multicollinearity — how well a feature can be predicted by the others. High VIF signals redundant features that inflate model variance and reduce interpretability.

Iterative VIF-based elimination was performed, removing features that exceeded the threshold until a stable set of 22 features remained. Features removed include `Age`, several one-hot encoded education and department columns, and experience-related variables.

**Final 22 selected features:**

### 3. Model Architecture – Ordinal Classification

**Why not standard multiclass?**

Standard classifiers treat all classes as independent. Ordinal classifiers respect the natural order: misclassifying a 2 as a 3 should be penalised less than misclassifying it as a 4.

**Threshold-based ordinal decomposition:**

The three-class ordinal problem is decomposed into two binary tasks:

|   Task  |  Label |     Meaning    |
|---------|--------|----------------|
| `y_ge3` | 0 or 1 | Is rating ≥ 3? |
| `y_ge4` | 0 or 1 | Is rating ≥ 4? |

**Prediction rule:**
- If `y ≥ 3` is **False** → label = **2**
- If `y ≥ 3` is **True** and `y ≥ 4` is **False** → label = **3**
- If both are **True** → label = **4**

Labels are additionally encoded via `LabelEncoder`: `{2→0, 3→1, 4→2}` for model compatibility.

### 4. Class Imbalance Handling – ADASYN

The dataset is imbalanced with a large majority in class 3 and a minority in classes 2 and 4. **ADASYN** (Adaptive Synthetic Sampling) generates synthetic samples for underrepresented classes by focusing more on hard-to-learn boundary samples compared to standard SMOTE.

Applied only to the **training split** to prevent data leakage.

### 5. Hyperparameter Tuning – GridSearchCV

3-fold cross-validated grid search is used for all models. The **scoring metric** is the F1-score for **class 4 only** (the most critical minority class) using a custom `make_scorer`.

### 6. Threshold Tuning for Minority Class

After GridSearchCV, predicted probabilities are post-processed. Instead of using the argmax rule uniformly, the threshold for predicting class 4 is swept across `[0.2, 0.6]` in steps of 0.05. The threshold yielding the highest F1 for class 4 is selected.

### 7. Statistical Validation – Bootstrap

**1000 bootstrap samples** are drawn from the test set (with replacement). For each bootstrap sample, F1 for class 4 is computed for all six trained models, yielding an empirical distribution of model performance.

**Outputs:**
- Mean F1 (class 4) per model
- 95% Bootstrap Confidence Intervals (2.5th and 97.5th percentile)
- Pairwise comparisons: reference model vs. all others
- Two-sided p-values to determine statistical significance (α = 0.05)

---

## Experiments (Scenarios)

| Scenario | Model         | Feature Set                  | Notes                              |
|----------|---------------|------------------------------|------------------------------------|
| **S1**   | XGBoost       | All features (post-encoding) | Baseline XGB                       |
| **S2**   | CatBoost      | All features                 | Baseline CatBoost                  |
| **S3**   | Random Forest | All features                 | Baseline RF                        |
| **S4A**  | XGBoost       | VIF-reduced (22 features)    | XGB with multicollinearity removed |
| **S4B**  | CatBoost      | VIF-reduced                  | CatBoost with VIF selection        |
| **S4C**  | Random Forest | VIF-reduced                  | RF with VIF selection              |

All scenarios use: **ADASYN + GridSearchCV + Threshold Tuning**

---

## Evaluation Metrics

| Metric               | Description                                                |
|----------------------|------------------------------------------------------------|
| **F1 (Class 4)**     | Primary metric — F1 for the minority "Above Average" class |
| **Macro F1**         | Unweighted average F1 across all classes                   |
| **Weighted F1**      | Support-weighted average F1                                |
| **Support**          | Number of true samples per class                           |
| **Confusion Matrix** | Full breakdown of predictions vs. actuals                  |
| **Bootstrap CI**     | 95% confidence interval for F1 (class 4)                   |
| **p-value**          | Statistical significance of pairwise differences           |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, modelling, evaluation, GridSearchCV |
| `xgboost` | XGBoost classifier |
| `catboost` | CatBoost classifier |
| `imbalanced-learn` | ADASYN oversampling |
| `statsmodels` | VIF computation |

---

## Results Summary

All six models are compared via bootstrap-ranked F1 (class 4). The final output includes:

- **Bootstrap Ranking Table** — models sorted by mean F1 (class 4) with 95% CI
- **Pairwise Significance Tests** — whether the best model is statistically superior
- **All Confusion Matrices** — printed per scenario
- **All Classification Reports** — precision, recall, F1 per class for each scenario
- The pipeline achieved a peak classification accuracy of 80.4%, demonstrating strong predictive performance across all three ordinal classes.

The best model is selected based on the highest bootstrap mean F1 for class 4 (Performance Rating = 4), the most underrepresented and high-stakes category.

---

## File Structure

```
.
├── TRSR_File_1_COMPLETED_FINAL.ipynb   # Main notebook (full pipeline)
├── Employee_Performance_Dataset.csv    # Raw dataset (required)
└── README.md                           # This file
```

---

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost catboost imbalanced-learn statsmodels
   ```

2. Place `Employee_Performance_Dataset.csv` in the same directory as the notebook.

3. Open and run `TRSR_File_1_COMPLETED_FINAL.ipynb` top to bottom. All scenarios execute sequentially and final bootstrap results are printed at the end.

---

*Built with a focus on ethical, unbiased, statistically validated ML for HR decision support.*
