# Salary Prediction with Linear Regression

Predict salary from age, gender, education level, job title and years of experience, and test whether engineered features improve the model.

**Dataset:** [Salary Prediction for Beginners (Kaggle, rkiattisak)](https://www.kaggle.com/datasets/rkiattisak/salary-prediction-for-beginer), 375 rows, 6 columns.
**Tools:** pandas, numpy, matplotlib, seaborn, scikit-learn
**Notebook:** `salary_prediction_colab.ipynb`

## Business objective
Support consistent, bias-aware compensation decisions during hiring by estimating a fair salary for a candidate.

## Data cleaning
| Step | Result |
|---|---|
| Missing values | 2 completely empty rows dropped |
| Duplicates | 49 duplicate rows removed (375 → 324 rows) |
| Education labels | Already 3 clean categories (Bachelor's, Master's, PhD) |
| Salary outlier | 1 row with salary 350 for a junior analyst with 1.5 years' experience dropped as a probable typo (324 → 323 rows) |
| Rare job titles | Titles appearing fewer than 3 times grouped into "Other": 174 → 37 categories |

The threshold of 3 was chosen by comparing options: threshold 2 would create 67 sparse columns for about 258 training rows, while threshold 5 would put 78% of rows into "Other".

## Key EDA findings
- Salary ranges from 30k to 250k and is slightly right-skewed (0.45).
- Salary rises with education: Bachelor's ≈ 74k, Master's ≈ 128k, PhD ≈ 158k.
- Salary is almost linear in experience (correlation 0.925), so a linear model already does well.
- Men's average salary is about 8k higher than women's (≈104k vs ≈96k). This is a raw difference, not adjusted for experience, education or role.
- Age and Years of Experience are 0.98 correlated, so individual coefficients should not be interpreted.

## Feature engineering
All features were screened with 5-fold cross-validation repeated 10 times (50 splits, identical for every model), because the test set has only 65 rows.

| Feature | What it represents | Mean R² (CV) | Better than baseline in |
|---|---|---|---|
| *Baseline (no engineered features)* | | 0.8900 | |
| `experience_level` | Entry (≤3 yrs), Mid (3–10), Senior (>10) | 0.8888 | 54% of splits |
| `age_experience_ratio` | Age / (Years of Experience + 1) | 0.8894 | 48% of splits |
| `experience_squared` | Years of Experience² | 0.8871 | 50% of splits |
| `exp_x_education` | Experience × education rank | 0.8869 | 24% of splits |
| `department` | Job function (Marketing, Finance…) from title keywords | 0.8838 | 16% of splits |
| **`title_level`** | **Seniority from raw job-title keywords: Junior / Standard / Manager / Senior / Executive** | **0.9155** | **86% of splits** |
| `seniority_rank` | Same information as `title_level`, ordinal 1–5 in one column | 0.9089 | 86% of splits |

**What worked and what didn't**
- The two features suggested in the assignment (`experience_level`, `age_experience_ratio`) did **not** reliably improve the model. They repackage experience information that a linear model already has.
- `title_level` is the feature that improved performance. It recovers seniority for all rows, including the 52% that were grouped into "Other", and average salary rises clearly with it (Junior ≈ 48k → Executive ≈ 180k).
- `seniority_rank` is a one-column version of the same idea, not an independent second feature.

## Final results (80/20 split, `random_state=42`, same split for every model)
| Model | Features | Train R² | Test R² | MAE | RMSE |
|---|---|---|---|---|---|
| Baseline | 41 | 0.9140 | 0.8847 | 11,039 | 15,016 |
| + suggested (experience_level + ratio) | 44 | 0.9159 | 0.8867 | 11,194 | 14,885 |
| **+ title_level** | 45 | 0.9376 | **0.9101** | **9,695** | **13,262** |
| + seniority_rank | 42 | 0.9303 | 0.9075 | 9,985 | 13,452 |

`title_level` improves test R² by **+0.0254** and reduces MAE by about 1,340 (12%) versus the baseline. The cross-validated gain (+0.0256, with the model better in 86% of 50 splits) agrees, so the improvement is reliable.

![R2 comparison](figures/final_r2_comparison.png)
![Predicted vs actual](figures/final_predicted_vs_actual.png)

## Limitations
- **Small dataset:** about 258 training rows and 65 test rows, so single-split metrics are noisy. That is why cross-validation was used.
- **Feature screening:** seven features were tested on the same data, so the winner's gain may be slightly optimistic. A held-out dataset would be needed to confirm it.
- **Keyword rules are crude:** the largest errors (27k–36k) include "Senior Project Coordinator" (actual 80k, predicted 116k) and "Office Manager" (actual 65k, predicted 100k), where the title keyword doesn't match actual pay.
- **Average accuracy hides individual misses:** typical error is about 10k, but individuals can be off by more than 30k.

## Bias note
The model uses Gender as an input, and the data shows a raw pay gap. Predicting from historical pay can reproduce existing inequities. Before using any model like this for hiring decisions, remove Gender (and check proxies such as job title), audit outcomes by group, and use predictions only as one input to human review.

## Repository structure
```
├── README.md
├── salary_prediction_colab.ipynb
├── data/            Salary_Data.csv, salary_cleaned.csv
├── figures/         EDA and results plots
└── *.csv            metrics tables (final and cross-validation)
```
