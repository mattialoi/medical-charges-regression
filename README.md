# Medical Charges Prediction — Linear Regression from Scratch

Midterm project for the **Supervised Learning** course (BSc Data Science & Artificial Intelligence, SUPSI — A.Y. 2025/2026).

The goal is to predict the medical **charges (USD)** of a patient from demographic, lifestyle and socio‑economic features, comparing scikit‑learn models with **linear regression implemented from scratch** (Batch and Mini‑Batch Gradient Descent), feature augmentation and regularization.

## Highlights

- Full **data‑quality pipeline**: duplicates, invalid/inconsistent values, negative values, under‑age records
- **Encoding** (one‑hot, binary, ordinal), **outlier capping**, **log transform** of skewed income
- **KNN imputation** (k = 3) of missing `smoker` and `physical_activity`
- **Feature selection** combining ANOVA, tree‑based importance, linear models and RFE (≈ half of the features kept)
- **Gradient Descent implemented by hand** (batch and mini‑batch) with learning‑rate search, early stopping and numerical‑stability checks
- **Hand‑crafted augmented dataset** to capture non‑linear interactions
- **Lasso / Ridge** regularization and analysis of the number of active features vs α
- **Stratified analysis by gender**

## Results

| Task / Model | Dataset | R² | MSE |
|---|---|---|---|
| Linear Regression (sklearn) | with income | 0.7596 | 34,302,279 |
| Linear Regression (sklearn) | without income | 0.6496 | 49,985,216 |
| Batch Gradient Descent | with income | 0.7603 | 34,192,182 |
| Batch Gradient Descent | without income | 0.6502 | 49,899,155 |
| Mini‑Batch Gradient Descent | with income | 0.7604 | 34,181,447 |
| Mini‑Batch Gradient Descent | without income | 0.6502 | 49,900,799 |
| Batch GD on augmented features | augmented | 0.8544 | 20,768,743 |
| **Lasso (α = 5)** | augmented | **0.8673** | – |
| Ridge (α = 1) | augmented | 0.8621 | – |
| Batch GD — Male | with / without income | 0.7193 / 0.6453 | – |
| Batch GD — Female | with / without income | 0.7480 / 0.6875 | – |

**Takeaways:** income is the most influential feature; the optimization method (closed form vs GD vs mini‑batch) barely changes accuracy, while **feature augmentation and regularization** give the largest improvement (R² from ~0.76 to ~0.87).

## Repository contents

| File | Description |
|---|---|
| `Loi_Mattia_midtermproject_SL.ipynb` | Full notebook (EDA, preprocessing, models, results) |
| `Loi_Mattia_midterm_report.pdf` | Written report with discussion of every task |
| `data.csv` | Dataset (3,098 samples, 14 features) |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Loi_Mattia_midtermproject_SL.ipynb
```

## Tech stack

Python · NumPy · pandas · scikit‑learn · Matplotlib · Seaborn · missingno

## Author

**Mattia Loi** — BSc Data Science & Artificial Intelligence, SUPSI
