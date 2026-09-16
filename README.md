# Hospital Readmission Prediction

A machine learning project that predicts whether a patient with diabetes will be readmitted to a hospital within 30 days. It uses clinical and administrative data from 130 U.S. hospitals and trains a logistic regression model to identify high-risk patients.

## Project Overview

Hospital readmissions can increase healthcare costs and may indicate gaps in post-discharge care. This project creates a binary classification model:

- **1 — Readmitted:** Patient was readmitted within 30 days (`<30`)
- **0 — Not readmitted:** Patient was readmitted after 30 days (`>30`) or was not readmitted (`NO`)

Because missing a high-risk patient may have a greater clinical impact than performing an unnecessary follow-up, the project evaluates lower classification thresholds to improve recall.

## Dataset

The project uses the **Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI Machine Learning Repository.

- **UCI dataset ID:** 296
- **Records:** 101,766 patient encounters
- **Original columns:** 48
- **Target variable:** `readmitted`

The dataset is downloaded automatically using the `ucimlrepo` package.

## Project Workflow

The notebook performs the following steps:

1. Downloads and explores the dataset
2. Identifies missing values
3. Removes columns with large amounts of missing data
4. Fills remaining missing categorical values with `Unknown`
5. Converts the target into a binary variable
6. One-hot encodes categorical features
7. Splits the data into training and test sets
8. Standardizes the features
9. Trains an L2-regularized logistic regression model
10. Evaluates the model using ROC-AUC, precision, recall, F1-score, and confusion matrices
11. Compares different classification thresholds
12. Discusses the clinical impact of false positives and false negatives

## Data Preprocessing

The following columns are removed because of high levels of missing data:

- `weight`
- `max_glu_serum`
- `A1Cresult`
- `medical_specialty`
- `payer_code`

Missing values in the following columns are replaced with `Unknown`:

- `race`
- `diag_1`
- `diag_2`
- `diag_3`

After preprocessing:

- **Features:** 42
- **Categorical features:** 31
- **Features after one-hot encoding:** 2,330
- **Positive class:** 11,357 encounters, approximately 11.16% of the dataset

## Model

The project uses logistic regression with L2 regularization:

```python
LogisticRegression(
    penalty="l2",
    max_iter=1000,
    random_state=42
)
```

The dataset is divided into:

- **Training set:** 81,412 records
- **Test set:** 20,354 records
- **Test size:** 20%
- **Split method:** Stratified by the target variable

## Results

The final model results are:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.635 |
| Selected threshold | 0.10 |
| Precision | 0.154 |
| Recall | 0.631 |
| F1-score | 0.247 |

### Threshold Comparison

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.50 | 0.436 | 0.026 | 0.048 |
| 0.40 | 0.352 | 0.041 | 0.073 |
| 0.30 | 0.329 | 0.087 | 0.137 |
| 0.20 | 0.241 | 0.200 | 0.218 |
| 0.10 | 0.154 | 0.631 | 0.247 |

The default threshold of `0.50` produces very low recall for patients readmitted within 30 days. Lowering the threshold to `0.10` improves recall, but it also increases the number of false positives.

## Clinical Interpretation

- A **false negative** occurs when a patient who will be readmitted is classified as low risk. The patient may not receive timely follow-up or intervention.
- A **false positive** occurs when a patient who will not be readmitted is classified as high risk. This may result in unnecessary monitoring or resource use.

Since false negatives may have a greater clinical cost, the selected threshold prioritizes recall.

> **Important:** This model is an educational baseline. It should not be used for clinical decision-making without additional validation, fairness assessment, probability calibration, and review by qualified healthcare professionals.

## Installation

Install the required Python packages:

```bash
pip install ucimlrepo pandas scikit-learn matplotlib jupyter
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open and run:

```text
Hospital_Readmission_Prediction.ipynb
```

The notebook downloads the dataset automatically, so no manual dataset download is required.

## Requirements

- Python 3.9 or later
- pandas
- scikit-learn
- matplotlib
- ucimlrepo
- Jupyter Notebook or JupyterLab

## Possible Improvements

Future improvements could include:

- Handling class imbalance using class weights or resampling
- Grouping diagnosis codes into clinically meaningful categories
- Preventing patient-level data leakage
- Comparing additional machine learning models
- Performing cross-validation and hyperparameter tuning
- Calibrating predicted probabilities
- Selecting a threshold based on clinical costs
- Evaluating fairness across demographic groups
- Adding precision-recall and calibration curves
- Using explainability methods such as SHAP
- Validating the model on an external dataset

## License

This project is intended for educational and research purposes. Refer to the UCI Machine Learning Repository for the dataset’s license, citation requirements, and usage conditions.
