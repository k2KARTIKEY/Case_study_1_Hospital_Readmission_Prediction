# Case_study_1_Hospital_Readmission_Prediction
This repository contains two Machine Learning case studies developed for the MLE CA-1 assessment.

Case Study 1: Hospital Readmission Prediction
Objective
Predict whether a diabetic patient will be readmitted to the hospital within 30 days.

Dataset
UCI Diabetes 130-US Hospitals for Years 1999-2008 dataset.

Methodology
Data cleaning and missing value handling
Categorical feature encoding using One-Hot Encoding
Train-test split with stratification
Feature scaling using StandardScaler
Logistic Regression with L2 regularization
ROC-AUC evaluation
Classification threshold analysis
Clinical cost analysis of False Negatives and False Positives
Results
Model: Logistic Regression
Regularization: L2
ROC-AUC: 0.635
Selected Threshold: 0.10
Precision: 0.154
Recall: 0.631
F1-Score: 0.247
A lower threshold was considered because missing a patient who is actually at risk of readmission can have greater clinical consequences than generating an unnecessary follow-up alert.

