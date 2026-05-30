# Breast Cancer Prediction

## Project Overview
This project predicts whether a breast tumor is benign or malignant using machine learning.  
The model is trained on medical features extracted from the dataset and uses Logistic Regression for classification.

## Objective
The main goal of this project is to support early cancer detection by classifying tumors into:
- Benign
- Malignant

## Workflow
- Load and inspect the dataset
- Remove unnecessary columns such as `id`
- Convert the diagnosis column into numeric labels
- Normalize the input features
- Split the data into training and testing sets
- Train a Logistic Regression model
- Evaluate performance using accuracy and confusion matrix

## Dataset
The dataset file is loaded from:

`./dataset/cancer (1).csv`

## Target Label
The `diagnosis` column is used as the target:
- `M` means Malignant
- `B` means Benign

## Model Used
- Logistic Regression

## Evaluation Metrics
- Accuracy Score
- Confusion Matrix

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Install the required libraries
2. Place the dataset in the `dataset` folder
3. Run the Python script

## Output
The project prints:
- Dataset preview
- Dataset statistics
- Model accuracy
- Confusion matrix plot

## Notes
This project is focused on classification, not medical diagnosis.  
It should be used for educational and research purposes only.