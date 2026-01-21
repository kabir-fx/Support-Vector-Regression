# Support Vector Regression (SVR) Analysis

This directory contains an implementation of **Support Vector Regression (SVR)**, a type of SVM that supports linear and non-linear regression. It is particularly effective for high-dimensional spaces and non-linear data patterns.

## Dataset Overview

The model uses the `Position_Salaries.csv` dataset, which includes:

- **Features**:
  - `Level`: The hierarchy level of the position (1 to 10).
- **Target**:
  - `Salary`: The annual salary associated with each level.
- **Goal**: Predict the salary for a specific position level (e.g., 6.5) to determine if a potential candidate's salary expectations are consistent with the company's pay scale.

## Implementation Steps

The implementation follows these key steps:

1.  **Data Preprocessing**:
    - Importing libraries (`numpy`, `matplotlib`, `pandas`).
    - Reshaping the target variable `y` into a 2D array, as required by the scaler.
2.  **Feature Scaling**:
    - Applied **StandardScaler** to both the features (`x`) and the target variable (`y`). This is crucial for SVR as it does not have internal feature scaling like other models.
3.  **Model Training**:
    - Used `sklearn.svm.SVR` with the **RBF kernel** (Radial Basis Function).
    - The model was trained on the entire dataset to capture the non-linear relationship between level and salary.
4.  **Prediction and Evaluation**:
    - Predicted a salary for Level 6.5.
    - Used `inverse_transform` to convert the scaled prediction back to the original salary scale.
5.  **Visualization**:
    - Plotted the results in both low and high-resolution (smooth curve) to demonstrate how well the SVR model fits the salary data.

## Results

The SVR model with an RBF kernel provides a highly accurate fit for non-linear data:

- **Prediction for Level 6.5**: approximately **$170,370**, which is a realistic value between levels 6 and 7.
- The model successfully identifies the exponential growth in salaries at higher position levels while ignoring the "outlier" effect of the level 10 CEO salary that often distorts linear models.
