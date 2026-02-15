# Machine Learning Model Comparison

## Problem Statement

The goal of this project is to predict the quality of wine, categorized as 'low', 'medium', or 'high', based on its chemical properties.

## Dataset Description

The dataset used is a combination of the red and white wine quality datasets from the UCI Machine Learning Repository. It contains 6,497 samples and 13 columns, including 11 chemical properties (e.g., 'fixed acidity', 'volatile acidity', 'citric acid'), 'alcohol' content, and the 'wine_type' (red or white). The original 'quality' score (from 3 to 9) has been categorized into three classes: 'low' (3-5), 'medium' (6), and 'high' (7-9).

## Models Used

### Comparison Table

| ML Model Name     | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ----------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.5823   | 0.7324 | 0.59      | 0.58   | 0.58   | 0.3268 |
| Decision Tree     | 0.5915   | 0.7458 | 0.59      | 0.59   | 0.59   | 0.3576 |
| kNN               | 0.6708   | 0.7361 | 0.67      | 0.67   | 0.67   | 0.4814 |
| Naive Bayes       | 0.4815   | 0.6554 | 0.50      | 0.48   | 0.48   | 0.2172 |
| Random Forest (Ensemble) | 0.7162   | 0.8523 | 0.72      | 0.72   | 0.71   | 0.5401 |
| XGBoost (Ensemble)  | 0.6869   | 0.8358 | 0.69      | 0.69   | 0.69   | 0.5005 |

### Observations

| ML Model Name     | Observation about model performance |
| ----------------- | ----------------------------------- |
| Logistic Regression | Achieved a moderate accuracy of 58.2%. The AUC and MCC scores indicate a performance better than random guessing, but with room for improvement. |
| Decision Tree     | Performed slightly better than Logistic Regression with an accuracy of 59.2% and improved MCC and AUC scores. |
| kNN               | With an optimal k=1, this model showed a significant improvement with 67.1% accuracy and a strong MCC of 0.4814. |
| Naive Bayes       | Gaussian Naive Bayes had the lowest performance among the models across all metrics, with an accuracy of only 48.2%. |
| Random Forest (Ensemble) | This was the best-performing model with the highest scores in all metrics: Accuracy (71.6%), AUC (0.852), and MCC (0.540). Feature importance analysis revealed 'alcohol' to be the most influential predictor. |
| XGBoost (Ensemble)  | A strong performer and the second-best model, with high scores across the board, including an impressive AUC of 0.836. |
