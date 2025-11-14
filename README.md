# Breast Cancer Prediction Project

## 1. Introduction

This project aims to build a machine learning model to predict breast cancer diagnosis (Benign or Malignant) based on various features extracted from digital images of fine needle aspirate (FNA) of breast masses. The goal is to classify a diagnosis as either `0` (Benign) or `1` (Malignant).

## 2. Data Loading and Preprocessing

The dataset, `data.csv`, was loaded into a Pandas DataFrame. Initial data exploration revealed the presence of an 'id' column and an 'Unnamed: 32' column with all null values, which were subsequently dropped.

### Key Preprocessing Steps:

- **Column Dropping**: The 'id' and 'Unnamed: 32' columns were removed as they are not relevant for the prediction task.
- **Target Variable Transformation**: The `diagnosis` column, originally categorical ('B' for Benign, 'M' for Malignant), was converted into a binary numerical format where 'B' is mapped to `0` and 'M' to `1`.
- **Feature-Target Split**: The dataset was split into features (`X`) and the target variable (`y`).
- **Feature Scaling**: All features in `X` were scaled using `StandardScaler` to ensure that no single feature dominates the model due to its scale.
- **Data Splitting**: The data was divided into training and testing sets using `train_test_split`, with 80% for training and 20% for testing, ensuring reproducibility with `random_state=42`.

## 3. Model Training

A Logistic Regression model was chosen for this binary classification task. The model was initialized and then trained on the scaled training data (`X_train`, `y_train`).

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
```

After training, the model made predictions on the unseen test data (`X_test`).

```python
y_predict = lr.predict(X_test)
```

## 4. Model Evaluation

The performance of the Logistic Regression model was evaluated using accuracy score and a detailed classification report.

### Accuracy Score

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.4f}")
```

**Result**: The model achieved an accuracy of `0.9737`.

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))
```

**Classification Report Output**:

```
              precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

### Side-by-Side Comparison

To visualize the model's predictions against the actual values, a DataFrame was created:

```python
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
display(results.head())
```
## 5. Conclusion

The Logistic Regression model demonstrated strong performance in predicting breast cancer diagnosis, achieving an accuracy of approximately 97.37%. The classification report indicates high precision, recall, and F1-scores for both classes (Benign and Malignant), suggesting that the model is well-suited for this binary classification task. The preprocessing steps, including feature scaling and categorical variable transformation, were crucial in achieving these results.