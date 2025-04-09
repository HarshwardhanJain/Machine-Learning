# =============================================================================
# Compare Baseline (Logistic Regression) and XGBoost on the Titanic Dataset
# (Simple Version)
# =============================================================================
#
# Instructions:
# 1. Download the Titanic dataset from https://www.kaggle.com/c/titanic/data.
# 2. Save the CSV file as 'titanic_dataset.csv' inside the folder 'Lab Work - 04_09_25'.
# 3. Install the required libraries:
#       pip install numpy pandas matplotlib scikit-learn xgboost
# 4. Run the script:
#       python titanic_simple_comparison.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import required modules from scikit-learn and XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------------
# Step 1: Load the Titanic Dataset
# -------------------------------
# Define path: update the path if needed.
titanic_path = r'Lab Work - 04_09_25\titanic_dataset.csv'
if not os.path.exists(titanic_path):
    raise FileNotFoundError("Titanic dataset not found. Please download it from "
                            "https://www.kaggle.com/c/titanic/data and place the file "
                            "'titanic_dataset.csv' inside the folder 'Lab Work - 04_09_25'.")

df = pd.read_csv(titanic_path)
print("Titanic Dataset Preview:")
print(df.head())

# -------------------------------
# Step 2: Preprocess the Data
# -------------------------------
# Drop columns that are unlikely to help with prediction
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values for 'Age' and 'Embarked'
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables: encode 'Sex' and one-hot encode 'Embarked'
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# -------------------------------
# Step 3: Split Data into Features and Target
# -------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# =============================================================================
# Baseline Model: Logistic Regression
# =============================================================================
print("\nTraining Baseline Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_preds)
lr_conf = confusion_matrix(y_test, lr_preds)

print("\nLogistic Regression Results:")
print("Accuracy: {:.2f}%".format(lr_accuracy * 100))
print("Confusion Matrix:")
print(lr_conf)

# Visualize the Logistic Regression confusion matrix counts
plt.figure()
lr_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
plt.bar(lr_labels, lr_conf.ravel())
plt.title("Logistic Regression: Confusion Matrix Counts")
plt.ylabel("Count")
plt.show()

# =============================================================================
# XGBoost Model (Default Parameters)
# =============================================================================
print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_conf = confusion_matrix(y_test, xgb_preds)

print("\nXGBoost Results:")
print("Accuracy: {:.2f}%".format(xgb_accuracy * 100))
print("Confusion Matrix:")
print(xgb_conf)

# Visualize the XGBoost confusion matrix counts
plt.figure()
xgb_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
plt.bar(xgb_labels, xgb_conf.ravel())
plt.title("XGBoost: Confusion Matrix Counts")
plt.ylabel("Count")
plt.show()

# =============================================================================
# Model Comparison
# =============================================================================
print("\nModel Comparison:")
print("Logistic Regression Accuracy: {:.2f}%".format(lr_accuracy * 100))
print("XGBoost Accuracy: {:.2f}%".format(xgb_accuracy * 100))

# Plot a bar chart comparing the accuracies
models = ['Logistic Regression', 'XGBoost']
accuracies = [lr_accuracy * 100, xgb_accuracy * 100]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison: Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.2f}%", ha='center')
plt.show()
