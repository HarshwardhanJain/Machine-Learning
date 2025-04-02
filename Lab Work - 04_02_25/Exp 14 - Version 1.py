import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
data = pd.read_csv(r"Lab Work - 04_02_25\Titanic_Dataset.csv")

# Drop unnecessary columns
data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Handle missing values
data = data.assign(
    Age=data["Age"].fillna(data["Age"].median()),
    Embarked=data["Embarked"].fillna(data["Embarked"].mode()[0])
)

# Encode categorical variables
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model on all features (before elimination)
model_all_features = LogisticRegression()
model_all_features.fit(X_train, y_train)
accuracy_before_elimination = model_all_features.score(X_test, y_test)

# Apply Recursive Feature Elimination
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Train model on selected features (after elimination)
model.fit(X_train_rfe, y_train)
accuracy_after_elimination = model.score(X_test_rfe, y_test)

# Display selected features
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features.tolist())

# Display accuracies
print("Accuracy Before Elimination:", accuracy_before_elimination)
print("Accuracy After Elimination:", accuracy_after_elimination)

# Debugging Steps

# 1. Check coefficients before and after elimination
print("\nCoefficients (Before Elimination):", model_all_features.coef_)
print("Coefficients (After Elimination):", model.coef_)

# 2. Check feature rankings from RFE
print("\nFeature Rankings:", rfe.ranking_)

# 3. Check correlation between eliminated feature and target
eliminated_feature = [col for col in X.columns if col not in selected_features][0]
print(f"\nCorrelation ({eliminated_feature} vs Survived):", data[eliminated_feature].corr(data["Survived"]))

# 4. Experiment with fewer features
rfe = RFE(model, n_features_to_select=4)  # Select only 4 features
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
model.fit(X_train_rfe, y_train)
print("\nAccuracy After Elimination (4 Features):", model.score(X_test_rfe, y_test))

# 5. Use a different model (Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("\nRF Accuracy (Before Elimination):", rf_model.score(X_test, y_test))

rf_model.fit(X_train_rfe, y_train)
print("RF Accuracy (After Elimination):", rf_model.score(X_test_rfe, y_test))
