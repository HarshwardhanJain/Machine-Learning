# experiment_18_wine.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Load and scale Wine dataset
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 1. Filter Method ===
print("\n--- Filter Method (SelectKBest with ANOVA) ---")
selector_filter = SelectKBest(score_func=f_classif, k=8)
X_filter = selector_filter.fit_transform(X_train, y_train)
selected_features_filter = np.array(feature_names)[selector_filter.get_support()]
model_filter = LogisticRegression(max_iter=1000)
acc_filter = cross_val_score(model_filter, X_filter, y_train, cv=5).mean()
print("Selected Features:", selected_features_filter)
print("Cross-Validated Accuracy:", round(acc_filter * 100, 2), "%")

# === 2. Wrapper Method ===
print("\n--- Wrapper Method (RFE with Logistic Regression) ---")
estimator = LogisticRegression(max_iter=1000)
selector_wrapper = RFE(estimator, n_features_to_select=8)
X_wrapper = selector_wrapper.fit_transform(X_train, y_train)
selected_features_wrapper = np.array(feature_names)[selector_wrapper.get_support()]
acc_wrapper = cross_val_score(estimator, X_wrapper, y_train, cv=5).mean()
print("Selected Features:", selected_features_wrapper)
print("Cross-Validated Accuracy:", round(acc_wrapper * 100, 2), "%")

# === 3. Embedded Method ===
print("\n--- Embedded Method (Random Forest Feature Importance) ---")
model_embedded = RandomForestClassifier(random_state=42)
model_embedded.fit(X_train, y_train)
importances = model_embedded.feature_importances_
indices = np.argsort(importances)[-8:]
X_embedded = X_train[:, indices]
selected_features_embedded = np.array(feature_names)[indices]
acc_embedded = cross_val_score(model_embedded, X_embedded, y_train, cv=5).mean()
print("Selected Features:", selected_features_embedded)
print("Cross-Validated Accuracy:", round(acc_embedded * 100, 2), "%")

# === 4. Hybrid Method === (Filter + Wrapper)
print("\n--- Hybrid Method (SelectKBest + RFE) ---")
# First filter
X_filter_hybrid = SelectKBest(f_classif, k=10).fit_transform(X_train, y_train)
# Then wrapper
selector_hybrid = RFE(LogisticRegression(max_iter=1000), n_features_to_select=8)
X_hybrid = selector_hybrid.fit_transform(X_filter_hybrid, y_train)
acc_hybrid = cross_val_score(LogisticRegression(), X_hybrid, y_train, cv=5).mean()
print("Cross-Validated Accuracy:", round(acc_hybrid * 100, 2), "%")

# === 5. Dimensionality Reduction (PCA) ===
print("\n--- Dimensionality Reduction (PCA) ---")
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_train)
acc_pca = cross_val_score(LogisticRegression(max_iter=1000), X_pca, y_train, cv=5).mean()
print("Cross-Validated Accuracy with PCA:", round(acc_pca * 100, 2), "%")

# === Summary Table ===
print("\n=== Accuracy Comparison Summary ===")
summary = {
    'Method': ['Filter (SelectKBest)', 'Wrapper (RFE)', 'Embedded (RF)', 'Hybrid (Filter+Wrapper)', 'PCA (Dim. Red.)'],
    'Accuracy (%)': [round(acc_filter * 100, 2), round(acc_wrapper * 100, 2),
                     round(acc_embedded * 100, 2), round(acc_hybrid * 100, 2), round(acc_pca * 100, 2)]
}
df_summary = pd.DataFrame(summary)
print(df_summary)
