# experiment_17_wine.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Step 1: Load and prepare the dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-Test Split (only for supervised)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === SUPERVISED MODEL ===
print("\n--- Supervised Learning: Logistic Regression ---")
supervised_model = LogisticRegression(max_iter=1000)
supervised_model.fit(X_train, y_train)
y_pred_sup = supervised_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_sup))
print("Classification Report:\n", classification_report(y_test, y_pred_sup))

# === UNSUPERVISED MODEL ===
print("\n--- Unsupervised Learning: K-Means Clustering ---")
unsupervised_model = KMeans(n_clusters=3, random_state=42)
y_pred_unsup = unsupervised_model.fit_predict(X_scaled)

# Evaluate clustering using ARI and silhouette score
ari = adjusted_rand_score(y, y_pred_unsup)
sil_score = silhouette_score(X_scaled, y_pred_unsup)

print("Adjusted Rand Index (vs. true labels):", ari)
print("Silhouette Score (cluster quality):", sil_score)

# Optional: Plot clusters using first 2 principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))

# Supervised Visualization
plt.subplot(1, 2, 1)
plt.title("True Wine Classes (Supervised)")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# Unsupervised Visualization
plt.subplot(1, 2, 2)
plt.title("KMeans Clusters (Unsupervised)")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_unsup, cmap='viridis', edgecolor='k')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.tight_layout()
plt.show()
