import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert original data to a DataFrame for better visualization
original_df = pd.DataFrame(X, columns=iris.feature_names)
original_df['Target'] = y

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convert PCA-transformed data to a DataFrame
pca_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Target'] = y

# Plot side-by-side visualizations
plt.figure(figsize=(16, 6))

# Original dataset visualization
plt.subplot(1, 2, 1)
sns.scatterplot(x=original_df.iloc[:, 0], y=original_df.iloc[:, 1], hue=y, palette='viridis', alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Original Dataset')
plt.legend(labels=iris.target_names)

# PCA-transformed dataset visualization
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.legend(labels=iris.target_names)

plt.tight_layout()
plt.show()

# Explained variance ratio
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)
