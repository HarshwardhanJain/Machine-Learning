import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load Titanic dataset (replace with actual path if needed)
df = pd.read_csv(r"Lab Work - 04_02_25\Titanic_Dataset.csv")

# Drop non-numeric and non-relevant columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Encode categorical variables
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'].fillna('S', inplace=True)
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Fill missing values
df.fillna(df.median(), inplace=True)

# Define features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# Standardize features for better performance
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 1. Variance Threshold (removes low variance features)
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X_scaled)
var_features = X.columns[selector.get_support()]

# 2. Chi-Square Test
chi2_selector = SelectKBest(score_func=chi2, k=5)
X_chi2 = chi2_selector.fit_transform(X_scaled, y)
chi2_features = X.columns[chi2_selector.get_support()]

# 3. Mutual Information
mi_selector = SelectKBest(score_func=mutual_info_classif, k=5)
X_mi = mi_selector.fit_transform(X_scaled, y)
mi_features = X.columns[mi_selector.get_support()]

# 4. ANOVA F-test
anova_selector = SelectKBest(score_func=f_classif, k=5)
X_anova = anova_selector.fit_transform(X_scaled, y)
anova_features = X.columns[anova_selector.get_support()]

# 5. Correlation Coefficient
corr_matrix = df.corr()
corr_target = corr_matrix['Survived'].abs().sort_values(ascending=False)
selected_corr_features = corr_target.index[1:6]  # Top 5 features excluding 'Survived'
X_corr = df[selected_corr_features]

print("Selected features from each method:")
print("Variance Threshold:", var_features.tolist())
print("Chi-Square:", chi2_features.tolist())
print("Mutual Information:", mi_features.tolist())
print("ANOVA F-test:", anova_features.tolist())
print("Correlation Coefficient:", selected_corr_features.tolist())

# Visualization using Matplotlib and Seaborn
plt.figure(figsize=(12, 6))
methods = ['Variance Threshold', 'Chi-Square', 'Mutual Information', 'ANOVA F-test', 'Correlation Coefficient']
selected_features = [var_features, chi2_features, mi_features, anova_features, selected_corr_features]
patterns = ["//", "\\", "-|", "+", "o"]  # Different hatch patterns
colors = sns.color_palette("husl", len(methods))

for i, (method, features) in enumerate(zip(methods, selected_features), 1):
    plt.subplot(2, 3, i)
    bars = sns.barplot(x=features, y=range(len(features)), palette=[colors[i-1]]*len(features))
    for bar in bars.patches:
        bar.set_hatch(patterns[i-1 % len(patterns)])
    plt.xticks(rotation=45)
    plt.title(method)

plt.tight_layout()
plt.show()
