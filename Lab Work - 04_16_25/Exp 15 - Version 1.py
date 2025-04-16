# compare_xgboost_vs_baseline.py
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Traditional models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# XGBoost
from xgboost import XGBClassifier

def load_titanic_data():
    df = sns.load_dataset("titanic").dropna(subset=['age', 'embarked', 'sex', 'fare', 'pclass'])
    df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'])
    df['embarked'] = le_embarked.fit_transform(df['embarked'])
    
    X = df.drop('survived', axis=1)
    y = df['survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_iris_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_models(name, X_train, X_test, y_train, y_test, baseline_model, xgb_model):
    baseline_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    baseline_pred = baseline_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    baseline_acc = accuracy_score(y_test, baseline_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    print(f"\n=== {name} Dataset Accuracy Comparison ===")
    print(f"Baseline Model ({type(baseline_model).__name__}) Accuracy: {baseline_acc * 100:.2f}%")
    print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
    return baseline_acc, xgb_acc

def main():
    # Titanic: Logistic Regression vs. XGBoost
    X_train_t, X_test_t, y_train_t, y_test_t = load_titanic_data()
    titanic_baseline = LogisticRegression(max_iter=1000)
    titanic_xgb = XGBClassifier(eval_metric='logloss')
    evaluate_models("Titanic", X_train_t, X_test_t, y_train_t, y_test_t, titanic_baseline, titanic_xgb)

    # Iris: KNN vs. XGBoost
    X_train_i, X_test_i, y_train_i, y_test_i = load_iris_data()
    iris_baseline = KNeighborsClassifier(n_neighbors=3)
    iris_xgb = XGBClassifier(eval_metric='mlogloss')
    evaluate_models("Iris", X_train_i, X_test_i, y_train_i, y_test_i, iris_baseline, iris_xgb)

if __name__ == "__main__":
    main()
