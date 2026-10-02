# Week 2: Supervised Machine Learning Models

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())

# --------------------------------------------------
# 2. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# --------------------------------------------------
# 3. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 4. Create Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# --------------------------------------------------
# 5. Train and Evaluate Models
# --------------------------------------------------

results = []

for name, model in models.items():

    # KNN and Logistic Regression benefit from scaling
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted"
    )
    recall = recall_score(
        y_test, y_pred, average="weighted"
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n------------------------------")
    print(name)
    print("------------------------------")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

# --------------------------------------------------
# 6. Compare Models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df)

# Save results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nResults saved to model_comparison.csv")