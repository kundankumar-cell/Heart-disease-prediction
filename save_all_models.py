"""
Serializes the remaining trained models from HeartdiseaseFinal.ipynb
(Logistic Regression, SVM, Decision Tree) while preserving Naive Bayes and KNN.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

print("Loading dataset...")
df = pd.read_csv("heart.csv")
df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Use existing verified scaler.pkl
scaler = joblib.load("scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

models_to_save = {
    "logistic_regression_heart.pkl": LogisticRegression(),
    "svm_heart_model.pkl": SVC(probability=True, random_state=42),
    "decision_tree_heart.pkl": DecisionTreeClassifier(random_state=42),
}

print("\nTraining and evaluating models:")
for filename, model in models_to_save.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"  • {filename:<30} -> Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%")
    
    # Save model artifact
    joblib.dump(model, filename)
    print(f"    Saved: {filename}")

# Verify existing models remain untouched
print("\nVerifying existing models:")
nb = joblib.load("Naive Bayes_heart.pkl")
nb_pred = nb.predict(X_test_scaled)
print(f"  • Naive Bayes_heart.pkl           -> Acc: {accuracy_score(y_test, nb_pred)*100:.2f}%, F1: {f1_score(y_test, nb_pred)*100:.2f}%")

knn = joblib.load("knn_heart_model.pkl")
knn_pred = knn.predict(X_test_scaled)
print(f"  • knn_heart_model.pkl             -> Acc: {accuracy_score(y_test, knn_pred)*100:.2f}%, F1: {f1_score(y_test, knn_pred)*100:.2f}%")

print("\nAll 5 models are now safely serialized and ready for inference!")

