# diabetes_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Split data into features (X) and target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy: {rf_accuracy}")

# Train and evaluate SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
print(f"SVM Accuracy: {svm_accuracy}")

# Save both models separately
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
print("Both models saved as rf_model.pkl and svm_model.pkl")
