import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Small educational dataset
data = {
    "login_attempts": [2, 3, 1, 4, 2, 20, 30, 15, 25, 40, 3, 2, 18, 35, 1, 5],
    "failed_attempts": [0, 1, 0, 1, 0, 18, 27, 13, 22, 38, 1, 0, 16, 32, 0, 2],
    "time_interval": [60, 45, 90, 50, 70, 2, 1, 3, 2, 1, 55, 80, 3, 1, 100, 40],
    "suspicious_source": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, "model.pkl")
print("\nModel saved as model.pkl")
