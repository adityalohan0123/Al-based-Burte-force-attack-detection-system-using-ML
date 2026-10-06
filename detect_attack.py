import joblib
import pandas as pd

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Model not found. Run: python src/train_model.py")
    raise SystemExit

print("AI-Based Brute Force Detection")
print("--------------------------------")

try:
    login_attempts = int(input("Number of login attempts: "))
    failed_attempts = int(input("Number of failed attempts: "))
    time_interval = float(input("Average time interval between attempts (seconds): "))
    suspicious_source = int(input("Suspicious source? Enter 1 for Yes, 0 for No: "))
except ValueError:
    print("Please enter valid numeric values.")
    raise SystemExit

sample = pd.DataFrame([{
    "login_attempts": login_attempts,
    "failed_attempts": failed_attempts,
    "time_interval": time_interval,
    "suspicious_source": suspicious_source
}])

prediction = model.predict(sample)[0]

if prediction == 1:
    print("\nResult: Possible Brute Force Attack detected.")
else:
    print("\nResult: Activity appears normal.")
