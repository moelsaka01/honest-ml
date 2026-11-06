"""A deliberately flawed churn model, used to demo `ml-roast`. DO NOT copy this code.

Run:  python skills/ml-roast/scripts/roast_scan.py examples/bad_example.py
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("examples/train.csv")
df = df.dropna()
df["signup_date"] = pd.to_datetime(df["signup_date"])

X = df.drop(columns=["churned", "signup_date"])
X = pd.get_dummies(X)
y = df["churned"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fitted on ALL rows, before the split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))  # no baseline, one seed, no CI
