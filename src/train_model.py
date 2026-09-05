from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths
CSV_PATH = PROJECT_ROOT / "landmarks" / "landmarks.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sign_language_rf.pkl"


# 1. Load the landmark data
df = pd.read_csv(CSV_PATH)


# 2. Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]


# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 5. Train the model
model.fit(X_train, y_train)


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")


# 8. Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
