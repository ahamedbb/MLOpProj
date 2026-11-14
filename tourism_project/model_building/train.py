import os
import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

mlflow.xgboost.autolog()

# Load data splits
X_train = pd.read_csv("Xtrain.csv" if os.path.exists("Xtrain.csv") else "tourism_project/data/Xtrain.csv")
X_test = pd.read_csv("Xtest.csv" if os.path.exists("Xtest.csv") else "tourism_project/data/Xtest.csv")
y_train = pd.read_csv("ytrain.csv" if os.path.exists("ytrain.csv") else "tourism_project/data/ytrain.csv").values.ravel()
y_test = pd.read_csv("ytest.csv" if os.path.exists("ytest.csv") else "tourism_project/data/ytest.csv").values.ravel()

# Explicitly cast object columns to category dtype after CSV round-trip
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

with mlflow.start_run() as run:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        enable_categorical=True  # <--- Required for native categorical handling in XGBoost
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Training Complete. Test Accuracy: {acc:.4f}")

    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")
