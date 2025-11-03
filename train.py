# train.py
"""
AI for Clean Water (SDG 6)
Train and export a RandomForest model for water potability prediction.
Expected dataset filename: water_potability.csv (same folder)
"""

import os
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(path="water_potability.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place 'water_potability.csv' in project folder."
        )
    df = pd.read_csv(path)
    return df


def build_pipeline(random_state=42):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline


def train_and_save(df, model_out="water_quality_model.pkl", test_size=0.2, random_state=42):
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and metadata
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": list(X.columns)
    }, model_out)
    print(f"✅ Model saved successfully to {model_out}")
    return pipeline, acc


def main():
    parser = argparse.ArgumentParser(description="Train water potability model")
    parser.add_argument("--data", default="water_potability.csv", help="Path to CSV dataset")
    parser.add_argument("--out", default="water_quality_model.pkl", help="Output model file (.pkl)")
    args = parser.parse_args()

    df = load_data(args.data)
    train_and_save(df, model_out=args.out)


if __name__ == "__main__":
    main()
