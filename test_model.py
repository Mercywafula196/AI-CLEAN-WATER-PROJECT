# test_model.py
"""
Automated tests for AI for Clean Water (SDG 6)
Ensures model loads correctly, predicts without errors, and outputs valid probabilities.
"""

import joblib
import pandas as pd
import numpy as np
import pytest

MODEL_PATH = "water_quality_model.pkl"

def load_model():
    """Load the trained model bundle."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"]

def test_model_load():
    """Test if model loads correctly."""
    model = load_model()
    assert model is not None, "Model failed to load."

def test_prediction_shape():
    """Test if model returns predictions of correct shape."""
    model = load_model()
    # Create dummy input sample
    sample = pd.DataFrame([{
        "pH": 7.0,
        "Hardness": 200.0,
        "Solids": 15000.0,
        "Chloramines": 7.0,
        "Sulfate": 300.0,
        "Conductivity": 400.0,
        "Organic_carbon": 10.0,
        "Trihalomethanes": 80.0,
        "Turbidity": 4.0
    }])
    preds = model.predict(sample)
    assert preds.shape == (1,), f"Unexpected prediction shape: {preds.shape}"

def test_prediction_values():
    """Ensure model outputs valid classification labels (0 or 1)."""
    model = load_model()
    sample = pd.DataFrame([{
        "pH": 7.5,
        "Hardness": 250.0,
        "Solids": 10000.0,
        "Chloramines": 5.0,
        "Sulfate": 350.0,
        "Conductivity": 500.0,
        "Organic_carbon": 12.0,
        "Trihalomethanes": 70.0,
        "Turbidity": 3.0
    }])
    preds = model.predict(sample)
    assert preds[0] in [0, 1], f"Invalid prediction value: {preds[0]}"

def test_prediction_probabilities():
    """Ensure model outputs valid probability scores (between 0 and 1)."""
    model = load_model()
    sample = pd.DataFrame([{
        "pH": 6.5,
        "Hardness": 180.0,
        "Solids": 12000.0,
        "Chloramines": 6.5,
        "Sulfate": 320.0,
        "Conductivity": 380.0,
        "Organic_carbon": 9.5,
        "Trihalomethanes": 85.0,
        "Turbidity": 4.2
    }])
    probs = model.predict_proba(sample)
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range."
    assert abs(probs[0].sum() - 1) < 1e-6, "Probabilities do not sum to 1."
