import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_logistic_pipeline(C=1):
    """Create an unfitted logistic regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=1000)),
    ])


def make_xgboost_model(**parameters):
    """Create an unfitted XGBoost model with reproducible defaults."""
    settings = {
        "n_jobs": 4,
        "random_state": 0,
    }

    settings.update(parameters)

    return XGBClassifier(**settings)


def calibration_inputs(probabilities):
    """Convert probabilities to the inputs used by the calibrator."""
    return np.log(np.clip(probabilities, 1e-10, 1))


def fit_calibrated_model(
    pipeline,
    X,
    y,
    fit_indices,
    calibration_indices,
):
    """Fit the base model and calibrator on separate earlier data."""
    base_model = clone(pipeline)
    base_model.fit(X.iloc[fit_indices], y.iloc[fit_indices])

    calibration_probs = base_model.predict_proba(
        X.iloc[calibration_indices]
    )

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(
        calibration_inputs(calibration_probs),
        y.iloc[calibration_indices],
    )

    return base_model, calibrator


def predict_calibrated(base_model, calibrator, X):
    """Return recalibrated outcome probabilities."""
    probabilities = base_model.predict_proba(X)

    return calibrator.predict_proba(
        calibration_inputs(probabilities)
    )