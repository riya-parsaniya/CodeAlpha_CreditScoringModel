"""
train.py
--------
Train Logistic Regression, Decision Tree, and Random Forest models
for the CodeAlpha Credit Scoring Model project.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")


def get_models() -> dict:
    """Return a dictionary of baseline models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
    }


def train_all_models(X_train, y_train) -> dict:
    """
    Train all three models and return a dict of fitted models.
    Also prints 5-fold cross-validation F1 scores.
    """
    models = get_models()
    trained = {}

    for name, model in models.items():
        print(f"\n[train] Training: {name} ...")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"        CV F1 scores: {np.round(cv_scores, 3)}")
        print(f"        Mean CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        trained[name] = model

    return trained


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Run a small GridSearchCV on Random Forest to find better hyperparameters.
    Returns the best estimator.
    """
    print("\n[tune] Tuning Random Forest with GridSearchCV ...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"[tune] Best params : {grid_search.best_params_}")
    print(f"[tune] Best ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def save_model(model, filename: str):
    """Save a trained model to outputs/models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"[save] Model saved → {path}")


def load_model(filename: str):
    """Load a saved model from outputs/models/."""
    path = os.path.join(MODELS_DIR, filename)
    model = joblib.load(path)
    print(f"[load] Model loaded ← {path}")
    return model
