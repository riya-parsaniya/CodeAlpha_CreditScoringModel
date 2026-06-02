"""
evaluate.py
-----------
Model evaluation module (SAFE + RUNNABLE)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

PLOTS_DIR = os.path.join("outputs", "plots")


# ---------------- METRICS ----------------
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else 0
    }


# ---------------- REPORT ----------------
def print_report(model, name, X_test, y_test):
    print(f"\n===== {name} =====")
    print(classification_report(y_test, model.predict(X_test)))


# ---------------- MAIN EVAL ----------------
def evaluate_all_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        print_report(model, name, X_test, y_test)
        results[name] = get_metrics(model, X_test, y_test)

    return results


# ---------------- CONFUSION MATRIX ----------------
def plot_confusion_matrices(models, X_test, y_test):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


# ---------------- ROC CURVE ----------------
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)

            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()


# ---------------- FEATURE IMPORTANCE ----------------
def plot_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature importance")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), np.array(feature_names)[idx], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


# ---------------- METRICS PLOT ----------------
def plot_metrics_comparison(results):
    import pandas as pd

    df = pd.DataFrame(results).T

    df.plot(kind="bar", figsize=(10, 5))
    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.show()