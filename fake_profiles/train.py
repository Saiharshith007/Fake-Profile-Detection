"""Train and compare fake-profile classifiers on a held-out test set.

Usage: python -m fake_profiles.train
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    get_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from fake_profiles.data import load_dataset

SEED = 42
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def build_models() -> dict[str, ClassifierMixin]:
    # Counts are heavy-tailed, so log-scale them for the distance/gradient-based models.
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "SVM": make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), SVC()),
        "Neural Network": make_pipeline(
            FunctionTransformer(np.log1p),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=SEED),
        ),
    }


def evaluate(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Fit each model on the training set and score it on the test set (fake = positive)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    roc_auc = get_scorer("roc_auc")
    rows = []
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        rows.append(
            {
                "model": name,
                "cv_accuracy": f"{cv_acc.mean():.3f} +/- {cv_acc.std():.3f}",
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc(model, X_test, y_test),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def plot_results(models, X_test, y_test, path: Path) -> None:
    """Save ROC curves and confusion matrices for the fitted models."""
    fig = Figure(figsize=(4.5 * (len(models) + 1), 4.2))
    axes = fig.subplots(1, len(models) + 1)
    for ax, (name, model) in zip(axes[1:], models.items(), strict=True):
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=axes[0])
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=["Genuine", "Fake"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    axes[0].set_title("ROC (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"{len(X_train)} train / {len(X_test)} test profiles, {y.mean():.0%} fake\n")

    models = build_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)
    print(results.to_string(float_format="%.3f"))

    REPORTS_DIR.mkdir(exist_ok=True)
    plot_results(models, X_test, y_test, REPORTS_DIR / "results.png")
    print(f"\nPlots saved to {REPORTS_DIR / 'results.png'}")


if __name__ == "__main__":
    main()
