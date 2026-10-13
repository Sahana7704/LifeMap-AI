"""
Utility functions for metrics, parallel bootstrap confidence intervals, and experiment logging.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from joblib import Parallel, delayed


def calculate_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def _fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    rank = np.argsort(order) + 1
    n_pos = np.count_nonzero(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((np.sum(rank[y_true == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _single_bootstrap_iter(seed: int, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, sample_size: int):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    idx = rng.randint(0, n, size=sample_size)
    b_true = y_true[idx]
    b_prob = y_prob[idx]
    
    n_pos = np.count_nonzero(b_true)
    if n_pos == 0 or n_pos == sample_size:
        return None
        
    b_pred = (b_prob >= threshold).astype(int)
    acc = float(np.mean(b_pred == b_true))
    rec = float(np.sum((b_pred == 1) & (b_true == 1)) / n_pos)
    auc = _fast_auc(b_true, b_prob)
    pr = float(average_precision_score(b_true, b_prob))
    return acc, rec, auc, pr


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for accuracy, recall, roc-auc, and pr-auc across CPU cores."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    sample_size = min(n, 50000) if n > 50000 else n

    results = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(_single_bootstrap_iter)(seed + i, y_true, y_prob, threshold, sample_size)
        for i in range(n_bootstraps)
    )

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {k: {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0} for k in ["accuracy", "recall", "roc_auc", "pr_auc"]}

    accs = [r[0] for r in valid_results]
    recs = [r[1] for r in valid_results]
    aucs = [r[2] for r in valid_results]
    prs = [r[3] for r in valid_results]

    def get_ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "accuracy": get_ci(accs),
        "recall": get_ci(recs),
        "roc_auc": get_ci(aucs),
        "pr_auc": get_ci(prs),
    }


def log_experiment(
    csv_path: Path,
    experiment_data: dict,
) -> None:
    """Append experiment details to experiments.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([experiment_data])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)
