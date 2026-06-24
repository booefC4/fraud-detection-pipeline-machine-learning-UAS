"""
evaluate.py
Metrik untuk klasifikasi sangat imbalanced + plot.

Metrik utama:
- ROC-AUC      : metrik resmi kompetisi IEEE-CIS, robust terhadap threshold.
- PR-AUC (AP)  : lebih informatif saat positif jarang (~3.5%).
Pada threshold optimal (maks F1) juga dilaporkan precision/recall/F1 + confusion matrix.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score,
)


def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def compute_metrics(y_true, y_prob, threshold=None) -> dict:
    if threshold is None:
        threshold = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_plots(y_true, y_prob, out_prefix, threshold=None):
    """Simpan ROC, PR curve, dan confusion matrix. Return list path file."""
    if threshold is None:
        threshold = best_f1_threshold(y_true, y_prob)
    paths = []

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    p = f"{out_prefix}_roc.png"; plt.savefig(p, dpi=120); plt.close(); paths.append(p)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(); plt.tight_layout()
    p = f"{out_prefix}_pr.png"; plt.savefig(p, dpi=120); plt.close(); paths.append(p)

    # Confusion matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Legit", "Fraud"]); plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix @ thr={threshold:.3f}"); plt.tight_layout()
    p = f"{out_prefix}_cm.png"; plt.savefig(p, dpi=120); plt.close(); paths.append(p)

    return paths


def pretty_print(metrics: dict, title="METRICS"):
    print(f"\n=== {title} ===")
    print(json.dumps(metrics, indent=2))
