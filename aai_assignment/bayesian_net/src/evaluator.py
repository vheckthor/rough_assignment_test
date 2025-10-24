from typing import Dict, List, Tuple
import time
import numpy as np
from sklearn import metrics, calibration

from bayesian_net.src.inference.enumeration import predict_distribution
from bayesian_net.src.cpt import BNModel


def _to_binary_value(val) -> int:
    s = str(val).lower()
    if s in ["1", "yes", "true"]:
        return 1
    return 0


def evaluate_model(model: BNModel, test_rows: List[Dict[str, str]], target: str) -> Tuple[Dict[str, float], float]:
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    start = time.time()
    for row in test_rows:
        true_val = row[target]
        y_true.append(_to_binary_value(true_val))
        evidence = {k: v for k, v in row.items() if k != target}
        dist = predict_distribution(model, evidence, target)
        # pick probability of positive class
        p1 = None
        if '1' in dist:
            p1 = float(dist['1'])
        elif 1 in dist:
            p1 = float(dist[1])
        elif 'yes' in dist:
            p1 = float(dist['yes'])
        else:
            # fallback: choose the max as positive
            p1 = float(max(dist.values()))
        y_prob.append(p1)
        y_pred.append(1 if p1 >= 0.5 else 0)
    infer_time = time.time() - start

    # clean NaNs
    y_prob = [0.0 if (isinstance(p, float) and (np.isnan(p))) else p for p in y_prob]

    # metrics
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    brier = metrics.brier_score_loss(y_true, y_prob)

    P = np.asarray(y_true, dtype=float) + 1e-5
    Q = np.asarray(y_prob, dtype=float) + 1e-5
    kl = float(np.sum(P * np.log(P / Q)))

    # ECL
    n_bins = int(np.ceil(np.log2(len(y_true)) + 1)) if len(y_true) > 0 else 10
    prob_true, prob_pred = calibration.calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    nonempty = bin_counts > 0
    weights = bin_counts[nonempty] / np.sum(bin_counts[nonempty]) if np.sum(nonempty) > 0 else np.array([1.0])
    ecl = float(np.sum(weights * np.abs(prob_true - prob_pred))) if len(weights) == len(prob_true) else float(np.mean(np.abs(prob_true - prob_pred)))

    metrics_dict = {
        'bal_acc': float(bal_acc),
        'f1': float(f1),
        'auc': float(auc),
        'brier': float(brier),
        'kl': float(kl),
        'ecl': float(ecl),
    }
    return metrics_dict, infer_time

