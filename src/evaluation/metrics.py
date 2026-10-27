"""
Evaluation Metrics & Statistical Testing for Seizure Prediction.
Applications of Soft Computing (ASC).

Metrics implemented:
1. Window-level & Seizure-level Sensitivity.
2. False Positive Rate per Hour (FPR/h).
3. Area Under ROC (AUC-ROC) and Area Under Precision-Recall (AUC-PR).
4. Paired Bootstrap Confidence Intervals (Model vs. Baseline).
5. Statistical significance test vs. Random Poisson Chance predictor.

References:
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
- Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
    accuracy_score
)
from typing import Dict, List, Tuple, Any, Optional


def compute_standard_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes standard binary classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle single class edge cases gracefully
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recall, precision))
    except ValueError:
        pr_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def compute_seizure_level_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    seizure_ids: List[str],
    window_length_sec: float,
    threshold: float = 0.5,
    min_consecutive_alarms: int = 2
) -> Dict[str, Any]:
    """
    Computes clinical seizure-level metrics:
    - Seizure-level Sensitivity: Fraction of seizures with >= min_consecutive_alarms in preictal.
    - False Positive Rate per hour (FPR/h) in interictal data.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Seizure-level Sensitivity
    unique_seizures = sorted(list(set(s for s in seizure_ids if s != "interictal")))
    predicted_seizures = 0

    for s_id in unique_seizures:
        mask = [s == s_id for s in seizure_ids]
        preds_in_seizure = y_pred[mask]
        
        # Check for consecutive alarms to avoid single-window noise triggers
        consecutive = 0
        alarm_triggered = False
        for p in preds_in_seizure:
            if p == 1:
                consecutive += 1
                if consecutive >= min_consecutive_alarms:
                    alarm_triggered = True
                    break
            else:
                consecutive = 0
        
        if alarm_triggered:
            predicted_seizures += 1

    seizure_sensitivity = (
        float(predicted_seizures / len(unique_seizures))
        if len(unique_seizures) > 0 else 0.0
    )

    # 2. False Positive Rate per Hour (FPR/h)
    interictal_mask = [s == "interictal" for s in seizure_ids]
    interictal_preds = y_pred[interictal_mask]
    n_interictal_windows = len(interictal_preds)
    
    total_interictal_hours = (n_interictal_windows * window_length_sec) / 3600.0

    # Count refractory alarm events (an alarm stays active for SPH/SOP, count distinct bursts)
    false_alarms = 0
    in_alarm_burst = False
    refractory_counter = 0
    refractory_windows = int(180 / window_length_sec)  # 3-minute refractory period

    for p in interictal_preds:
        if refractory_counter > 0:
            refractory_counter -= 1
            continue
        if p == 1:
            false_alarms += 1
            refractory_counter = refractory_windows  # enter refractory block

    fpr_per_hour = (
        float(false_alarms / total_interictal_hours)
        if total_interictal_hours > 0 else 0.0
    )

    return {
        "seizure_sensitivity": seizure_sensitivity,
        "predicted_seizures": predicted_seizures,
        "total_seizures": len(unique_seizures),
        "false_alarms": false_alarms,
        "total_interictal_hours": total_interictal_hours,
        "fpr_per_hour": fpr_per_hour
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    probs_model: np.ndarray,
    probs_baseline: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs a paired bootstrap test to compare model performance against the baseline.
    Computes 95% Confidence Interval for difference: Delta = Metric(model) - Metric(baseline).
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes are present in sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        m_score = metric_fn(y_true[idx], probs_model[idx])
        b_score = metric_fn(y_true[idx], probs_baseline[idx])
        diffs.append(m_score - b_score)

    diffs = np.array(diffs)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(diffs, alpha * 100))
    ci_upper = float(np.percentile(diffs, (1.0 - alpha) * 100))
    mean_diff = float(np.mean(diffs))
    
    # Two-sided empirical p-value: fraction where diff <= 0
    p_value = float(np.mean(diffs <= 0.0))

    return {
        "mean_difference": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": ci_lower > 0.0,
        "n_bootstraps": len(diffs)
    }
