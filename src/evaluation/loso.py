"""
Leave-One-Seizure-Out (LOSO) Cross-Validation Harness.
Applications of Soft Computing (ASC).

Provides a unified, leak-free LOSO evaluation loop that:
  1. Iterates over all K seizures as held-out test folds.
  2. Accepts any model with a fit(X, y) / predict_proba(X) API.
  3. Computes window-level and seizure-level metrics per fold.
  4. Aggregates mean ± std across folds.
  5. Runs paired bootstrap comparison against the Logistic Regression baseline.

The LOSO protocol strictly follows:
  - Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
  - Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
  - Ibrahim et al. (2023), Front Signal Process (DOI: 10.3389/frsip.2023.1175305)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import (
    compute_standard_metrics,
    compute_seizure_level_metrics,
    paired_bootstrap_test
)
from src.config import default_config, EEGConfig

logger = logging.getLogger(__name__)


def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    seizure_ids: List[str],
    model,
    config: Optional[EEGConfig] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full Leave-One-Seizure-Out cross-validation for a given model.

    For each distinct preictal seizure block:
      - Test set: all windows from that seizure (preictal) + proportional interictal slice.
      - Train set: all remaining preictal windows from other seizures + remaining interictal.

    Args:
        X: Feature matrix (N, n_features) for biomarker models, or
           raw EEG tensor (N, n_channels, n_samples) for EEGNet.
        y: Binary labels (N,) — 1=preictal, 0=interictal.
        seizure_ids: List of per-window seizure identifiers.
        model: Any model with .fit(X, y) and .predict_proba(X) -> (N, 2).
        config: EEGConfig instance (uses defaults if None).
        verbose: Log per-fold metrics if True.

    Returns:
        Dictionary with:
          - 'fold_results': list of per-fold metric dicts.
          - 'aggregated': mean ± std across folds.
          - 'all_y_true': concatenated true labels across all folds.
          - 'all_y_prob': concatenated predicted probabilities across all folds.
          - 'all_seizure_ids': concatenated seizure IDs for seizure-level metrics.
    """
    cfg = config or default_config

    # Unique preictal seizure IDs (excludes 'interictal')
    unique_seizures = sorted([s for s in set(seizure_ids) if s != 'interictal'])
    K = len(unique_seizures)

    if K < 2:
        raise ValueError(
            f"LOSO requires at least 2 distinct seizures; found {K}. "
            "Consider using a patient with more seizure events."
        )

    logger.info(f"Running LOSO CV: K={K} folds, model={type(model).__name__}")

    X = np.array(X)
    y = np.array(y)
    seizure_ids = list(seizure_ids)

    interictal_mask = np.array([s == 'interictal' for s in seizure_ids])
    n_interictal = int(interictal_mask.sum())

    fold_results = []
    all_y_true, all_y_prob, all_sids = [], [], []

    for fold_idx, held_out_seizure in enumerate(unique_seizures):
        # ---- Build test set ----
        preictal_test_mask = np.array([s == held_out_seizure for s in seizure_ids])
        n_preictal_test = int(preictal_test_mask.sum())

        # Proportional interictal test slice: same ratio as training would see
        # We reserve 1/K of the interictal pool for the test fold
        interictal_per_fold = max(1, n_interictal // K)
        interictal_idx = np.where(interictal_mask)[0]
        # Deterministic slice to avoid data leakage via random shuffling
        test_interictal_idx = interictal_idx[
            fold_idx * interictal_per_fold: (fold_idx + 1) * interictal_per_fold
        ]

        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[preictal_test_mask] = True
        test_mask[test_interictal_idx] = True

        # ---- Build training set ----
        # Remaining preictal (all other seizures) + remaining interictal
        other_preictal_mask = (y == 1) & ~preictal_test_mask
        remaining_interictal_mask = interictal_mask.copy()
        remaining_interictal_mask[test_interictal_idx] = False
        train_mask = other_preictal_mask | remaining_interictal_mask

        if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
            logger.warning(f"Fold {fold_idx}: empty train or test set, skipping.")
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        sids_test = [seizure_ids[i] for i in np.where(test_mask)[0]]

        if len(np.unique(y_train)) < 2:
            logger.warning(f"Fold {fold_idx}: training set has only one class, skipping.")
            continue

        # ---- Fit and evaluate ----
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        std_metrics = compute_standard_metrics(y_test, y_prob)
        seizure_metrics = compute_seizure_level_metrics(
            y_test, y_prob, sids_test,
            window_length_sec=cfg.window_length_sec
        )

        fold_metrics = {
            'fold': fold_idx,
            'held_out_seizure': held_out_seizure,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            **std_metrics,
            **seizure_metrics
        }

        fold_results.append(fold_metrics)

        all_y_true.append(y_test)
        all_y_prob.append(y_prob)
        all_sids.extend(sids_test)

        if verbose:
            logger.info(
                f"Fold {fold_idx+1}/{K} | Seizure: {held_out_seizure} | "
                f"AUC={std_metrics['roc_auc']:.4f} | "
                f"Sens={seizure_metrics['seizure_sensitivity']*100:.1f}% | "
                f"FPR/h={seizure_metrics['fpr_per_hour']:.3f}"
            )

    # ---- Aggregate across folds ----
    all_y_true_arr = np.concatenate(all_y_true)
    all_y_prob_arr = np.concatenate(all_y_prob)

    aggregated = _aggregate_fold_results(fold_results)

    return {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'all_y_true': all_y_true_arr,
        'all_y_prob': all_y_prob_arr,
        'all_seizure_ids': all_sids
    }


def _aggregate_fold_results(fold_results: List[Dict]) -> Dict[str, Any]:
    """Compute mean and std for scalar metrics across LOSO folds."""
    if not fold_results:
        return {}

    scalar_keys = [
        'roc_auc', 'pr_auc', 'sensitivity', 'specificity', 'f1_score',
        'accuracy', 'seizure_sensitivity', 'fpr_per_hour',
        'predicted_seizures', 'total_seizures', 'false_alarms',
        'total_interictal_hours'
    ]

    aggregated = {}
    for k in scalar_keys:
        values = [f[k] for f in fold_results if k in f]
        if values:
            aggregated[f'{k}_mean'] = float(np.mean(values))
            aggregated[f'{k}_std'] = float(np.std(values))

    return aggregated


def compare_models_bootstrap(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    baseline_name: str = 'Baseline (LogReg)',
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute paired bootstrap 95% CI for Delta AUC = Model - Baseline for all models.

    Args:
        y_true: True binary labels.
        model_probs: Dict mapping model name -> predicted probability array (N,).
        baseline_name: Key in model_probs that is the reference baseline.
        n_bootstraps: Bootstrap replications.
        random_state: RNG seed for reproducibility.

    Returns:
        Dict mapping model_name -> bootstrap test result dict.
    """
    if baseline_name not in model_probs:
        raise ValueError(f"Baseline '{baseline_name}' not found in model_probs keys.")

    baseline_probs = model_probs[baseline_name]
    results = {}

    for name, probs in model_probs.items():
        if name == baseline_name:
            continue
        bootstrap_res = paired_bootstrap_test(
            y_true=y_true,
            probs_model=probs,
            probs_baseline=baseline_probs,
            metric_fn=roc_auc_score,
            n_bootstraps=n_bootstraps,
            random_state=random_state
        )
        results[name] = bootstrap_res
        logger.info(
            f"Bootstrap [{name} vs {baseline_name}]: "
            f"ΔAUCμ={bootstrap_res['mean_difference']:+.4f} "
            f"95%CI=[{bootstrap_res['ci_lower']:+.4f}, {bootstrap_res['ci_upper']:+.4f}] "
            f"p={bootstrap_res['p_value']:.4f} "
            f"sig={'YES' if bootstrap_res['is_significant'] else 'NO'}"
        )

    return results
