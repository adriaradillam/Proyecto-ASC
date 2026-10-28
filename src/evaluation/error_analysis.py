"""
Checkpoint 3: In-Depth Error Analysis for EEG Seizure Prediction.
Applications of Soft Computing (ASC).

Implements the following analyses required by Week 5 milestone:

1. Per-Seizure False Alarm Breakdown
   - Which seizures are missed, which are correctly predicted?
   - What is the lead time (time from first alarm to seizure onset)?
   - Are false alarms clustered in specific recording segments?

2. Lead-Time Distribution Analysis
   - Distribution of alarm lead times across seizures and folds.
   - Clinically meaningful if lead time > SPH (5 min).

3. False Alarm Temporal Clustering
   - Are false positives uniformly distributed (Poisson) or clustered?
   - Inter-alarm interval analysis.

4. Biomarker Ablation Study
   - Which biomarker subsets drive performance?
   - Systematically remove one feature group at a time (delta/theta/alpha/etc).

5. Spatial Scalp Analysis
   - Which channels contribute most to discrimination?
   - Heatmap of per-channel importance across the 18 bipolar channels.

References:
    Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
    Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
    Ibrahim et al. (2023), Front Signal Process (DOI: 10.3389/frsip.2023.1175305)
    Jiang et al. (2023), Front Neurosci (DOI: 10.3389/fnins.2023.1191683)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import default_config, EEGConfig, FIGURES_DIR, TABLES_DIR
from src.evaluation.metrics import compute_standard_metrics, compute_seizure_level_metrics
from src.features.biomarkers import BiomarkerExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Per-Seizure Breakdown
# ---------------------------------------------------------------------------

def per_seizure_breakdown(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    seizure_ids: List[str],
    window_start_times: np.ndarray,   # seconds from start of recording per window
    config: Optional[EEGConfig] = None,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    For each seizure, compute:
      - Predicted (True/False)?
      - Lead time to seizure onset (seconds from first alarm in preictal period).
      - Number of alarms in preictal period.
      - Peak preictal probability.

    Args:
        y_true: Binary labels per window.
        y_prob: Predicted probability per window.
        seizure_ids: Per-window seizure identifier.
        window_start_times: Start time (seconds) of each window in recording.
        config: EEGConfig (uses defaults if None).
        threshold: Decision threshold for alarm.

    Returns:
        List of per-seizure result dicts.
    """
    cfg = config or default_config
    y_pred = (y_prob >= threshold).astype(int)
    unique_seizures = sorted([s for s in set(seizure_ids) if s != 'interictal'])
    sop_sec = cfg.sop_min * 60.0

    results = []
    for s_id in unique_seizures:
        mask = np.array([s == s_id for s in seizure_ids])
        preds_in_preictal = y_pred[mask]
        probs_in_preictal = y_prob[mask]
        times_in_preictal = window_start_times[mask]

        # Check if alarm was triggered
        alarm_indices = np.where(preds_in_preictal == 1)[0]
        alarm_triggered = len(alarm_indices) > 0

        # Lead time: time from first alarm to end of preictal window
        # The preictal window ends at t_onset - SPH, so lead time relative
        # to alarm = sop_sec - time_within_preictal_window
        if alarm_triggered:
            first_alarm_idx = alarm_indices[0]
            time_within_preictal = times_in_preictal[first_alarm_idx] - times_in_preictal[0]
            lead_time_sec = max(0.0, sop_sec - time_within_preictal)
        else:
            lead_time_sec = 0.0

        results.append({
            'seizure_id': s_id,
            'n_preictal_windows': int(mask.sum()),
            'predicted': bool(alarm_triggered),
            'lead_time_sec': float(lead_time_sec),
            'lead_time_min': float(lead_time_sec / 60.0),
            'n_alarms_in_preictal': int(len(alarm_indices)) if alarm_triggered else 0,
            'peak_preictal_prob': float(probs_in_preictal.max()),
            'mean_preictal_prob': float(probs_in_preictal.mean()),
        })

    return results


# ---------------------------------------------------------------------------
# 2. Lead-Time Distribution
# ---------------------------------------------------------------------------

def compute_lead_time_distribution(
    per_seizure_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute statistics of alarm lead times across correctly predicted seizures.

    Args:
        per_seizure_results: Output of per_seizure_breakdown().

    Returns:
        Dict with lead time statistics (mean, median, std, min, max in minutes).
    """
    predicted = [r for r in per_seizure_results if r['predicted']]
    if not predicted:
        return {'n_predicted': 0, 'lead_times_min': []}

    lead_times = np.array([r['lead_time_min'] for r in predicted])

    return {
        'n_predicted': len(predicted),
        'n_total': len(per_seizure_results),
        'lead_times_min': lead_times.tolist(),
        'mean_lead_time_min': float(lead_times.mean()),
        'median_lead_time_min': float(np.median(lead_times)),
        'std_lead_time_min': float(lead_times.std()),
        'min_lead_time_min': float(lead_times.min()),
        'max_lead_time_min': float(lead_times.max()),
        'frac_above_sph': float(np.mean(lead_times > 5.0))   # clinically meaningful if > 5 min (SPH)
    }


# ---------------------------------------------------------------------------
# 3. False Alarm Temporal Clustering
# ---------------------------------------------------------------------------

def analyze_false_alarm_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_start_times: np.ndarray
) -> Dict[str, Any]:
    """
    Analyse whether false alarms (FPs on interictal) are uniformly distributed
    (consistent with a Poisson process) or clustered.

    Computes:
      - Inter-alarm intervals (IAI) histogram
      - Coefficient of variation of IAI (CV = std/mean; CV ≈ 1 for Poisson, >1 for clustered)
      - Fano factor (variance / mean of alarm counts per hour; 1 = Poisson)

    Args:
        y_true: Binary labels.
        y_pred: Binary predictions.
        window_start_times: Start times of windows (seconds).

    Returns:
        Dict with clustering statistics.
    """
    interictal_mask = (y_true == 0)
    fp_mask = interictal_mask & (y_pred == 1)

    fp_times = window_start_times[fp_mask]

    if len(fp_times) < 2:
        return {
            'n_false_alarms': int(fp_mask.sum()),
            'iai_cv': None,
            'fano_factor': None,
            'interpretation': 'Insufficient false alarms for clustering analysis'
        }

    fp_times_sorted = np.sort(fp_times)
    iai = np.diff(fp_times_sorted)

    iai_mean = float(iai.mean())
    iai_std  = float(iai.std())
    cv = float(iai_std / iai_mean) if iai_mean > 0 else 0.0

    # Fano factor: count FAs per 1-hour bin
    total_duration_h = (window_start_times.max() - window_start_times.min()) / 3600.0
    n_bins = max(1, int(total_duration_h))
    counts, _ = np.histogram(fp_times / 3600.0, bins=n_bins)
    fano = float(counts.var() / counts.mean()) if counts.mean() > 0 else 0.0

    interpretation = (
        "Clustered (Fano > 1, CV > 1 — bursts of false alarms)" if (fano > 1.2 and cv > 1.2)
        else "Poisson-like (uniformly distributed false alarms)" if (0.7 < fano < 1.3)
        else "Underdispersed (regular false alarms — unlikely in EEG)"
    )

    return {
        'n_false_alarms': int(fp_mask.sum()),
        'iai_mean_sec': iai_mean,
        'iai_std_sec': iai_std,
        'iai_cv': cv,
        'fano_factor': fano,
        'iai_values_sec': iai.tolist(),
        'interpretation': interpretation
    }


# ---------------------------------------------------------------------------
# 4. Biomarker Ablation Study
# ---------------------------------------------------------------------------

def biomarker_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    seizure_ids: List[str],
    model_class,
    model_kwargs: dict,
    config: Optional[EEGConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Systematically removes one biomarker group at a time and measures ΔAUC
    to identify which features are most informative for seizure prediction.

    Feature groups (indices into the 180-feature vector per 18 channels × 10 features):
      - Relative band power (δ, θ, α, β, γ): features 0-4 per channel
      - Slow/fast ratio: feature 5 per channel
      - Spectral entropy: feature 6 per channel
      - Hjorth (Activity, Mobility, Complexity): features 7-9 per channel

    Args:
        X: Feature matrix (N, 180).
        y: Labels (N,).
        seizure_ids: Per-window seizure IDs.
        model_class: Model class (e.g., RandomForestSeizurePredictor).
        model_kwargs: Keyword arguments for model constructor.
        config: EEGConfig.

    Returns:
        Dict mapping ablation_name -> {'auc_mean', 'auc_std', 'delta_auc'}.
    """
    from src.evaluation.loso import run_loso_cv

    cfg = config or default_config
    n_features_per_channel = 10  # 5 rel_power + 1 slow_fast + 1 entropy + 3 hjorth
    n_channels = X.shape[1] // n_features_per_channel

    # Feature group slice indices within each channel block
    feature_groups = {
        'all_features': None,           # Baseline: use all features
        'ablate_delta': [0],            # Remove relative delta power
        'ablate_theta': [1],            # Remove relative theta power
        'ablate_alpha': [2],            # Remove relative alpha power
        'ablate_beta':  [3],            # Remove relative beta power
        'ablate_gamma': [4],            # Remove relative gamma power
        'ablate_band_power': [0,1,2,3,4], # Remove ALL band powers
        'ablate_slow_fast': [5],        # Remove slow-to-fast ratio
        'ablate_entropy': [6],          # Remove spectral entropy
        'ablate_hjorth': [7, 8, 9],     # Remove all Hjorth parameters
    }

    results = {}

    # Baseline AUC (all features)
    baseline_model = model_class(**model_kwargs)
    baseline_res = run_loso_cv(X, y, seizure_ids, baseline_model, cfg)
    baseline_auc = baseline_res['aggregated'].get('roc_auc_mean', 0.0)

    logger.info(f"Ablation study — baseline AUC (all features): {baseline_auc:.4f}")

    results['all_features'] = {
        'auc_mean': baseline_auc,
        'auc_std': baseline_res['aggregated'].get('roc_auc_std', 0.0),
        'delta_auc': 0.0
    }

    for group_name, feature_indices in feature_groups.items():
        if feature_indices is None:
            continue

        # Build ablated feature matrix: zero out the target features for all channels
        X_ablated = X.copy()
        for ch_idx in range(n_channels):
            block_start = ch_idx * n_features_per_channel
            for fi in feature_indices:
                X_ablated[:, block_start + fi] = 0.0

        ablation_model = model_class(**model_kwargs)
        ablation_res = run_loso_cv(X_ablated, y, seizure_ids, ablation_model, cfg)
        abl_auc = ablation_res['aggregated'].get('roc_auc_mean', 0.0)

        delta = abl_auc - baseline_auc   # negative = ablated features were helpful
        results[group_name] = {
            'auc_mean': abl_auc,
            'auc_std': ablation_res['aggregated'].get('roc_auc_std', 0.0),
            'delta_auc': float(delta)
        }
        logger.info(
            f"  [{group_name}] AUC={abl_auc:.4f} | ΔAUC={delta:+.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# 5. Spatial Channel Importance (per-channel AUC)
# ---------------------------------------------------------------------------

def channel_importance_analysis(
    X: np.ndarray,
    y: np.ndarray,
    seizure_ids: List[str],
    model_class,
    model_kwargs: dict,
    config: Optional[EEGConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute the contribution of each EEG channel by training a model
    on features from that channel alone and measuring LOSO AUC.

    Args:
        X: Full feature matrix (N, n_channels * n_features_per_channel).
        y: Labels (N,).
        seizure_ids: Per-window seizure IDs.
        model_class: Model class.
        model_kwargs: Model constructor arguments.
        config: EEGConfig.

    Returns:
        Dict mapping channel_name -> {'auc_mean', 'auc_std'}.
    """
    from src.evaluation.loso import run_loso_cv

    cfg = config or default_config
    n_features_per_channel = 10
    n_channels = X.shape[1] // n_features_per_channel
    channels = cfg.channels[:n_channels]

    results = {}
    for ch_idx, ch_name in enumerate(channels):
        block_start = ch_idx * n_features_per_channel
        block_end   = block_start + n_features_per_channel
        X_ch = X[:, block_start:block_end]

        model = model_class(**model_kwargs)
        loso_res = run_loso_cv(X_ch, y, seizure_ids, model, cfg, verbose=False)
        agg = loso_res['aggregated']

        results[ch_name] = {
            'auc_mean': agg.get('roc_auc_mean', 0.5),
            'auc_std': agg.get('roc_auc_std', 0.0)
        }
        logger.info(f"  [{ch_name}] AUC={results[ch_name]['auc_mean']:.4f}")

    return results
