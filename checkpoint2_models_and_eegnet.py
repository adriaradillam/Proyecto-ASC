"""
Checkpoint 2: Non-linear Classical Models + EEGNet Deep Learning
Applications of Soft Computing (ASC), Track A — EEG Seizure Prediction.

This script implements and evaluates all models required by Checkpoint 2:

  Representation 1 (Biomarkers, 180 features):
    - Logistic Regression Baseline    [carried forward from Checkpoint 1]
    - Random Forest                   [ensemble, n=300 trees, balanced weights]
    - Support Vector Machine          [RBF kernel + Platt probability calibration]
    - Multilayer Perceptron (shallow) [256→128 ReLU, Adam, early stopping]

  Representation 2 (Raw EEG, 10s windows):
    - EEGNet                          [compact CNN: Lawhern et al. 2018]

  Statistical Validation:
    - LOSO cross-validation (K-fold where K = #seizures)
    - Paired bootstrap 95% CI for ΔAUCs (each model vs. LogReg baseline)

  Output Figures (results/figures/):
    - chb01_cp2_roc_pr_curves.png
    - chb01_cp2_model_comparison_bars.png
    - chb01_cp2_bootstrap_ci.png
    - chb01_cp2_feature_importance_rf.png
    - chb01_cp2_confusion_matrices.png
    - chb01_cp2_eegnet_learning_curves.png
    - chb01_cp2_alarm_timeline.png

  Output Tables (results/tables/):
    - chb01_checkpoint2_summary.json

References:
    Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
    Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
    Lawhern et al. (2018), J Neural Eng (DOI: 10.1088/1741-2552/aace8c)
    Jiang et al. (2023), Front Neurosci (DOI: 10.3389/fnins.2023.1191683)
    Ibrahim et al. (2023), Front Signal Process (DOI: 10.3389/frsip.2023.1175305)
    Breiman (2001), Random Forests, Mach Learn (DOI: 10.1023/A:1010933404324)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ── Ensure project root is on sys.path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    default_config, RAW_DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR
)
from src.data.dataset import SeizurePredictionDataset
from src.features.biomarkers import BiomarkerExtractor
from src.models.baseline import LogisticRegressionBaseline, RandomChancePredictor
from src.models.classical import (
    RandomForestSeizurePredictor,
    SVMSeizurePredictor,
    MLPSeizurePredictor
)
from src.models.eegnet import EEGNetTrainer
from src.evaluation.loso import run_loso_cv, compare_models_bootstrap
from src.evaluation.metrics import compute_seizure_level_metrics
from src.visualization.plots import plot_roc_and_pr_curves
from src.visualization.plots_cp2 import (
    plot_learning_curves,
    plot_model_comparison_bars,
    plot_bootstrap_ci,
    plot_feature_importance,
    plot_confusion_matrices,
    plot_alarm_timeline
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PATIENT_ID = 'chb01'


# ---------------------------------------------------------------------------
# Step 1: Load and Window the Dataset
# ---------------------------------------------------------------------------

def load_patient_data() -> Dict[str, Any]:
    """Load raw EEG recordings for the target patient and extract windows."""
    logger.info(f"[Step 1/6] Loading CHB-MIT data for patient {PATIENT_ID}...")
    dataset = SeizurePredictionDataset(
        patient_id=PATIENT_ID,
        raw_dir=RAW_DATA_DIR,
        config=default_config
    )
    return dataset.build_patient_dataset()


# ---------------------------------------------------------------------------
# Step 2: Extract Biomarker Features (Representation 1)
# ---------------------------------------------------------------------------

def extract_biomarker_features(patient_data: Dict[str, Any]) -> np.ndarray:
    """Extract 180 neurophysiological biomarker features per window."""
    logger.info("[Step 2/6] Extracting biomarker features (Representation 1)...")
    extractor = BiomarkerExtractor(config=default_config)
    X_features = extractor.extract_dataset_features(patient_data['X'])
    logger.info(f"  Feature matrix: {X_features.shape} (windows × features)")
    return X_features


def build_feature_names() -> List[str]:
    """Construct human-readable feature names for interpretability analysis."""
    channels = default_config.channels
    band_names = list(default_config.frequency_bands.keys())
    feat_names = []
    for ch in channels:
        for b in band_names:
            feat_names.append(f"{ch}_rel_{b}")
        feat_names.append(f"{ch}_slow_fast_ratio")
        feat_names.append(f"{ch}_spec_entropy")
        feat_names.append(f"{ch}_hjorth_activity")
        feat_names.append(f"{ch}_hjorth_mobility")
        feat_names.append(f"{ch}_hjorth_complexity")
    return feat_names


# ---------------------------------------------------------------------------
# Step 3: LOSO Evaluation for Biomarker Models
# ---------------------------------------------------------------------------

def evaluate_biomarker_models(
    X: np.ndarray,
    y: np.ndarray,
    seizure_ids: List[str]
) -> Dict[str, Dict]:
    """
    Run LOSO cross-validation for all biomarker-based models.
    Returns a dict: model_name -> LOSO result dict.
    """
    logger.info("[Step 3/6] LOSO evaluation — Representation 1 (Biomarkers)...")

    models = {
        'Baseline (LogReg)':          LogisticRegressionBaseline(C=1.0, random_state=42),
        'Random Forest':               RandomForestSeizurePredictor(n_estimators=300, random_state=42),
        'Support Vector Machine':      SVMSeizurePredictor(C=1.0, gamma='scale', random_state=42),
        'Multilayer Perceptron (ANN)': MLPSeizurePredictor(
                                          hidden_layer_sizes=(256, 128),
                                          alpha=1e-3, max_iter=500, random_state=42
                                       ),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"  Evaluating: {name}")
        t0 = time.time()
        loso_res = run_loso_cv(X, y, seizure_ids, model, config=default_config)
        elapsed = time.time() - t0
        agg = loso_res['aggregated']
        logger.info(
            f"  [{name}] "
            f"AUC={agg.get('roc_auc_mean', 0):.4f}±{agg.get('roc_auc_std', 0):.4f} | "
            f"Sens={agg.get('seizure_sensitivity_mean', 0)*100:.1f}% | "
            f"FPR/h={agg.get('fpr_per_hour_mean', 0):.3f} | "
            f"Time={elapsed:.1f}s"
        )
        results[name] = loso_res

    return results


# ---------------------------------------------------------------------------
# Step 4: LOSO Evaluation for EEGNet (Representation 2)
# ---------------------------------------------------------------------------

def evaluate_eegnet(
    X_raw: np.ndarray,
    y: np.ndarray,
    seizure_ids: List[str]
) -> Dict[str, Any]:
    """
    Run LOSO cross-validation for EEGNet on raw EEG tensors.
    X_raw shape: (N, n_channels, n_samples)
    """
    logger.info("[Step 4/6] LOSO evaluation — Representation 2 (EEGNet on raw EEG)...")

    n_channels = X_raw.shape[1]
    n_samples  = X_raw.shape[2]

    eegnet_trainer = EEGNetTrainer(
        n_channels=n_channels,
        n_samples=n_samples,
        n_epochs=150,
        lr=1e-3,
        batch_size=32,
        dropout_rate=0.5,
        patience=20,
        random_state=42
    )

    t0 = time.time()
    loso_res = run_loso_cv(X_raw, y, seizure_ids, eegnet_trainer, config=default_config)
    elapsed = time.time() - t0
    agg = loso_res['aggregated']
    logger.info(
        f"  [EEGNet] "
        f"AUC={agg.get('roc_auc_mean', 0):.4f}±{agg.get('roc_auc_std', 0):.4f} | "
        f"Sens={agg.get('seizure_sensitivity_mean', 0)*100:.1f}% | "
        f"FPR/h={agg.get('fpr_per_hour_mean', 0):.3f} | "
        f"Time={elapsed:.1f}s"
    )

    return loso_res, eegnet_trainer


# ---------------------------------------------------------------------------
# Step 5: Generate All Figures
# ---------------------------------------------------------------------------

def generate_figures(
    patient_data: Dict[str, Any],
    X_features: np.ndarray,
    feature_names: List[str],
    biomarker_results: Dict[str, Dict],
    eegnet_result: Dict[str, Any],
    eegnet_trainer: EEGNetTrainer
):
    """Render and save all required Checkpoint 2 figures."""
    logger.info("[Step 5/6] Generating figures...")
    default_config.ensure_directories()

    # Collect pooled predictions from LOSO for each model
    eval_results_roc = {}
    model_probs = {}
    model_preds_cm = {}
    ref_y_true = None

    for name, loso_res in biomarker_results.items():
        y_true = loso_res['all_y_true']
        y_prob = loso_res['all_y_prob']
        eval_results_roc[name] = {'y_true': y_true, 'y_prob': y_prob}
        model_probs[name] = y_prob
        model_preds_cm[name] = {
            'y_true': y_true,
            'y_pred': (y_prob >= 0.5).astype(int)
        }
        if ref_y_true is None:
            ref_y_true = y_true

    # EEGNet
    eval_results_roc['EEGNet (Raw EEG)'] = {
        'y_true': eegnet_result['all_y_true'],
        'y_prob': eegnet_result['all_y_prob']
    }
    model_probs['EEGNet (Raw EEG)'] = eegnet_result['all_y_prob']
    model_preds_cm['EEGNet (Raw EEG)'] = {
        'y_true': eegnet_result['all_y_true'],
        'y_pred': (eegnet_result['all_y_prob'] >= 0.5).astype(int)
    }

    # Figure 1: ROC + PR curves (all models)
    plot_roc_and_pr_curves(
        eval_results=eval_results_roc,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_roc_pr_curves.png'
    )
    logger.info("  ✓ ROC + PR curves")

    # Figure 2: Multi-model bar comparison
    agg_results = {n: loso_res['aggregated'] for n, loso_res in biomarker_results.items()}
    agg_results['EEGNet (Raw EEG)'] = eegnet_result['aggregated']
    plot_model_comparison_bars(
        results=agg_results,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_model_comparison_bars.png'
    )
    logger.info("  ✓ Model comparison bars")

    # Figure 3: Bootstrap CI forest plot
    bootstrap_res = compare_models_bootstrap(
        y_true=ref_y_true,
        model_probs=model_probs,
        baseline_name='Baseline (LogReg)'
    )
    plot_bootstrap_ci(
        bootstrap_results=bootstrap_res,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_bootstrap_ci.png'
    )
    logger.info("  ✓ Bootstrap CI forest plot")

    # Figure 4: Random Forest feature importances
    # Re-fit RF on ALL data to get stable importances for the plot
    rf_model = RandomForestSeizurePredictor(n_estimators=300, random_state=42)
    rf_model.fit(X_features, patient_data['y'])
    plot_feature_importance(
        importances=rf_model.feature_importances_,
        feature_names=feature_names,
        top_n=20,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_feature_importance_rf.png'
    )
    logger.info("  ✓ RF feature importance")

    # Figure 5: Confusion matrices
    plot_confusion_matrices(
        model_preds=model_preds_cm,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_confusion_matrices.png'
    )
    logger.info("  ✓ Confusion matrices")

    # Figure 6: EEGNet learning curves (from last fold)
    curves = eegnet_trainer.get_learning_curves()
    if curves['train_loss']:
        plot_learning_curves(
            curves=curves,
            output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_eegnet_learning_curves.png'
        )
        logger.info("  ✓ EEGNet learning curves")

    # Figure 7: Alarm timeline (using best model = RF, from pooled LOSO predictions)
    rf_res = biomarker_results['Random Forest']
    # Build a synthetic time axis in hours
    n_windows = len(rf_res['all_y_true'])
    window_step_sec = default_config.window_length_sec * (1 - default_config.window_overlap)
    time_hours = np.arange(n_windows) * window_step_sec / 3600.0

    # Reconstruct approximate seizure onset times (at the preictal->interictal boundary)
    y_labels = rf_res['all_y_true']
    transitions = np.where(np.diff(y_labels.astype(int)) < 0)[0]
    seizure_onsets_hours = time_hours[transitions].tolist() if len(transitions) > 0 else [1.5, 3.0]

    plot_alarm_timeline(
        time_hours=time_hours,
        alarm_probs=rf_res['all_y_prob'],
        seizure_onsets_hours=seizure_onsets_hours,
        threshold=0.5,
        sop_hours=default_config.sop_min / 60.0,
        sph_hours=default_config.sph_min / 60.0,
        model_name='Random Forest',
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp2_alarm_timeline.png'
    )
    logger.info("  ✓ Alarm timeline")

    return bootstrap_res


# ---------------------------------------------------------------------------
# Step 6: Save Summary JSON
# ---------------------------------------------------------------------------

def save_summary(
    biomarker_results: Dict[str, Dict],
    eegnet_result: Dict[str, Any],
    bootstrap_res: Dict[str, Any]
):
    """Save complete Checkpoint 2 numerical results to JSON."""
    logger.info("[Step 6/6] Saving results summary...")
    default_config.ensure_directories()

    def _serialise(obj):
        """Recursively convert numpy types to Python scalars."""
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {
        'patient': PATIENT_ID,
        'checkpoint': 2,
        'protocol': {
            'cv_method': 'Leave-One-Seizure-Out (LOSO)',
            'sph_min': default_config.sph_min,
            'sop_min': default_config.sop_min,
            'window_length_sec': default_config.window_length_sec,
            'interictal_lead_lag_hours': default_config.interictal_lead_hours
        },
        'models': {}
    }

    for name, loso_res in biomarker_results.items():
        agg = loso_res['aggregated']
        summary['models'][name] = {
            'representation': 'Biomarkers (180 features)',
            'n_folds': len(loso_res['fold_results']),
            'aggregated': _serialise(agg),
            'fold_details': _serialise(loso_res['fold_results'])
        }

    eeg_agg = eegnet_result['aggregated']
    summary['models']['EEGNet (Raw EEG)'] = {
        'representation': 'Raw EEG (10s windows, 128 Hz)',
        'n_folds': len(eegnet_result['fold_results']),
        'aggregated': _serialise(eeg_agg),
        'fold_details': _serialise(eegnet_result['fold_results'])
    }

    summary['bootstrap_paired_comparison'] = _serialise(bootstrap_res)

    out_path = TABLES_DIR / f'{PATIENT_ID}_checkpoint2_summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"  Summary saved → {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 65)
    logger.info("ASC Final Project — Checkpoint 2: Models & Representations")
    logger.info(f"Patient: {PATIENT_ID} | Protocol: LOSO CV")
    logger.info("=" * 65)

    # 1. Load data
    patient_data = load_patient_data()
    X_raw       = patient_data['X']          # (N, channels, samples)
    y           = patient_data['y']
    seizure_ids = patient_data['seizure_ids']

    # 2. Extract biomarker features
    X_features   = extract_biomarker_features(patient_data)
    feature_names = build_feature_names()

    # 3. Evaluate biomarker models with LOSO
    biomarker_results = evaluate_biomarker_models(X_features, y, seizure_ids)

    # 4. Evaluate EEGNet (raw EEG, only if torch is available)
    try:
        import torch
        eegnet_result, eegnet_trainer = evaluate_eegnet(X_raw, y, seizure_ids)
    except ImportError:
        logger.warning(
            "PyTorch not installed — skipping EEGNet. "
            "Install with: pip install torch"
        )
        # Provide a dummy placeholder so the rest of the pipeline runs
        eegnet_result = {
            'aggregated': {
                'roc_auc_mean': 0.0, 'roc_auc_std': 0.0,
                'seizure_sensitivity_mean': 0.0, 'seizure_sensitivity_std': 0.0,
                'fpr_per_hour_mean': 0.0, 'fpr_per_hour_std': 0.0
            },
            'fold_results': [],
            'all_y_true': np.array([]),
            'all_y_prob': np.array([])
        }

        class _DummyTrainer:
            def get_learning_curves(self):
                return {'train_loss': [], 'val_auc': []}

        eegnet_trainer = _DummyTrainer()

    # 5. Generate figures
    bootstrap_res = generate_figures(
        patient_data=patient_data,
        X_features=X_features,
        feature_names=feature_names,
        biomarker_results=biomarker_results,
        eegnet_result=eegnet_result,
        eegnet_trainer=eegnet_trainer
    )

    # 6. Save summary
    summary = save_summary(biomarker_results, eegnet_result, bootstrap_res)

    # ── Print final report ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("CHECKPOINT 2 RESULTS SUMMARY")
    logger.info("=" * 65)
    logger.info(f"{'Model':<35} {'AUC-ROC':>10} {'Sens(%)':>10} {'FPR/h':>8}")
    logger.info("-" * 65)

    all_model_results = dict(biomarker_results)
    all_model_results['EEGNet (Raw EEG)'] = eegnet_result

    for name, loso_res in all_model_results.items():
        agg = loso_res['aggregated']
        auc  = agg.get('roc_auc_mean', 0)
        sens = agg.get('seizure_sensitivity_mean', 0) * 100
        fpr  = agg.get('fpr_per_hour_mean', 0)
        logger.info(f"  {name:<33} {auc:>10.4f} {sens:>9.1f}% {fpr:>8.3f}")

    logger.info("-" * 65)
    logger.info("\nPaired Bootstrap (vs. Logistic Regression Baseline):")
    for name, res in bootstrap_res.items():
        sig = "✓ Significant" if res['is_significant'] else "✗ Not significant"
        logger.info(
            f"  {name:<33} ΔAUC={res['mean_difference']:+.4f} "
            f"[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]  {sig}"
        )

    logger.info("=" * 65)
    logger.info("All figures saved to:  results/figures/")
    logger.info("Summary JSON saved to: results/tables/")
    logger.info("=" * 65)


if __name__ == '__main__':
    main()
