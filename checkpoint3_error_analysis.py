"""
Checkpoint 3: In-Depth Error Analysis & Lead-Time Analysis.
Applications of Soft Computing (ASC), Track A — EEG Seizure Prediction.

This script implements Week 5 milestone deliverables:

  1. Per-Seizure Breakdown (hit/miss, peak probability, lead time)
  2. Lead-Time Distribution Analysis
  3. False Alarm Temporal Clustering (IAI, Fano factor)
  4. Biomarker Ablation Study (ΔAUC per feature group)
  5. Spatial Channel Importance Analysis (per-channel AUC)

  Output Figures (results/figures/):
    - chb01_cp3_per_seizure_breakdown.png
    - chb01_cp3_lead_time_histogram.png
    - chb01_cp3_false_alarm_iai.png
    - chb01_cp3_ablation_study.png
    - chb01_cp3_channel_heatmap.png

  Output Tables (results/tables/):
    - chb01_checkpoint3_error_analysis.json

An honest negative result with solid analysis beats a leaky success.
  — ASC Final Project Specification

References:
    Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
    Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
    Ibrahim et al. (2023), Front Signal Process (DOI: 10.3389/frsip.2023.1175305)
    Jiang et al. (2023), Front Neurosci (DOI: 10.3389/fnins.2023.1191683)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    default_config, RAW_DATA_DIR, FIGURES_DIR, TABLES_DIR
)
from src.data.dataset import SeizurePredictionDataset
from src.features.biomarkers import BiomarkerExtractor
from src.models.classical import RandomForestSeizurePredictor
from src.evaluation.loso import run_loso_cv
from src.evaluation.error_analysis import (
    per_seizure_breakdown,
    compute_lead_time_distribution,
    analyze_false_alarm_clustering,
    biomarker_ablation_study,
    channel_importance_analysis
)
from src.visualization.plots_cp3 import (
    plot_per_seizure_breakdown,
    plot_lead_time_histogram,
    plot_false_alarm_iai,
    plot_ablation_study,
    plot_channel_heatmap
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PATIENT_ID = 'chb01'


def main():
    logger.info("=" * 65)
    logger.info("ASC Final Project — Checkpoint 3: Error Analysis")
    logger.info(f"Patient: {PATIENT_ID} | Best Model: Random Forest")
    logger.info("=" * 65)

    default_config.ensure_directories()

    # ── 1. Load Data ──────────────────────────────────────────────────────
    logger.info("[Step 1/5] Loading data...")
    dataset = SeizurePredictionDataset(
        patient_id=PATIENT_ID,
        raw_dir=RAW_DATA_DIR,
        config=default_config
    )
    patient_data = dataset.build_patient_dataset()
    X_raw       = patient_data['X']
    y           = patient_data['y']
    seizure_ids = patient_data['seizure_ids']

    # ── 2. Extract Biomarker Features ────────────────────────────────────
    logger.info("[Step 2/5] Extracting biomarker features...")
    extractor  = BiomarkerExtractor(config=default_config)
    X_features = extractor.extract_dataset_features(X_raw)

    # Build approximate window start times for temporal analysis
    n_windows = len(y)
    window_step_sec = default_config.window_length_sec * (1 - default_config.window_overlap)
    window_start_times = np.arange(n_windows) * window_step_sec

    # ── 3. LOSO Evaluation (Random Forest — best classical model) ─────────
    logger.info("[Step 3/5] LOSO evaluation with Random Forest...")
    rf_model   = RandomForestSeizurePredictor(n_estimators=300, random_state=42)
    rf_loso    = run_loso_cv(X_features, y, seizure_ids, rf_model, config=default_config)
    y_prob_all = rf_loso['all_y_prob']
    y_true_all = rf_loso['all_y_true']
    sids_all   = rf_loso['all_seizure_ids']

    # ── 4. Per-Seizure Breakdown ─────────────────────────────────────────
    logger.info("[Step 4/5] Running error analyses...")

    # 4a. Per-seizure breakdown
    # Reconstruct window_start_times for the test folds (same ordering as LOSO output)
    n_test = len(y_true_all)
    wst_test = np.arange(n_test) * window_step_sec

    seizure_breakdown = per_seizure_breakdown(
        y_true=y_true_all,
        y_prob=y_prob_all,
        seizure_ids=sids_all,
        window_start_times=wst_test,
        config=default_config
    )

    for r in seizure_breakdown:
        status = "✓ HIT" if r['predicted'] else "✗ MISS"
        logger.info(
            f"  {r['seizure_id']}: {status} | "
            f"Peak prob={r['peak_preictal_prob']:.3f} | "
            f"Lead={r['lead_time_min']:.1f} min"
        )

    # 4b. Lead-time distribution
    lead_time_stats = compute_lead_time_distribution(seizure_breakdown)
    logger.info(
        f"  Lead time: mean={lead_time_stats.get('mean_lead_time_min', 0):.1f} min | "
        f"≥SPH: {lead_time_stats.get('frac_above_sph', 0)*100:.0f}%"
    )

    # 4c. False alarm clustering
    y_pred_all = (y_prob_all >= 0.5).astype(int)
    fa_clustering = analyze_false_alarm_clustering(y_true_all, y_pred_all, wst_test)
    logger.info(
        f"  FA clustering: n={fa_clustering['n_false_alarms']} | "
        f"CV={fa_clustering.get('iai_cv', 'N/A')} | "
        f"Fano={fa_clustering.get('fano_factor', 'N/A')} | "
        f"{fa_clustering.get('interpretation', '')}"
    )

    # 4d. Biomarker ablation
    logger.info("  Running biomarker ablation study (takes ~2-5 min)...")
    ablation_results = biomarker_ablation_study(
        X=X_features,
        y=y,
        seizure_ids=seizure_ids,
        model_class=RandomForestSeizurePredictor,
        model_kwargs={'n_estimators': 100, 'random_state': 42}   # Fewer trees for speed
    )

    # 4e. Channel importance
    logger.info("  Running channel importance analysis (takes ~2-5 min)...")
    channel_results = channel_importance_analysis(
        X=X_features,
        y=y,
        seizure_ids=seizure_ids,
        model_class=RandomForestSeizurePredictor,
        model_kwargs={'n_estimators': 100, 'random_state': 42}
    )

    # ── 5. Generate Figures ────────────────────────────────────────────────
    logger.info("[Step 5/5] Generating figures...")

    plot_per_seizure_breakdown(
        per_seizure_results=seizure_breakdown,
        model_name='Random Forest',
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp3_per_seizure_breakdown.png'
    )
    logger.info("  ✓ Per-seizure breakdown")

    plot_lead_time_histogram(
        lead_time_stats=lead_time_stats,
        sph_min=default_config.sph_min,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp3_lead_time_histogram.png'
    )
    logger.info("  ✓ Lead-time histogram")

    if fa_clustering.get('n_false_alarms', 0) > 2:
        plot_false_alarm_iai(
            fa_clustering=fa_clustering,
            output_path=FIGURES_DIR / f'{PATIENT_ID}_cp3_false_alarm_iai.png'
        )
        logger.info("  ✓ False alarm IAI")

    plot_ablation_study(
        ablation_results=ablation_results,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp3_ablation_study.png'
    )
    logger.info("  ✓ Ablation study")

    plot_channel_heatmap(
        channel_results=channel_results,
        output_path=FIGURES_DIR / f'{PATIENT_ID}_cp3_channel_heatmap.png'
    )
    logger.info("  ✓ Channel heatmap")

    # ── Save Summary JSON ─────────────────────────────────────────────────
    def _ser(obj):
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_ser(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    summary = {
        'patient': PATIENT_ID,
        'checkpoint': 3,
        'best_model': 'Random Forest',
        'per_seizure_breakdown': _ser(seizure_breakdown),
        'lead_time_analysis': _ser(lead_time_stats),
        'false_alarm_clustering': _ser(fa_clustering),
        'ablation_study': _ser(ablation_results),
        'channel_importance': _ser(channel_results)
    }

    out_path = TABLES_DIR / f'{PATIENT_ID}_checkpoint3_error_analysis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 65)
    logger.info("Checkpoint 3 Complete")
    logger.info(f"Figures: results/figures/ | JSON: {out_path}")
    logger.info("=" * 65)


if __name__ == '__main__':
    main()
