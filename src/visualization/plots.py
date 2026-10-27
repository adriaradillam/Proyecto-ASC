"""
Publication-Quality Visualization Suite for EEG Seizure Prediction.
Applications of Soft Computing (ASC).

Generates figures required by project syllabus:
1. Multi-channel EEG Signal Traces (Interictal vs Preictal vs Ictal).
2. Biomarker Trajectory Analysis (Temporal progression towards seizure).
3. ROC and Precision-Recall Curves (Model vs Baseline with intervals).
4. Training Loss and Validation Metric Learning Curves.
5. Per-Patient Performance Comparison (Sensitivity & FPR/h).
6. Continuous Seizure Timeline with Triggered Alarms.
7. Model vs Random Poisson Chance Boundary.

References:
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import FIGURES_DIR, default_config

# Set high-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})


def plot_eeg_signals_comparison(
    interictal_window: np.ndarray,
    preictal_window: np.ndarray,
    channels: List[str],
    sfreq: float,
    output_path: Optional[Path] = None
):
    """
    Plots side-by-side multi-channel EEG traces for an interictal vs preictal segment.
    """
    n_channels = min(8, interictal_window.shape[0])  # Show top 8 channels for visual clarity
    n_samples = interictal_window.shape[1]
    time_sec = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("Scalp EEG Dynamics: Interictal Baseline vs. Preictal State (CHB-MIT)", fontsize=14, y=0.98)

    offset = 4.0  # Vertical separation between channels
    
    for i in range(n_channels):
        ch_name = channels[i] if i < len(channels) else f"Ch {i+1}"
        
        # Interictal
        axes[i, 0].plot(time_sec, interictal_window[i], color="#1f77b4", lw=0.9)
        axes[i, 0].set_ylabel(ch_name, rotation=0, labelpad=25, va='center', fontweight='bold')
        axes[i, 0].set_ylim(-3.5, 3.5)
        
        # Preictal
        axes[i, 1].plot(time_sec, preictal_window[i], color="#d62728", lw=0.9)
        axes[i, 1].set_ylim(-3.5, 3.5)

    axes[0, 0].set_title("Interictal (Baseline State, >= 4h from seizure)", color="#1f77b4", fontweight='bold')
    axes[0, 1].set_title("Preictal (Transition Phase, 15 min prior to onset)", color="#d62728", fontweight='bold')
    
    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_biomarker_trajectories(
    time_to_seizure_min: np.ndarray,
    delta_power: np.ndarray,
    gamma_power: np.ndarray,
    spectral_entropy: np.ndarray,
    output_path: Optional[Path] = None
):
    """
    Visualizes how biomarkers evolve as time approaches seizure onset (t = 0).
    Demonstrates neurophysiological preictal shift.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Biomarker Trajectories Approaching Seizure Onset (SPH + SOP)", fontsize=14)

    # 1. Delta (Slow wave)
    axes[0].plot(time_to_seizure_min, delta_power, color="#2ca02c", lw=2, label="Relative Delta Power (0.5-4 Hz)")
    axes[0].axvspan(-5, 0, color="gray", alpha=0.3, label="SPH (5 min Intervention Horizon)")
    axes[0].set_ylabel("Relative Power")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Gamma (Fast wave)
    axes[1].plot(time_to_seizure_min, gamma_power, color="#ff7f0e", lw=2, label="Relative Gamma Power (30-45 Hz)")
    axes[1].axvspan(-5, 0, color="gray", alpha=0.3)
    axes[1].set_ylabel("Relative Power")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Spectral Entropy
    axes[2].plot(time_to_seizure_min, spectral_entropy, color="#9467bd", lw=2, label="Spectral Entropy")
    axes[2].axvspan(-5, 0, color="gray", alpha=0.3)
    axes[2].set_xlabel("Time to Seizure Onset (minutes)")
    axes[2].set_ylabel("Norm. Entropy")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_and_pr_curves(
    eval_results: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[Path] = None
):
    """
    Plots comparative ROC and Precision-Recall curves for all evaluated models against baseline.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        'Baseline (LogReg)': '#7f7f7f',
        'Random Forest': '#2ca02c',
        'Support Vector Machine': '#1f77b4',
        'Multilayer Perceptron (ANN)': '#ff7f0e',
        'Deep EEGNet (Raw EEG)': '#d62728'
    }

    # 1. ROC Curves
    for model_name, res in eval_results.items():
        y_true = res['y_true']
        y_prob = res['y_prob']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        c = colors.get(model_name, '#333333')
        style = '--' if 'Baseline' in model_name else '-'
        axes[0].plot(fpr, tpr, color=c, lw=2, linestyle=style, label=f"{model_name} (AUC = {roc_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.7)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Precision-Recall Curves
    for model_name, res in eval_results.items():
        y_true = res['y_true']
        y_prob = res['y_prob']
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        c = colors.get(model_name, '#333333')
        style = '--' if 'Baseline' in model_name else '-'
        axes[1].plot(rec, prec, color=c, lw=2, linestyle=style, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")

    axes[1].set_title("Precision-Recall (PR) Curve", fontweight='bold')
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_patient_summary(
    patient_metrics: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None
):
    """
    Plots per-patient Sensitivity vs. False Alarm Rate per Hour (FPR/h).
    """
    patients = list(patient_metrics.keys())
    sensitivities = [patient_metrics[p]['sensitivity'] * 100 for p in patients]
    fprs = [patient_metrics[p]['fpr_per_hour'] for p in patients]

    x = np.arange(len(patients))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width/2, sensitivities, width, label='Seizure Sensitivity (%)', color='#2ca02c', alpha=0.85)
    b2 = ax2.bar(x + width/2, fprs, width, label='False Alarms / Hour (FPR/h)', color='#d62728', alpha=0.85)

    ax1.set_xlabel('Patient ID', fontweight='bold')
    ax1.set_ylabel('Sensitivity (%)', color='#2ca02c', fontweight='bold')
    ax2.set_ylabel('False Alarms / Hour', color='#d62728', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(patients)
    ax1.set_ylim(0, 110)
    ax2.set_ylim(0, max(max(fprs) * 1.5, 1.0))

    ax1.grid(True, linestyle="--", alpha=0.5, axis='y')
    fig.suptitle('Patient-Specific Performance (Leave-One-Seizure-Out CV)', fontsize=13)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_random_predictor_comparison(
    observed_sensitivity: float,
    observed_fpr_h: float,
    sop_min: float = 30.0,
    output_path: Optional[Path] = None
):
    """
    Plots the analytical Random Poisson chance sensitivity curve vs. the model's operating point.
    """
    fpr_range = np.linspace(0.01, 2.0, 100)
    sop_hours = sop_min / 60.0
    chance_sensitivity = 1.0 - np.exp(-fpr_range * sop_hours)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_range, chance_sensitivity * 100, 'k--', lw=2, label=f"Random Predictor Chance Level (SOP={sop_min}m)")
    
    plt.scatter([observed_fpr_h], [observed_sensitivity * 100], color='#d62728', s=120, zorder=5,
                label=f"Proposed System ({observed_sensitivity*100:.1f}%, {observed_fpr_h:.2f} FA/h)")

    plt.title("Statistical Validation Against Random Chance Predictor", fontweight='bold')
    plt.xlabel("False Alarms per Hour (FPR/h)")
    plt.ylabel("Sensitivity (%)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 2.0)
    plt.ylim(0, 100)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
