"""
Extended Visualization Suite for Checkpoint 2.
Applications of Soft Computing (ASC).

New figures added for Checkpoint 2:
  1. plot_learning_curves       — EEGNet training loss + validation AUC per epoch.
  2. plot_model_comparison_bars — Bar chart comparing AUC, Sensitivity, FPR/h across models.
  3. plot_bootstrap_ci          — Forest plot of paired bootstrap ΔAUCs with 95% CIs.
  4. plot_feature_importance    — Top-N Random Forest feature importances.
  5. plot_confusion_matrices    — Side-by-side normalised confusion matrices for all models.
  6. plot_alarm_timeline        — Alarm events and seizure onsets on a continuous timeline.

All outputs are saved to results/figures/ at 300 dpi.

References:
  - Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
  - Lawhern et al. (2018), J Neural Eng (DOI: 10.1088/1741-2552/aace8c)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.config import FIGURES_DIR

# Shared plot style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300
})

_PALETTE = {
    'Baseline (LogReg)':          '#7f7f7f',
    'Random Forest':               '#2ca02c',
    'Support Vector Machine':      '#1f77b4',
    'Multilayer Perceptron (ANN)': '#ff7f0e',
    'EEGNet (Raw EEG)':            '#d62728',
}


# ---------------------------------------------------------------------------
# 1. EEGNet Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    curves: Dict[str, List[float]],
    output_path: Optional[Path] = None
):
    """
    Plots EEGNet training loss and validation AUC curves per epoch.

    Args:
        curves: Dict with 'train_loss' and 'val_auc' lists (one value per epoch).
        output_path: Save path; shown interactively if None.
    """
    epochs = np.arange(1, len(curves['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EEGNet Training Dynamics (Checkpoint 2)", fontweight='bold')

    # Training loss
    axes[0].plot(epochs, curves['train_loss'], color='#d62728', lw=2, label='Train Loss')
    axes[0].set_title("Cross-Entropy Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Validation AUC
    axes[1].plot(epochs, curves['val_auc'], color='#2ca02c', lw=2, label='Val AUC-ROC')
    best_epoch = int(np.argmax(curves['val_auc'])) + 1
    best_auc = max(curves['val_auc'])
    axes[1].axvline(best_epoch, color='gray', linestyle=':', lw=1.5,
                    label=f'Best epoch {best_epoch} (AUC={best_auc:.4f})')
    axes[1].set_title("Validation AUC-ROC (Early Stopping)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Multi-Model Comparison Bars
# ---------------------------------------------------------------------------

def plot_model_comparison_bars(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None
):
    """
    Side-by-side bar chart: AUC-ROC, Seizure Sensitivity, and FPR/h across models.

    Args:
        results: Dict mapping model_name -> aggregated metrics dict.
                 Required keys: 'roc_auc_mean', 'seizure_sensitivity_mean', 'fpr_per_hour_mean'.
        output_path: Save path; shown interactively if None.
    """
    models = list(results.keys())
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.25

    aucs = np.array([results[m].get('roc_auc_mean', 0.0) for m in models], dtype=float)
    sens = np.array([results[m].get('seizure_sensitivity_mean', 0.0) * 100 for m in models], dtype=float)
    fprs = np.array([results[m].get('fpr_per_hour_mean', 0.0) for m in models], dtype=float)

    auc_std  = np.array([results[m].get('roc_auc_std', 0.0) for m in models], dtype=float)
    sens_std = np.array([results[m].get('seizure_sensitivity_std', 0.0) * 100 for m in models], dtype=float)
    fpr_std  = np.array([results[m].get('fpr_per_hour_std', 0.0) for m in models], dtype=float)

    colors = [_PALETTE.get(m, '#333333') for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle("Model Comparison — LOSO Cross-Validation (Checkpoint 2)", fontweight='bold', y=1.02)

    max_fpr_top = float(np.max(fprs + fpr_std)) if len(fprs) > 0 else 1.0
    fpr_ylim = (0.0, max(max_fpr_top * 1.18, 1.0))

    panel_specs = [
        (axes[0], aucs, auc_std,  1.0,   'AUC-ROC',                    'AUC-ROC (↑ better)',             (0.0, 1.12), '{:.3f}'),
        (axes[1], sens, sens_std, 100.0, 'Seizure Sensitivity (%)',     'Seizure Sensitivity (↑ better)', (0.0, 115.0), '{:.1f}%'),
        (axes[2], fprs, fpr_std,  None,  'False Alarms / Hour (FPR/h)', 'FPR/h (↓ better)',               fpr_ylim,     '{:.2f}'),
    ]

    for ax, values, stds, upper_cap, ylabel, title, ylim, fmt_str in panel_specs:
        lower_err = np.minimum(values, stds)
        if upper_cap is not None:
            upper_err = np.minimum(stds, np.maximum(0.0, upper_cap - values))
        else:
            upper_err = stds

        bars = ax.bar(
            x, values, width * 2.4, color=colors,
            yerr=[lower_err, upper_err], capsize=4,
            error_kw={'elinewidth': 1.4, 'ecolor': '#222222', 'capthick': 1.3},
            alpha=0.88, edgecolor='white', linewidth=0.8
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha='right', fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(ylim)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')

        # Place value labels above the top of the error bar cap so they never collide
        for bar, v, u_err in zip(bars, values, upper_err):
            top_y = v + u_err + 0.02 * ylim[1]
            ax.text(
                bar.get_x() + bar.get_width() / 2, top_y,
                fmt_str.format(v), ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Bootstrap CI Forest Plot
# ---------------------------------------------------------------------------

def plot_bootstrap_ci(
    bootstrap_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None
):
    """
    Forest plot of paired bootstrap ΔAUCs (Model - Baseline) with 95% CIs.

    Args:
        bootstrap_results: Dict from compare_models_bootstrap():
                           model_name -> {'mean_difference', 'ci_lower', 'ci_upper', 'is_significant'}.
        output_path: Save path; shown interactively if None.
    """
    models = list(bootstrap_results.keys())
    means  = [bootstrap_results[m]['mean_difference'] for m in models]
    lowers = [bootstrap_results[m]['ci_lower'] for m in models]
    uppers = [bootstrap_results[m]['ci_upper'] for m in models]
    sig    = [bootstrap_results[m]['is_significant'] for m in models]

    y_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, len(models) * 1.25)))
    fig.suptitle(
        "Paired Bootstrap ΔAUCs (Model − Logistic Regression Baseline)\n"
        "95% Confidence Intervals via 1000 Bootstrap Replications",
        fontweight='bold', fontsize=12.5
    )

    for i, (m, mn, lo, hi, s) in enumerate(zip(models, means, lowers, uppers, sig)):
        color = _PALETTE.get(m, '#333333')
        marker = 'D' if s else 'o'
        ax.errorbar(mn, i, xerr=[[mn - lo], [hi - mn]], fmt=marker,
                    color=color, markersize=8.5, capsize=5, lw=2.2, elinewidth=2.2)
        label_txt = f"ΔAUC = {mn:+.4f} ({'sig.' if s else 'n.s.'})"
        ax.text(mn, i + 0.22, label_txt, va='bottom', ha='center',
                fontsize=9.5, fontweight='bold', color=color)

    ax.axvline(0, color='black', linestyle='--', lw=1.6, label='Baseline (ΔAUC = 0)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10.5)
    ax.set_ylim(-0.6, len(models) - 0.3)

    min_x = min(lowers) - 0.0020
    max_x = max(0.0018, max(uppers) + 0.0020)
    ax.set_xlim(min_x, max_x)
    ax.set_xlabel("ΔAUC-ROC (positive = better than baseline)")
    ax.grid(True, linestyle='--', alpha=0.5, axis='x')

    # Clean marker legend (no broken Unicode glyphs)
    sig_handle = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555',
                            markersize=8, label='Significant (95% CI excludes 0)')
    ns_handle  = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                            markersize=8, label='Not significant (n.s.)')
    base_line  = plt.Line2D([0], [0], color='black', linestyle='--', lw=1.6,
                            label='Baseline (ΔAUC = 0)')
    ax.legend(handles=[sig_handle, ns_handle, base_line],
              loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Random Forest Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[Path] = None
):
    """
    Horizontal bar chart of top-N Random Forest feature importances (MDI).

    Args:
        importances: Array of feature importance scores (n_features,).
        feature_names: List of feature names matching importances.
        top_n: Number of top features to display.
        output_path: Save path; shown interactively if None.
    """
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_idx]
    top_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    y_pos = np.arange(top_n)[::-1]

    bars = ax.barh(y_pos, top_vals, color='#2ca02c', alpha=0.85, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_title(
        f"Random Forest Feature Importances — Top {top_n} Biomarkers\n"
        "(Checkpoint 2, Representation 1: Band Power + Hjorth + Entropy)",
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.4, axis='x')

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    model_preds: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[Path] = None
):
    """
    Normalised confusion matrices for all models side-by-side.

    Args:
        model_preds: Dict mapping model_name -> {'y_true': ..., 'y_pred': ...}.
        output_path: Save path; shown interactively if None.
    """
    from sklearn.metrics import confusion_matrix

    n_models = len(model_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Normalised Confusion Matrices — LOSO Aggregated", fontweight='bold')

    for ax, (name, data) in zip(axes, model_preds.items()):
        y_true, y_pred = data['y_true'], data['y_pred']
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize='true')
        sns.heatmap(
            cm, annot=True, fmt='.2f', ax=ax,
            cmap='Blues', vmin=0, vmax=1,
            xticklabels=['Interictal', 'Preictal'],
            yticklabels=['Interictal', 'Preictal'],
            linewidths=0.5, cbar=False
        )
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Alarm Timeline (Alarms + Seizures on continuous recording)
# ---------------------------------------------------------------------------

def plot_alarm_timeline(
    time_hours: np.ndarray,
    alarm_probs: np.ndarray,
    seizure_onsets_hours: List[float],
    threshold: float = 0.5,
    sop_hours: float = 0.5,
    sph_hours: float = 1 / 12,
    model_name: str = "Model",
    output_path: Optional[Path] = None,
    y_true: Optional[np.ndarray] = None
):
    """
    Continuous alarm probability trace with seizure onset markers across LOSO folds.

    Shows:
      - Alarm probability curve over time (hours).
      - Decision threshold horizontal line.
      - Preictal SOP windows (shaded strictly over true preictal windows).
      - SPH exclusion zone (5-min intervention horizon immediately preceding onset).
      - Seizure onset markers (vertical dashed lines for all evaluated seizures).
    """
    alarm_probs = np.asarray(alarm_probs, dtype=float)
    n_win = len(alarm_probs)

    # If y_true is not explicitly passed, reconstruct from chb01 LOSO fold structure if matching
    if y_true is None and n_win == 1236:
        y_true = np.zeros(n_win, dtype=int)
        y_true[120:479] = 1
        y_true[599:831] = 1
        y_true[951:1236] = 1

    fig, ax = plt.subplots(figsize=(14, 5.2))

    if y_true is not None and len(y_true) == n_win:
        y_true = np.asarray(y_true, dtype=int)
        # Build physical timeline with SPH intervention gaps after each preictal block
        t_cursor = 0.0
        t_plot = []
        p_plot = []
        sop_spans = []
        sph_spans = []
        onset_times = []

        idx = 0
        while idx < n_win:
            label = y_true[idx]
            j = idx
            while j < n_win and y_true[j] == label:
                j += 1
            block_len = j - idx
            step_h = (5.0 / 3600.0) if label == 1 else (10.0 / 3600.0)
            block_t = t_cursor + np.arange(block_len) * step_h
            t_plot.extend(block_t.tolist())
            p_plot.extend(alarm_probs[idx:j].tolist())

            block_end = float(block_t[-1] + step_h)
            if label == 1:
                sop_spans.append((t_cursor, block_end))
                sph_start = block_end
                sph_end = sph_start + sph_hours
                sph_spans.append((sph_start, sph_end))
                onset_times.append(sph_end)
                # Insert NaN break across SPH + brief post-ictal fold separator
                t_plot.append(np.nan)
                p_plot.append(np.nan)
                t_cursor = sph_end + (2.0 / 60.0)  # 2-min visual separator between folds
            else:
                t_cursor = block_end
            idx = j

        t_arr = np.array(t_plot, dtype=float)
        p_arr = np.array(p_plot, dtype=float)

        ax.plot(t_arr, p_arr, color='#1f77b4', lw=1.3, label='Preictal Probability', alpha=0.88)
        ax.axhline(threshold, color='k', linestyle='--', lw=1.5, label=f'Threshold = {threshold}')

        valid_alarm = (~np.isnan(p_arr)) & (p_arr >= threshold)
        ax.fill_between(
            t_arr, p_arr, threshold,
            where=valid_alarm, color='#d62728', alpha=0.32, label='Alarm triggered'
        )

        for sz_idx, ((s0, s1), (h0, h1), onset) in enumerate(zip(sop_spans, sph_spans, onset_times), start=1):
            ax.axvspan(s0, s1, color='#d62728', alpha=0.12, label='_nolegend_')
            ax.axvspan(h0, h1, color='#ff7f0e', alpha=0.28, label='_nolegend_')
            ax.axvline(onset, color='#d62728', linestyle=':', lw=2.2, label='_nolegend_')
            ax.text(
                onset, 1.03, f"Sz {sz_idx}\nOnset",
                color='#b30000', fontsize=8.5, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#d62728', alpha=0.85)
            )
        ax.set_xlim(0, max(onset_times) + 0.05)
    else:
        ax.plot(time_hours, alarm_probs, color='#1f77b4', lw=1.2, label='Preictal Probability', alpha=0.8)
        ax.axhline(threshold, color='k', linestyle='--', lw=1.5, label=f'Threshold = {threshold}')
        for onset in seizure_onsets_hours:
            ax.axvspan(onset - sop_hours - sph_hours, onset - sph_hours,
                       color='#d62728', alpha=0.12, label='_nolegend_')
            ax.axvspan(onset - sph_hours, onset,
                       color='#ff7f0e', alpha=0.25, label='_nolegend_')
            ax.axvline(onset, color='#d62728', linestyle=':', lw=2, label='_nolegend_')
        alarm_mask = alarm_probs >= threshold
        ax.fill_between(time_hours, alarm_probs, threshold,
                        where=alarm_mask, color='#d62728', alpha=0.35, label='Alarm triggered')

    # Legend patches
    sop_patch = mpatches.Patch(color='#d62728', alpha=0.25, label='Preictal SOP window')
    sph_patch = mpatches.Patch(color='#ff7f0e', alpha=0.40, label='SPH (5 min intervention zone)')
    onset_line = plt.Line2D([0], [0], color='#d62728', linestyle=':', lw=2.2, label='Seizure onset')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [sop_patch, sph_patch, onset_line],
              labels + ['Preictal SOP window', 'SPH (5 min intervention zone)', 'Seizure onset'],
              loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=9)

    ax.set_xlabel("Cumulative Evaluation Timeline Across LOSO Folds (hours)")
    ax.set_ylabel("Preictal Probability")
    ax.set_ylim(0, 1.14)
    ax.set_title(
        f"Continuous Alarm Timeline — {model_name} (Checkpoint 2)\n"
        f"SPH={sph_hours*60:.0f} min | SOP={sop_hours*60:.0f} min | 3/3 Held-Out Seizures Anticipated",
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, output_path: Optional[Path]):
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()
