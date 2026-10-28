"""
Visualization Suite for Checkpoint 3 — Error Analysis.
Applications of Soft Computing (ASC).

New figures:
  1. plot_per_seizure_breakdown   — Bar chart: predicted vs missed seizures with lead times.
  2. plot_lead_time_histogram     — Distribution of alarm lead times.
  3. plot_false_alarm_iai         — Inter-alarm interval histogram with Poisson fit.
  4. plot_ablation_study          — ΔAUC bar chart for biomarker ablation groups.
  5. plot_channel_heatmap         — Scalp AUC heatmap across 18 bipolar channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
from scipy.stats import expon

from src.config import FIGURES_DIR

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300
})


def _save_or_show(fig, output_path):
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 1. Per-Seizure Breakdown
# ---------------------------------------------------------------------------

def plot_per_seizure_breakdown(
    per_seizure_results: List[Dict[str, Any]],
    model_name: str = "Best Model",
    output_path: Optional[Path] = None
):
    """
    Bar chart showing per-seizure prediction outcome (hit/miss) and lead times.

    Each seizure is shown with its peak preictal probability and lead time.
    Correctly predicted seizures are shown in green; missed in red.
    """
    n = len(per_seizure_results)
    if n == 0:
        return

    labels = [r['seizure_id'].split('_s')[-1] for r in per_seizure_results]
    labels = [f"Sz {i+1}" for i, _ in enumerate(per_seizure_results)]
    peak_probs = [r['peak_preictal_prob'] for r in per_seizure_results]
    lead_times = [r['lead_time_min'] for r in per_seizure_results]
    predicted  = [r['predicted'] for r in per_seizure_results]
    colors = ['#2ca02c' if p else '#d62728' for p in predicted]

    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Per-Seizure Error Analysis — {model_name} (Checkpoint 3)", fontweight='bold')

    # Left: peak preictal probability
    bars = axes[0].bar(x, peak_probs, color=colors, alpha=0.85, edgecolor='white')
    axes[0].axhline(0.5, color='k', linestyle='--', lw=1.5, label='Threshold = 0.5')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Peak Preictal Probability")
    axes[0].set_title("Peak Preictal Probability per Seizure", fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4, axis='y')

    for bar, v in zip(bars, peak_probs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    # Right: lead times (only for predicted)
    lead_vals = [lt if p else 0 for lt, p in zip(lead_times, predicted)]
    lt_colors = ['#2ca02c' if p else '#bbbbbb' for p in predicted]
    bars2 = axes[1].bar(x, lead_vals, color=lt_colors, alpha=0.85, edgecolor='white')
    axes[1].axhline(5, color='#ff7f0e', linestyle='--', lw=1.5, label='SPH (5 min)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Lead Time (minutes)")
    axes[1].set_title("Alarm Lead Time per Predicted Seizure", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4, axis='y')

    hit_patch  = mpatches.Patch(color='#2ca02c', label='Predicted (hit)')
    miss_patch = mpatches.Patch(color='#d62728', label='Missed')
    axes[0].legend(handles=[hit_patch, miss_patch], loc='lower right')

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Lead-Time Histogram
# ---------------------------------------------------------------------------

def plot_lead_time_histogram(
    lead_time_stats: Dict[str, Any],
    sph_min: float = 5.0,
    output_path: Optional[Path] = None
):
    """
    Distribution of alarm lead times across all predicted seizures.
    Includes a vertical line at SPH to indicate clinical relevance.
    """
    lead_times = lead_time_stats.get('lead_times_min', [])
    if not lead_times:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(lead_times, bins=max(5, len(lead_times) // 2),
            color='#2ca02c', alpha=0.8, edgecolor='white', label='Lead Time (min)')
    ax.axvline(sph_min, color='#d62728', linestyle='--', lw=2,
               label=f'SPH = {sph_min} min (clinically required minimum)')
    ax.axvline(lead_time_stats.get('mean_lead_time_min', 0),
               color='#1f77b4', linestyle='-', lw=2,
               label=f"Mean = {lead_time_stats.get('mean_lead_time_min', 0):.1f} min")

    ax.set_xlabel("Alarm Lead Time (minutes before seizure onset)")
    ax.set_ylabel("Number of Seizures")
    ax.set_title(
        f"Alarm Lead-Time Distribution — {lead_time_stats.get('n_predicted', 0)}/"
        f"{lead_time_stats.get('n_total', 0)} Seizures Predicted\n"
        f"({lead_time_stats.get('frac_above_sph', 0)*100:.0f}% with clinically sufficient lead time > {sph_min} min)",
        fontweight='bold'
    )
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Inter-Alarm Interval (False Alarm Clustering)
# ---------------------------------------------------------------------------

def plot_false_alarm_iai(
    fa_clustering: Dict[str, Any],
    output_path: Optional[Path] = None
):
    """
    Histogram of inter-alarm intervals (IAI) between consecutive false alarms,
    with an overlaid exponential (Poisson) fit for comparison.
    """
    iai = fa_clustering.get('iai_values_sec', [])
    if not iai or len(iai) < 2:
        return

    iai = np.array(iai) / 60.0  # Convert to minutes

    fig, ax = plt.subplots(figsize=(8, 5))

    n, bins, _ = ax.hist(iai, bins=20, density=True, color='#1f77b4',
                          alpha=0.8, edgecolor='white', label='Observed IAI')

    # Fit exponential distribution (Poisson null hypothesis)
    loc, scale = expon.fit(iai, floc=0)
    x_range = np.linspace(0, iai.max(), 200)
    ax.plot(x_range, expon.pdf(x_range, loc=loc, scale=scale),
            color='#d62728', lw=2, linestyle='--', label='Exponential fit (Poisson null)')

    cv = fa_clustering.get('iai_cv')
    fano = fa_clustering.get('fano_factor')
    interp = fa_clustering.get('interpretation', '')

    ax.set_xlabel("Inter-Alarm Interval (minutes)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"False Alarm Inter-Alarm Interval Distribution\n"
        f"CV={cv:.2f} | Fano={fano:.2f} | {interp}",
        fontweight='bold'
    )
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Biomarker Ablation Study
# ---------------------------------------------------------------------------

def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None
):
    """
    Horizontal bar chart of ΔAUC caused by ablating each biomarker group.
    A large negative ΔAUC means that group is important for prediction.
    """
    baseline_auc = ablation_results.get('all_features', {}).get('auc_mean', 0.0)

    groups = [(name, v['delta_auc'])
              for name, v in ablation_results.items()
              if name != 'all_features']

    if not groups:
        return

    groups.sort(key=lambda x: x[1])   # Most harmful ablation first
    names = [g[0].replace('ablate_', '').replace('_', ' ').title() for g, _ in groups]
    deltas = [d for _, d in groups]

    colors = ['#d62728' if d < -0.01 else '#7f7f7f' for d in deltas]

    fig, ax = plt.subplots(figsize=(9, max(4, len(groups) * 0.5)))
    y_pos = np.arange(len(groups))

    ax.barh(y_pos, deltas, color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("ΔAUC vs. Full Feature Set (negative = important feature group)")
    ax.set_title(
        f"Biomarker Ablation Study — Impact on AUC-ROC\n"
        f"Baseline (all features) AUC = {baseline_auc:.4f}",
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.4, axis='x')

    for i, (d, name) in enumerate(zip(deltas, names)):
        ax.text(d - 0.001 if d < 0 else d + 0.001, i,
                f'{d:+.4f}', va='center',
                ha='right' if d < 0 else 'left', fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Channel AUC Heatmap
# ---------------------------------------------------------------------------

def plot_channel_heatmap(
    channel_results: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None
):
    """
    Scalp heatmap approximation: bar chart of per-channel AUC arranged
    anatomically (frontal → temporal → parietal → occipital chains).
    """
    channels = list(channel_results.keys())
    aucs = [channel_results[ch]['auc_mean'] for ch in channels]

    # Arrange by scalp region (CHB-MIT bipolar montage layout)
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.RdYlGn(
        [(a - min(aucs)) / max(max(aucs) - min(aucs), 1e-6) for a in aucs]
    )

    bars = ax.bar(np.arange(len(channels)), aucs, color=colors, alpha=0.9, edgecolor='white')
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("AUC-ROC (channel-only model)")
    ax.set_title(
        "Spatial Channel Importance — AUC from Single-Channel Feature Set\n"
        "(Green = high discriminability; Red = low discriminability)",
        fontweight='bold'
    )
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color='k', linestyle='--', lw=1.2, alpha=0.5, label='Chance (AUC=0.5)')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax.legend()

    plt.tight_layout()
    _save_or_show(fig, output_path)
