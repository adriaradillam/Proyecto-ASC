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

    labels = []
    for i, r in enumerate(per_seizure_results):
        sid = r.get('seizure_id', '')
        rec = sid.replace('chb01_', '').replace('_s0', '') if sid else f"Sz {i+1}"
        labels.append(f"Sz {i+1}\n({rec})")

    peak_probs = [r['peak_preictal_prob'] for r in per_seizure_results]
    lead_times = [r['lead_time_min'] for r in per_seizure_results]
    predicted  = [r['predicted'] for r in per_seizure_results]
    colors = ['#2ca02c' if p else '#d62728' for p in predicted]

    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle(f"Per-Seizure Error Analysis — {model_name} (Checkpoint 3)", fontweight='bold')

    # Left: peak preictal probability
    bars = axes[0].bar(x, peak_probs, width=0.55, color=colors, alpha=0.88, edgecolor='white')
    thresh_line = axes[0].axhline(0.5, color='k', linestyle='--', lw=1.6, label='Threshold = 0.5')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Peak Preictal Probability")
    axes[0].set_title("Peak Preictal Probability per Seizure", fontweight='bold')
    axes[0].set_ylim(0, 1.14)
    axes[0].grid(True, linestyle='--', alpha=0.4, axis='y')

    for bar, v in zip(bars, peak_probs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    hit_patch  = mpatches.Patch(color='#2ca02c', label='Predicted (HIT)')
    miss_patch = mpatches.Patch(color='#d62728', label='Missed (MISS)')
    axes[0].legend(handles=[hit_patch, miss_patch, thresh_line],
                   loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=9)

    # Right: lead times (only for predicted)
    lead_vals = [lt if p else 0 for lt, p in zip(lead_times, predicted)]
    lt_colors = ['#2ca02c' if p else '#bbbbbb' for p in predicted]
    bars2 = axes[1].bar(x, lead_vals, width=0.55, color=lt_colors, alpha=0.88, edgecolor='white')
    axes[1].axhline(5, color='#ff7f0e', linestyle='--', lw=1.8, label='SPH = 5.0 min (minimum intervention time)')
    axes[1].axhspan(0, 5, color='#ff7f0e', alpha=0.12, label='SPH exclusion zone (< 5 min)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Lead Time (minutes before onset)")
    axes[1].set_title("Alarm Lead Time per Predicted Seizure", fontweight='bold')
    axes[1].set_ylim(0, max(35.5, max(lead_vals) * 1.18))
    axes[1].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.4, axis='y')

    for bar, v in zip(bars2, lead_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                     f'{v:.1f} min', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

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

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Explicit clinical 2-minute bins centered on even values (e.g., [29, 31] centered at 30.0 min)
    bins = np.arange(1.0, 35.0, 2.0)
    counts, edges, patches = ax.hist(
        lead_times, bins=bins, rwidth=0.82,
        color='#2ca02c', alpha=0.85, edgecolor='white', linewidth=1.2,
        label='Predicted Seizures (Lead Time)', zorder=3
    )

    ax.axvspan(0, sph_min, color='#d62728', alpha=0.10, label=f'SPH Exclusion Zone (< {sph_min:.0f} min)')
    ax.axvspan(sph_min, 35, color='#2ca02c', alpha=0.05, label=f'Actionable SOP Window ({sph_min:.0f}–35 min)')
    ax.axvline(sph_min, color='#d62728', linestyle='--', lw=2.0,
               label=f'SPH = {sph_min:.1f} min (clinically required minimum)', zorder=4)

    mean_lt = float(lead_time_stats.get('mean_lead_time_min', np.mean(lead_times)))
    ax.axvline(mean_lt, color='#1f77b4', linestyle=':', lw=2.2,
               label=f"Mean Lead Time = {mean_lt:.1f} min", zorder=4)

    # Annotate non-zero histogram bars
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        if count > 0:
            center = 0.5 * (left + right)
            ax.text(
                center, count + 0.14,
                f"n = {int(count)}/{lead_time_stats.get('n_total', len(lead_times))} seizures\n({mean_lt:.1f} min)",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1b6ca8',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#1b6ca8', alpha=0.9),
                zorder=5
            )

    ax.set_xlim(0, 35)
    ax.set_ylim(0, max(counts) + 0.85)
    ax.set_yticks(range(0, int(max(counts)) + 2))
    ax.set_xlabel("Alarm Lead Time (minutes before seizure onset)")
    ax.set_ylabel("Number of Seizures")
    ax.set_title(
        f"Alarm Lead-Time Distribution — {lead_time_stats.get('n_predicted', 0)}/"
        f"{lead_time_stats.get('n_total', 0)} Seizures Predicted\n"
        f"({lead_time_stats.get('frac_above_sph', 0)*100:.0f}% with clinically sufficient lead time > {sph_min} min)",
        fontweight='bold'
    )
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=9)
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

_ABLATION_LABELS = {
    'ablate_band_power': 'All Band Powers (δ–γ)',
    'ablate_gamma':      'Gamma Power (30–45 Hz)',
    'ablate_beta':       'Beta Power (13–30 Hz)',
    'ablate_alpha':      'Alpha Power (8–13 Hz)',
    'ablate_theta':      'Theta Power (4–8 Hz)',
    'ablate_delta':      'Delta Power (0.5–4 Hz)',
    'ablate_slow_fast':  'Slow / Fast Band Ratio',
    'ablate_entropy':    'Spectral Entropy',
    'ablate_hjorth':     'Hjorth Parameters (Act/Mob/Comp)',
}

def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None
):
    """
    Horizontal bar chart of ΔAUC caused by ablating each biomarker group.
    A negative ΔAUC means removing that group degrades performance (important feature group).
    """
    baseline_auc = ablation_results.get('all_features', {}).get('auc_mean', 0.0)

    groups = [(name, float(v['delta_auc']))
              for name, v in ablation_results.items()
              if name != 'all_features']

    if not groups:
        return

    groups.sort(key=lambda x: x[1])   # Most harmful ablation (most negative ΔAUC) at bottom
    names = [
        _ABLATION_LABELS.get(g_name, g_name.replace('ablate_', '').replace('_', ' ').title())
        for g_name, _ in groups
    ]
    deltas = [d for _, d in groups]

    colors = ['#d62728' if d < 0 else '#5c768d' for d in deltas]

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, len(groups) * 0.55)))
    y_pos = np.arange(len(groups))

    ax.barh(y_pos, deltas, height=0.68, color=colors, alpha=0.88, edgecolor='white')
    ax.axvline(0, color='black', lw=1.6, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("ΔAUC-ROC vs. Full Feature Set (negative = performance drop when removed)")
    ax.set_title(
        f"Biomarker Ablation Study — Impact on LOSO AUC-ROC\n"
        f"Full Feature Set Baseline AUC = {baseline_auc:.4f}",
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.4, axis='x')

    max_abs = max(max(abs(d) for d in deltas), 1e-4)
    offset = 0.04 * max_abs
    min_d = min(deltas)
    max_d = max(deltas)
    ax.set_xlim(min(min_d - 0.28 * max_abs, -0.0005), max(max_d + 0.28 * max_abs, 0.0005))

    for i, d in enumerate(deltas):
        x_txt = d - offset if d < 0 else d + offset
        ax.text(
            x_txt, i, f'{d:+.4f}',
            va='center', ha='right' if d < 0 else 'left',
            fontsize=9.5, fontweight='bold',
            color='#b30000' if d < 0 else '#334e68'
        )

    neg_patch = mpatches.Patch(color='#d62728', alpha=0.88, label='AUC degradation (essential biomarker group)')
    pos_patch = mpatches.Patch(color='#5c768d', alpha=0.88, label='Neutral / slight regularization gain when ablated')
    ax.legend(handles=[neg_patch, pos_patch], loc='lower right',
              frameon=True, facecolor='white', framealpha=0.92, fontsize=9)

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
    aucs = np.array([channel_results[ch]['auc_mean'] for ch in channels], dtype=float)

    fig, ax = plt.subplots(figsize=(13, 5.2))

    norm = plt.Normalize(vmin=float(aucs.min()), vmax=float(aucs.max()))
    cmap = plt.cm.RdYlGn
    colors = cmap(norm(aucs))

    x = np.arange(len(channels))
    bars = ax.bar(x, aucs, width=0.72, color=colors, alpha=0.92, edgecolor='#444444', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=35, ha='right', fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Single-Channel LOSO AUC-ROC")
    ax.set_title(
        "Spatial Channel Importance — LOSO AUC from Single-Channel Biomarkers\n"
        "(Green = highest predictive discriminability; Red = lowest discriminability)",
        fontweight='bold'
    )
    y_min = max(0.80, float(np.floor((aucs.min() - 0.04) * 20) / 20))
    ax.set_ylim(y_min, 1.025)
    ax.axhline(0.9911, color='#1f77b4', linestyle='--', lw=1.6, alpha=0.85,
               label='Full 17-Channel RF Model (AUC = 0.9911)')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    for bar, v in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.003,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=25)
    cbar.set_label("Single-Channel AUC-ROC", fontsize=9.5)

    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    _save_or_show(fig, output_path)
