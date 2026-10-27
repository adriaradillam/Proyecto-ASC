"""
Configuration parameters for Track A: EEG Seizure Prediction.
Applications of Soft Computing (ASC).

References:
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
- Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
- Lawhern et al. (2018), J. Neural Eng. (DOI: 10.1088/1741-2552/aace8c)
- Jiang et al. (2023), Front Neurosci (DOI: 10.3389/fnins.2023.1191683)
- Ibrahim et al. (2023), Front Signal Process (DOI: 10.3389/frsip.2023.1175305)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"


@dataclass
class EEGConfig:
    # Sampling & Preprocessing
    original_sfreq: float = 256.0      # CHB-MIT standard sampling rate in Hz
    target_sfreq: float = 128.0        # Downsampled for efficient model processing
    l_freq: float = 0.5                # High-pass filter cutoff (Hz)
    h_freq: float = 45.0               # Low-pass filter cutoff (Hz)
    notch_freq: float = 60.0           # Powerline interference notch filter (Hz)

    # Standard 18-channel bipolar montage common to CHB-MIT recordings
    channels: List[str] = field(default_factory=lambda: [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ])

    # Windowing for representations
    window_length_sec: float = 10.0    # 10s window (or 5s/30s)
    window_overlap: float = 0.5        # 50% overlap for preictal data augmentation
    interictal_overlap: float = 0.0    # No overlap for interictal to avoid over-representation

    # Horizon Definitions (Kuhlmann et al. 2018; Ren et al. 2022)
    # SPH = Seizure Prediction Horizon (intervention time, e.g., 5 min)
    # SOP = Seizure Occurrence Period (window where seizure is expected, e.g., 30 min)
    sph_min: float = 5.0               # minutes
    sop_min: float = 30.0              # minutes
    
    # Preictal duration: total window before seizure onset = SOP + SPH
    # Preictal window is [t_seizure - (sop + sph), t_seizure - sph]
    preictal_duration_min: float = 35.0 # minutes
    
    # Interictal separation: minimum distance from any seizure to ensure true baseline
    interictal_lead_hours: float = 4.0  # at least 4 hours before next seizure
    interictal_lag_hours: float = 4.0   # at least 4 hours after preceding seizure

    # Selected patient subset for benchmark (with high seizure count)
    target_patients: List[str] = field(default_factory=lambda: [
        'chb01', 'chb03', 'chb05', 'chb08', 'chb10'
    ])

    # Spectral Biomarker Bands
    frequency_bands: dict = field(default_factory=lambda: {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0)
    })

    # Reproducibility
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 999])
    default_seed: int = 42

    def ensure_directories(self):
        """Create necessary directories if they do not exist."""
        for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                  RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


default_config = EEGConfig()
