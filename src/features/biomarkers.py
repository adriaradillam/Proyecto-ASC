"""
EEG Biomarker Extraction Pipeline (Representation 1).
Applications of Soft Computing (ASC).

Extracts established clinical and neurophysiological biomarkers:
1. Relative and Absolute Band Power (Delta, Theta, Alpha, Beta, Gamma) via Welch's PSD.
2. Spectral Entropy (Shannon entropy of the normalized power spectrum).
3. Hjorth Parameters (Activity, Mobility, Complexity).
4. Slow-to-Fast Power Ratio ((Delta + Theta) / (Alpha + Beta)).

References:
- Jiang et al. (2023), Front Neurosci (DOI: 10.3389/fnins.2023.1191683)
- Alvarado-Rojas et al. (2014), Sci Rep (DOI: 10.1038/srep04545)
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Optional, Tuple
from src.config import default_config, EEGConfig


class BiomarkerExtractor:
    """Extracts frequency, entropy, and temporal biomarker features from EEG windows."""

    def __init__(self, config: Optional[EEGConfig] = None):
        self.config = config or default_config
        self.bands = self.config.frequency_bands
        self.sfreq = self.config.target_sfreq

    def compute_band_powers(self, window_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes absolute band power, relative band power, and slow-to-fast ratio.
        
        Args:
            window_data: shape (n_channels, n_samples)
            
        Returns:
            abs_power: (n_channels, n_bands)
            rel_power: (n_channels, n_bands)
            slow_fast_ratio: (n_channels, 1)
        """
        n_channels, n_samples = window_data.shape
        n_bands = len(self.bands)

        # Welch PSD estimation
        nperseg = min(int(self.sfreq * 2), n_samples)
        freqs, psd = signal.welch(window_data, fs=self.sfreq, nperseg=nperseg, axis=-1)

        # Frequency resolution df
        df = freqs[1] - freqs[0]

        abs_powers = np.zeros((n_channels, n_bands))
        band_names = list(self.bands.keys())

        # Total power in 0.5 - 45 Hz range
        total_power_mask = (freqs >= self.config.l_freq) & (freqs <= self.config.h_freq)
        total_power = np.sum(psd[:, total_power_mask] * df, axis=-1, keepdims=True) + 1e-12

        for i, (band_name, (fmin, fmax)) in enumerate(self.bands.items()):
            mask = (freqs >= fmin) & (freqs <= fmax)
            band_pow = np.sum(psd[:, mask] * df, axis=-1)
            abs_powers[:, i] = band_pow

        rel_powers = abs_powers / total_power

        # Slow-to-fast ratio: (delta + theta) / (alpha + beta)
        slow_power = abs_powers[:, 0] + abs_powers[:, 1]  # delta + theta
        fast_power = abs_powers[:, 2] + abs_powers[:, 3] + 1e-12  # alpha + beta
        slow_fast_ratio = (slow_power / fast_power)[:, np.newaxis]

        return abs_powers, rel_powers, slow_fast_ratio

    def compute_spectral_entropy(self, window_data: np.ndarray) -> np.ndarray:
        """
        Computes normalized Spectral Entropy for each channel.
        Quantifies spectral flatness and power distribution irregularity.
        
        Returns:
            spectral_entropy: (n_channels, 1)
        """
        n_channels, n_samples = window_data.shape
        nperseg = min(int(self.sfreq * 2), n_samples)
        freqs, psd = signal.welch(window_data, fs=self.sfreq, nperseg=nperseg, axis=-1)

        # Restrict to band of interest
        mask = (freqs >= self.config.l_freq) & (freqs <= self.config.h_freq)
        psd_band = psd[:, mask]
        
        # Normalize PSD to probability distribution
        psd_norm = psd_band / (np.sum(psd_band, axis=-1, keepdims=True) + 1e-12)
        
        # Shannon entropy
        psd_norm = np.clip(psd_norm, 1e-12, 1.0)
        ent = -np.sum(psd_norm * np.log2(psd_norm), axis=-1)
        
        # Normalized by log2(N_bins)
        max_ent = np.log2(psd_band.shape[-1])
        norm_ent = (ent / max_ent)[:, np.newaxis]

        return norm_ent

    def compute_hjorth_parameters(self, window_data: np.ndarray) -> np.ndarray:
        """
        Computes Hjorth parameters: Activity, Mobility, Complexity.
        
        Returns:
            hjorth_feats: (n_channels, 3) -> [Activity, Mobility, Complexity]
        """
        # 1. Activity: Variance of signal
        activity = np.var(window_data, axis=-1)

        # 2. First derivative
        d1 = np.diff(window_data, axis=-1)
        var_d1 = np.var(d1, axis=-1)

        # 3. Second derivative
        d2 = np.diff(d1, axis=-1)
        var_d2 = np.var(d2, axis=-1)

        # Mobility: sqrt(var(d1) / var(signal))
        mobility = np.sqrt(var_d1 / (activity + 1e-12))

        # Complexity: Mobility(d1) / Mobility(signal)
        mobility_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
        complexity = mobility_d1 / (mobility + 1e-12)

        return np.stack([activity, mobility, complexity], axis=-1)

    def extract_window_features(self, window_data: np.ndarray) -> np.ndarray:
        """
        Extracts all biomarker features for a single multi-channel window.
        
        Args:
            window_data: shape (n_channels, n_samples)
            
        Returns:
            features: 1D array of shape (n_channels * n_features_per_channel,)
        """
        abs_pow, rel_pow, slow_fast = self.compute_band_powers(window_data)
        spec_ent = self.compute_spectral_entropy(window_data)
        hjorth = self.compute_hjorth_parameters(window_data)

        # Concatenate per channel:
        # 5 rel_power + 1 slow_fast + 1 spectral_entropy + 3 hjorth = 10 features per channel
        channel_feats = np.concatenate([rel_pow, slow_fast, spec_ent, hjorth], axis=-1)
        
        # Flatten across all channels
        return channel_feats.flatten()

    def extract_dataset_features(self, X: np.ndarray) -> np.ndarray:
        """
        Batch extracts features for an entire dataset of windows.
        
        Args:
            X: np.ndarray of shape (N_windows, n_channels, n_samples)
            
        Returns:
            feature_matrix: np.ndarray of shape (N_windows, n_total_features)
        """
        N = X.shape[0]
        feats_list = []
        for i in range(N):
            f = self.extract_window_features(X[i])
            feats_list.append(f)
        return np.array(feats_list, dtype=np.float32)
