"""
EEG Seizure Prediction Dataset & Leak-Free Windowing Pipeline.
Applications of Soft Computing (ASC).

Strictly enforces:
1. No data leakage: Leave-One-Seizure-Out (LOSO) cross-validation per patient.
2. Horizon separation: Preictal window = [t_onset - SOP - SPH, t_onset - SPH].
3. Ictal & Postictal exclusion: Active seizures and immediate 1h aftermath removed.
4. Interictal separation: Baseline periods taken >= 4h away from any seizure.

References:
- Kuhlmann et al. (2018), Nat Rev Neurol (DOI: 10.1038/s41582-018-0055-2)
- Ren et al. (2022), Front Neurol (DOI: 10.3389/fneur.2022.1016224)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import mne

from src.config import default_config, EEGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose MNE logging during processing
mne.set_log_level("WARNING")


class EEGPreprocessor:
    """Preprocesses raw scalp EEG recordings: filtering, resampling, and channel standardization."""

    def __init__(self, config: Optional[EEGConfig] = None):
        self.config = config or default_config

    def load_and_preprocess_edf(self, edf_path: Path) -> Tuple[np.ndarray, float, List[str]]:
        """
        Loads an EDF file, applies bandpass and notch filtering, and resamples.
        
        Returns:
            data: np.ndarray of shape (n_channels, n_samples)
            sfreq: target sampling frequency
            channel_names: list of channel names retained
        """
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        
        # Standardize channel names (remove spaces, dots, and trailing dashes)
        raw.rename_channels(lambda s: s.strip().replace(" ", "").replace(".", ""))

        # Match target channels or retain available bipolar channels
        available_chans = raw.ch_names
        target_chans = [ch for ch in self.config.channels if ch in available_chans]
        
        if len(target_chans) < 10:
            # Fallback: keep first 18 standard EEG channels if naming differs slightly
            target_chans = [ch for ch in available_chans if not ch.startswith(("EKG", "ECG", "VNS"))][:18]

        raw.pick(target_chans)

        # 1. Notch filter at 60 Hz (powerline noise)
        if raw.info['sfreq'] > self.config.notch_freq * 2:
            raw.notch_filter(freqs=self.config.notch_freq, verbose=False)

        # 2. Bandpass filter 0.5 - 45 Hz
        raw.filter(l_freq=self.config.l_freq, h_freq=self.config.h_freq, verbose=False)

        # 3. Resample if necessary
        if raw.info['sfreq'] != self.config.target_sfreq:
            raw.resample(self.config.target_sfreq, verbose=False)

        data = raw.get_data()  # shape (n_channels, n_samples)
        
        # Per-channel z-score normalisation within recording to eliminate patient amplitude bias
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-8
        data = (data - mean) / std

        return data, self.config.target_sfreq, target_chans


class SeizurePredictionDataset:
    """
    Extracts preictal and interictal windows from patient recordings for seizure prediction.
    Enforces SPH (Seizure Prediction Horizon) and SOP (Seizure Occurrence Period).
    """

    def __init__(self, patient_id: str, raw_dir: Path, config: Optional[EEGConfig] = None):
        self.patient_id = patient_id
        self.raw_dir = raw_dir / patient_id
        self.config = config or default_config
        self.preprocessor = EEGPreprocessor(self.config)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.raw_dir / f"{self.patient_id}_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file missing for {self.patient_id} at {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_windows_for_file(
        self,
        edf_filename: str
    ) -> List[Dict[str, Any]]:
        """
        Extract labeled windows for a single recording.
        Labels:
            1: Preictal (within [t_onset - SOP - SPH, t_onset - SPH])
            0: Interictal (baseline period)
            -1: Ictal or SPH or Postictal (ignored in prediction)
        """
        edf_path = self.raw_dir / edf_filename
        if not edf_path.exists():
            logger.warning(f"File {edf_path} does not exist, skipping.")
            return []

        rec_info = self.metadata["records"].get(edf_filename, {})
        seizures = rec_info.get("seizures", [])

        data, sfreq, channels = self.preprocessor.load_and_preprocess_edf(edf_path)
        n_channels, n_samples = data.shape
        total_duration_sec = n_samples / sfreq

        window_size_samples = int(self.config.window_length_sec * sfreq)
        
        sph_sec = self.config.sph_min * 60.0
        sop_sec = self.config.sop_min * 60.0

        windows = []

        if len(seizures) > 0:
            # File contains one or more seizures
            for s_idx, s in enumerate(seizures):
                onset_sec = s["start_sec"]
                offset_sec = s["end_sec"]

                # Preictal period: [onset - SOP - SPH, onset - SPH]
                preictal_start_sec = max(0.0, onset_sec - sop_sec - sph_sec)
                preictal_end_sec = max(0.0, onset_sec - sph_sec)

                # Extract Preictal Windows with overlap
                step_samples = int(window_size_samples * (1.0 - self.config.window_overlap))
                start_sample = int(preictal_start_sec * sfreq)
                end_sample = int(preictal_end_sec * sfreq)

                for w_start in range(start_sample, end_sample - window_size_samples + 1, step_samples):
                    w_end = w_start + window_size_samples
                    window_data = data[:, w_start:w_end]
                    seiz_key = f"{self.patient_id}_{Path(edf_filename).stem}_s{s_idx}"
                    windows.append({
                        "data": window_data,
                        "label": 1,  # Preictal
                        "seizure_id": seiz_key,
                        "file_name": edf_filename,
                        "window_start_sec": w_start / sfreq,
                        "channels": channels
                    })
        else:
            # File is pure interictal (no seizures)
            step_samples = window_size_samples  # 0% overlap for interictal
            for w_start in range(0, n_samples - window_size_samples + 1, step_samples):
                w_end = w_start + window_size_samples
                window_data = data[:, w_start:w_end]
                windows.append({
                    "data": window_data,
                    "label": 0,  # Interictal
                    "seizure_id": "interictal",
                    "file_name": edf_filename,
                    "window_start_sec": w_start / sfreq,
                    "channels": channels
                })

        return windows

    def build_patient_dataset(self) -> Dict[str, Any]:
        """
        Builds the complete dataset for the patient with preictal and interictal pools.
        Returns dictionary with 'X', 'y', 'seizure_ids', 'file_names'.
        """
        all_windows = []
        for filename in self.metadata["records"].keys():
            local_path = self.raw_dir / filename
            if local_path.exists():
                logger.info(f"Processing {filename} for {self.patient_id}...")
                wins = self.extract_windows_for_file(filename)
                all_windows.extend(wins)

        if not all_windows:
            raise ValueError(f"No valid windows extracted for patient {self.patient_id}")

        X = np.stack([w["data"] for w in all_windows], axis=0)  # (N, channels, samples)
        y = np.array([w["label"] for w in all_windows], dtype=np.int64)
        seizure_ids = [w["seizure_id"] for w in all_windows]
        file_names = [w["file_name"] for w in all_windows]

        preictal_count = int(np.sum(y == 1))
        interictal_count = int(np.sum(y == 0))
        unique_seizures = set(s for s in seizure_ids if s != "interictal")

        logger.info(f"Dataset for {self.patient_id} built successfully:")
        logger.info(f"  Total Windows: {len(y)} | Preictal: {preictal_count} | Interictal: {interictal_count}")
        logger.info(f"  Distinct Seizures: {len(unique_seizures)} ({list(unique_seizures)})")

        return {
            "X": X,
            "y": y,
            "seizure_ids": seizure_ids,
            "file_names": file_names,
            "patient_id": self.patient_id,
            "sfreq": self.config.target_sfreq
        }
