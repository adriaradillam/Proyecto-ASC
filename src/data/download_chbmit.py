"""
CHB-MIT Scalp EEG Dataset Downloader and Metadata Parser.
PhysioNet: https://physionet.org/content/chbmit/1.0.0/
DOI: 10.13026/C2K01R

Gold standard scalp EEG database for epileptic seizure prediction research.
24 pediatric subjects, recorded at 256 Hz with 16-bit resolution.
"""

import os
import re
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from src.config import RAW_DATA_DIR, default_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fast AWS S3 mirror with PhysioNet fallback
PHYSIONET_BASE_URL = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/"
PHYSIONET_FALLBACK_URL = "https://physionet.org/files/chbmit/1.0.0/"


class DownloadProgressBar(tqdm):
    """Custom progress bar hook for urllib download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    """Download a file from PhysioNet with progress bar and integrity checks."""
    if output_path.exists() and not force:
        logger.info(f"File already exists: {output_path.name}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        logger.info(f"Downloading {url} -> {output_path.name}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=temp_path, reporthook=t.update_to)
        
        temp_path.replace(output_path)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to download {url}: {e}")
        return False


def parse_summary_file(summary_path: Path) -> Dict[str, Any]:
    """
    Parse a CHB-MIT patient summary text file (e.g., chb01-summary.txt).
    Extracts sampling frequency, channels, file durations, and exact seizure onset/offset times.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract Frequency
    freq_match = re.search(r"Data Sampling Rate:\s*(\d+)\s*Hz", content)
    sampling_rate = int(freq_match.group(1)) if freq_match else 256

    # Parse file blocks
    file_blocks = re.split(r"File Name:\s*", content)
    records = {}

    for block in file_blocks[1:]:
        lines = block.strip().split("\n")
        filename = lines[0].strip()
        
        # Start and end time of file
        start_match = re.search(r"File Start Time:\s*([^\n]+)", block)
        end_match = re.search(r"File End Time:\s*([^\n]+)", block)
        seizure_num_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        
        num_seizures = int(seizure_num_match.group(1)) if seizure_num_match else 0
        seizures = []

        if num_seizures > 0:
            start_times = re.findall(r"Seizure\s*(?:\d+)?\s*Start Time:\s*(\d+)\s*seconds", block)
            end_times = re.findall(r"Seizure\s*(?:\d+)?\s*End Time:\s*(\d+)\s*seconds", block)

            for s_start, s_end in zip(start_times, end_times):
                seizures.append({
                    "start_sec": int(s_start),
                    "end_sec": int(s_end),
                    "duration_sec": int(s_end) - int(s_start)
                })

        records[filename] = {
            "file_name": filename,
            "start_time": start_match.group(1).strip() if start_match else None,
            "end_time": end_match.group(1).strip() if end_match else None,
            "num_seizures": num_seizures,
            "seizures": seizures
        }

    return {
        "sampling_rate": sampling_rate,
        "records": records
    }


def download_patient_data(
    patient_id: str,
    output_dir: Optional[Path] = None,
    seizure_files_only: bool = True,
    interictal_hours_needed: int = 4
) -> Dict[str, Any]:
    """
    Download patient metadata summary and required EDF files.
    
    Args:
        patient_id: e.g., 'chb01'
        output_dir: Directory to save the raw EDF files
        seizure_files_only: If True, downloads all seizure recordings plus
                            sufficient pure interictal files (at least 4h separated).
        interictal_hours_needed: Minimum hours of baseline interictal data to acquire.
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_filename = f"{patient_id}-summary.txt"
    summary_url = f"{PHYSIONET_BASE_URL}{patient_id}/{summary_filename}"
    summary_path = output_dir / summary_filename

    # 1. Download summary
    if not download_file(summary_url, summary_path):
        raise RuntimeError(f"Could not retrieve summary for {patient_id}")

    # 2. Parse summary
    metadata = parse_summary_file(summary_path)
    records = metadata["records"]

    # Save parsed metadata as JSON
    json_path = output_dir / f"{patient_id}_metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Patient {patient_id}: {len(records)} records found.")
    
    # 3. Determine which files to download
    seizure_files = [fn for fn, r in records.items() if r["num_seizures"] > 0]
    total_seizures = sum(r["num_seizures"] for r in records.values())
    logger.info(f"Patient {patient_id}: {len(seizure_files)} files containing {total_seizures} seizures.")

    files_to_download = list(seizure_files)

    if seizure_files_only:
        # Pick interictal recordings that are far from any seizure
        # Take the earliest and latest recordings with 0 seizures
        pure_interictal = [fn for fn, r in records.items() if r["num_seizures"] == 0]
        # Choose at least interictal_hours_needed files
        files_to_download.extend(pure_interictal[:interictal_hours_needed])
    else:
        files_to_download = list(records.keys())

    # 4. Download EDF files
    for filename in files_to_download:
        edf_url = f"{PHYSIONET_BASE_URL}{patient_id}/{filename}"
        edf_path = output_dir / filename
        download_file(edf_url, edf_path)

    return metadata


if __name__ == "__main__":
    default_config.ensure_directories()
    # Test downloading summary and key records for patient 01
    logger.info("Starting download for patient chb01...")
    download_patient_data("chb01", seizure_files_only=True, interictal_hours_needed=2)
