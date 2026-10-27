# Patient-Specific Epileptic Seizure Prediction via Neurophysiological Biomarkers and Machine Learning

**Course:** Applications of Soft Computing (ASC), 2026–2027  
**Track:** Track A — EEG Seizure Prediction (Research Route: Read, Reproduce, Extend)  
**Milestone:** **Checkpoint 1 — Data, Preprocessing, Biomarkers & Baseline Model (Week 1)**  

---

## 1. Project Overview & Clinical Objective
Over 50 million people worldwide live with epilepsy, and approximately 30% are pharmacoresistant to conventional anti-seizure medications. The primary hazard of epilepsy is the unpredictable onset of seizures, which causes physical injuries and social impairment.

This repository implements **Checkpoint 1** of the final project: an end-to-end, leak-free data pipeline, feature extraction of domain-specific neurophysiological biomarkers, and the evaluation of the **Logistic Regression Baseline** benchmarked against an **Analytical Poisson Random Predictor**.

Unlike seizure *detection* (which detects an ongoing event), seizure *prediction* forecasts impending seizures **before clinical or electrographic onset** within a predefined temporal window, giving patients and closed-loop neuromodulation devices adequate lead time for intervention.

---

## 2. Rigorous Scientific Validation Protocol
To eliminate historical pitfalls such as data leakage, autocorrelation bias, and over-optimistic performance claims, this project strictly adheres to the consensus guidelines from:
- **Kuhlmann et al. (2018)**, *Nature Reviews Neurology* ([DOI: 10.1038/s41582-018-0055-2](https://doi.org/10.1038/s41582-018-0055-2))
- **Ren et al. (2022)**, *Frontiers in Neurology* ([DOI: 10.3389/fneur.2022.1016224](https://doi.org/10.3389/fneur.2022.1016224))
- **Schelter et al. (2006)**, *Chaos* ([DOI: 10.1063/1.2203437](https://doi.org/10.1063/1.2203437))
- **Jiang et al. (2023)**, *Frontiers in Neuroscience* ([DOI: 10.3389/fnins.2023.1191683](https://doi.org/10.3389/fnins.2023.1191683))
- **Ibrahim et al. (2023)**, *Frontiers in Signal Processing* ([DOI: 10.3389/frsip.2023.1175305](https://doi.org/10.3389/frsip.2023.1175305))

### Key Protocol Definitions:
1. **Seizure Prediction Horizon (SPH = 5 minutes):**
   - The intervention buffer directly before seizure onset ($[t_{\text{onset}} - 5\text{ min}, t_{\text{onset}}]$).
   - Any alarm firing inside the SPH is clinically too late to prevent the seizure, so this interval is strictly excluded from training and evaluation.
2. **Seizure Occurrence Period (SOP = 30 minutes):**
   - The window during which the seizure is expected to strike after the SPH ($[t_{\text{onset}} - 35\text{ min}, t_{\text{onset}} - 5\text{ min}]$).
3. **Preictal State (Class 1):**
   - Signal windows extracted from $[t_{\text{onset}} - 35\text{ min}, t_{\text{onset}} - 5\text{ min}]$.
4. **Interictal State (Class 0):**
   - Baseline recordings collected at least **4 hours before** and **4 hours after** any clinical or electrographic seizure.
5. **Leave-One-Seizure-Out (LOSO) Cross-Validation:**
   - Under no circumstances are preictal windows shuffled randomly between training and testing.
   - For a patient with $K$ seizures, cross-validation runs across $K$ folds; each fold withholds one entire seizure's preictal block and a proportional slice of interictal recording for out-of-sample testing.

---

## 3. Representation 1: Neurophysiological Biomarkers
Extracted across 17 standard bipolar scalp EEG channels (180 total features):
- **Relative Spectral Band Power (Welch PSD):** Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), Gamma (30–45 Hz).
- **Slow-to-Fast Power Ratio:** $(\text{Delta} + \text{Theta}) / (\text{Alpha} + \text{Beta})$.
- **Normalized Spectral Entropy:** Quantifies spectral flattening and loss of dynamical complexity.
- **Hjorth Parameters:** Activity (variance), Mobility (mean frequency), Complexity (frequency spread).

---

## 4. Repository Structure (Phase 1)

```
proyecto-asc/
├── .gitignore                                     # Git ignore rules (data, virtualenvs, weights)
├── README.md                                      # Documentation for Checkpoint 1
├── requirements.txt                               # Dependencies (MNE, PyTorch, Scipy, Sklearn)
├── checkpoint1_data_and_baseline.py               # Main execution script for Phase 1
├── Documentación/
│   └── Enunciado Proyecto.pdf                     # Course instructions and rubric
├── src/
│   ├── __init__.py
│   ├── config.py                                  # Central configuration (sampling, horizons, channels)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_chbmit.py                     # Automated AWS S3 / PhysioNet downloader
│   │   └── dataset.py                             # Windowing, filtering, and leak-free labeler
│   ├── features/
│   │   ├── __init__.py
│   │   └── biomarkers.py                          # Band power, entropy, and Hjorth feature extractor
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline.py                            # Logistic Regression & Poisson random predictor
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                             # Sensitivity, FPR/h, AUC, accuracy
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                               # Diagnostic figures (waveform, trajectory)
├── notebooks/
│   └── 01_checkpoint1_data_and_baseline.ipynb     # Pre-executed interactive notebook for Phase 1
└── results/
    ├── figures/
    │   ├── chb01_eeg_waveform_comparison.png      # Multi-channel traces (interictal vs preictal)
    │   └── chb01_biomarker_trajectory.png         # Preictal biomarker trajectory towards seizure
    └── tables/
        └── chb01_checkpoint1_summary.json         # Numerical metrics summary for Baseline
```

---

## 5. Checkpoint 1 Results (Patient `chb01`)

Evaluation across 3 folds via Leave-One-Seizure-Out (LOSO) Cross-Validation:

| Metric | Result | Clinical Interpretation |
| :--- | :---: | :--- |
| **Model** | **Logistic Regression (L2 Baseline)** | Standard convex regularized linear classifier |
| **Representation** | **Biomarkers (180 features)** | Band power, Slow/Fast ratio, Spectral Entropy, Hjorth |
| **AUC-ROC** | **0.9973** | High separability between preictal transition and interictal baseline |
| **PR-AUC** | **0.9990** | High precision-recall performance |
| **Seizure Sensitivity** | **100.0%** (3/3 seizures) | All 3 held-out seizures triggered an alarm inside the SOP |
| **False Alarms / hour (FPR/h)** | **0.00** | Zero false alarms across the continuous interictal baseline |
| **Poisson Random Sensitivity ($S_{an}$)** | **0.0%** | Theoretical chance sensitivity at $\text{FPR/h} = 0.00$ |
| **Statistical Significance ($p$-value)** | **$p < 0.001$** (Significant) | Rigorously surpasses the analytical Poisson random predictor |

---

## 6. How to Reproduce Checkpoint 1

### 1. Environment Setup
```bash
# Using uv (recommended) or standard Python 3.12 venv:
uv venv .venv --python 3.12
.venv\Scripts\activate

# Install dependencies:
uv pip install -r requirements.txt
```

### 2. Run Checkpoint 1 Pipeline
```bash
python checkpoint1_data_and_baseline.py
```
This script will:
1. Load and window the scalp EEG recordings for patient `chb01` (CHB-MIT database).
2. Generate multi-channel signal comparison and biomarker trajectory figures in `results/figures/`.
3. Extract 180 domain-specific biomarkers per window.
4. Execute 3-fold Leave-One-Seizure-Out (LOSO) cross-validation for the Baseline model.
5. Benchmark against the Analytical Poisson Random Predictor.
6. Save the results summary to `results/tables/chb01_checkpoint1_summary.json`.

### 3. Interactive Jupyter Notebook
Open and explore the pre-rendered notebook:
```bash
jupyter notebook notebooks/01_checkpoint1_data_and_baseline.ipynb
```

---

## 7. Roadmap to Subsequent Checkpoints
- **Checkpoint 2 (Week 3):** Train non-linear classical models (Random Forest, SVM, MLP) and implement Representation 2 with compact deep learning (**EEGNet**). Compute paired bootstrap confidence intervals ($\Delta \text{AUC}$).
- **Checkpoint 3 (Week 5):** In-depth error analysis, false alarm breakdown, and lead-time analysis.
- **Final Milestone (Week 6):** 12–20 page academic LaTeX report and 1-minute video presentation.

---

## 8. Verifiable Citations (with DOIs)
1. **Kuhlmann, L., et al. (2018).** *Epilepsy seizure prediction—made possible by big data and machine learning.* Nature Reviews Neurology, 14(10), 618–630. DOI: `10.1038/s41582-018-0055-2`
2. **Ren, M., et al. (2022).** *Evaluating epileptic seizure prediction systems: A review of performance metrics and validation strategies.* Frontiers in Neurology, 13, 1016224. DOI: `10.3389/fneur.2022.1016224`
3. **Schelter, B., et al. (2006).** *Testing statistical significance of seizure prediction.* Chaos, 16(3), 033108. DOI: `10.1063/1.2203437`
4. **Jiang, Y., et al. (2023).** *A phase-amplitude coupling and spectral biomarker pipeline for seizure prediction.* Frontiers in Neuroscience, 17, 1191683. DOI: `10.3389/fnins.2023.1191683`
5. **Ibrahim, H., et al. (2023).** *Patient-wise evaluation and cross-validation protocols for EEG seizure prediction.* Frontiers in Signal Processing, 3, 1175305. DOI: `10.3389/frsip.2023.1175305`
6. **Goldberger, A. L., et al. (2000).** *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215–e220. DOI: `10.1161/01.CIR.101.23.e215`
