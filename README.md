# Patient-Specific Epileptic Seizure Prediction via Neurophysiological Biomarkers and Compact Deep Convolutional Networks (EEGNet)

**Author:** Adrián Muñoz Aradilla  
**Course:** Applications of Soft Computing (ASC), 2026–2027  
**Track:** Track A — EEG Seizure Prediction (Research Route: Read, Reproduce, Extend)

---

## 1. Project Overview & Clinical Objective
Over 50 million people worldwide live with epilepsy, and approximately 30% are pharmacoresistant to conventional anti-seizure medications (ASMs). The primary hazard of epilepsy is the unpredictable onset of seizures, which causes injuries, psychological distress, and sudden unexpected death in epilepsy (SUDEP).

This project implements an end-to-end, leak-free machine learning and deep learning system for **prospective seizure prediction** from continuous scalp EEG recordings (CHB-MIT database). 

Unlike seizure *detection* (which identifies an ongoing seizure), seizure *prediction* forecasts impending seizures **before clinical or electrographic onset** within a predefined temporal window, giving patients and closed-loop neuromodulation devices adequate lead time for intervention.

---

## 2. Rigorous Scientific Validation Protocol
To eliminate historical pitfalls such as data leakage, autocorrelation bias, and over-optimistic performance claims, this project strictly adheres to the consensus guidelines from:
- **Kuhlmann et al. (2018)**, *Nature Reviews Neurology* ([DOI: 10.1038/s41582-018-0055-2](https://doi.org/10.1038/s41582-018-0055-2))
- **Ren et al. (2022)**, *Frontiers in Neurology* ([DOI: 10.3389/fneur.2022.1016224](https://doi.org/10.3389/fneur.2022.1016224))
- **Lawhern et al. (2018)**, *Journal of Neural Engineering* ([DOI: 10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c))
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
5. **Exclusions (Zero-Leakage):**
   - Ictal periods ($[t_{\text{onset}}, t_{\text{offset}}]$) and postictal intervals ($[t_{\text{offset}}, t_{\text{offset}} + 1\text{ hour}]$) are completely excised.
6. **Leave-One-Seizure-Out (LOSO) Cross-Validation:**
   - Under no circumstances are preictal windows shuffled randomly between training and testing.
   - For a patient with $K$ seizures, cross-validation runs across $K$ folds; each fold withholds one entire seizure's preictal block and a proportional slice of interictal recording for out-of-sample testing.

---

## 3. Signal Representations & Models

| Category | Model | Signal Representation | Characteristics |
| :--- | :--- | :--- | :--- |
| **Baseline** | Logistic Regression (L2) | Representation 1 (Biomarkers) | Simple linear baseline with balanced class weights |
| **Chance Benchmark** | Analytical Poisson Predictor | Theoretical / Monte Carlo | Chance sensitivity: $1 - e^{-\text{FPR/h} \times (\text{SOP}/60)}$ |
| **Classical Model 1** | Random Forest | Representation 1 (Biomarkers) | Non-linear ensemble with balanced subsampling |
| **Classical Model 2** | Support Vector Machine (RBF) | Representation 1 (Biomarkers) | Calibrated probability scaling, radial basis function |
| **Classical Model 3** | Multilayer Perceptron (ANN) | Representation 1 (Biomarkers) | 2 hidden layers (64, 32), ReLU, Adam, Early stopping |
| **Deep Learning** | **EEGNet** (Lawhern et al. 2018) | **Representation 2 (Raw EEG Tensors)** | Spatio-temporal convolutions (2D temporal conv + depthwise spatial conv + separable conv) |

### Representation 1: Neurophysiological Biomarkers
- **Relative Band Power (Welch PSD):** Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), Gamma (30–45 Hz).
- **Slow-to-Fast Ratio:** $(\text{Delta} + \text{Theta}) / (\text{Alpha} + \text{Beta})$.
- **Normalized Spectral Entropy:** Measures spectral irregularity and loss of complexity.
- **Hjorth Parameters:** Activity (variance), Mobility (mean frequency), Complexity (frequency spread).

### Representation 2: Raw Spatio-temporal Multi-Channel Tensors
- Input shape: $(B, 1, 17, 1280)$ corresponding to 17 standard bipolar scalp channels over 10-second windows sampled at 128 Hz.
- Trained end-to-end using PyTorch with Class-Weighted Cross-Entropy and Early Stopping.

---

## 4. Repository Structure

```
proyecto-asc/
├── .gitignore                      # Git ignore rules (data, virtualenvs, weights)
├── README.md                       # Complete project documentation & guide
├── requirements.txt                # Python dependencies (PyTorch, MNE, Scipy, Sklearn)
├── run_experiments.py              # Main entrypoint to execute end-to-end experiments
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central configuration (sampling, horizons, channels)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_chbmit.py      # Automated PhysioNet / AWS S3 record downloader
│   │   └── dataset.py              # Windowing, filtering, and leak-free labeler
│   ├── features/
│   │   ├── __init__.py
│   │   └── biomarkers.py           # Band power, entropy, and Hjorth feature extractor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py             # Logistic Regression & Poisson random predictor
│   │   ├── classical.py            # Random Forest, Support Vector Machine, MLP
│   │   └── deep_eegnet.py          # PyTorch EEGNet architecture
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Sensitivity, FPR/h, AUC, Paired Bootstrap test
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                # Publication figures (signals, trajectories, ROC/PR)
├── notebooks/
│   └── 01_EEG_Seizure_Prediction_Benchmark.ipynb
├── results/
│   ├── figures/                    # Generated high-resolution plots
│   └── tables/                     # Metric summaries and statistical test JSONs
└── report/
    ├── main.tex                    # Complete 12-20 page structured LaTeX report
    └── references.bib              # Verified BibTeX citations with DOIs
```

---

---

## 5. Experimental Results (Leave-One-Seizure-Out CV)

Cross-validated evaluation on patient `chb01` across 3 complete folds (holding out 1 full seizure and distinct interictal recordings per fold):

| Model Family | Signal Representation | AUC-ROC | PR-AUC | Seizure Sens. (%) | FPR/h | Poisson Chance $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | Representation 1 (Biomarkers) | **0.997** | **0.999** | **100.0%** | **0.00** | **$p < 0.001$** (Significant) |
| **Random Forest** | Representation 1 (Biomarkers) | 0.974 | 0.990 | 100.0% | 5.00 | $p = 0.773$ (Not Significant) |
| **Support Vector Machine (RBF)** | Representation 1 (Biomarkers) | 0.993 | 0.997 | 100.0% | 7.00 | $p = 0.912$ (Not Significant) |
| **Multilayer Perceptron (ANN)** | Representation 1 (Biomarkers) | 0.989 | 0.996 | 100.0% | 4.00 | $p = 0.646$ (Not Significant) |
| **Deep EEGNet (Proposed)** | **Representation 2 (Raw EEG)** | 0.979 | 0.992 | **100.0%** | **0.00** | **$p < 0.001$** (Significant) |

### Key Experimental Takeaways:
1. **Critical Role of FPR/h:** Although all five models detected 100% of seizures, models with false alarm rates between 4.0 and 7.0 FA/h fail the analytical Poisson chance benchmark ($p > 0.05$). Under a Poisson process with high false alarms, high sensitivity can be attained purely by chance!
2. **Superiority of Zero-False-Alarm Models:** Both the regularized Baseline (Logistic Regression on spectral biomarkers) and the compact Deep EEGNet on raw waveforms achieved **$0.00$ FPR/h**, rigorously exceeding the Poisson chance threshold ($p < 0.001$).
3. **End-to-End Spatio-temporal Deep Learning:** EEGNet operates directly on multi-channel $17 \times 1280$ tensors without manual feature extraction, demonstrating that spatial depthwise convolutions successfully isolate discriminative preictal rhythms while rejecting common-mode artifacts.

---

## 6. Deliverables & Presentation

- **Interactive Jupyter Notebook:** [`notebooks/01_EEG_Seizure_Prediction_Benchmark.ipynb`](notebooks/01_EEG_Seizure_Prediction_Benchmark.ipynb) (pre-computed with inline graphs, tables, and narrative).
- **Academic LaTeX Manuscript:** [`report/main.tex`](report/main.tex) and [`report/references.bib`](report/references.bib) (IEEE/NeurIPS style, 12–20 pages).
- **1-Minute Video Pitch Script & Storyboard:** [`Documentación/VIDEO_SCRIPT_1MIN.md`](Documentación/VIDEO_SCRIPT_1MIN.md) (timed second-by-second teleprompter script and visual slides).
- **High-Resolution Figures:** Generated in [`results/figures/`](results/figures/).

---

## 7. Quickstart & Reproduction

### 1. Environment Setup
```bash
# Using uv (recommended) or standard venv with Python 3.12:
uv venv .venv --python 3.12
.venv\Scripts\activate

# Install dependencies:
uv pip install -r requirements.txt
```

### 2. Run Experiments (Download, Train, Evaluate, and Plot)
```bash
python run_experiments.py
```
This script will:
1. Load and verify the patient EEG records (CHB-MIT database).
2. Segment the continuous recording into preictal ($[t-35\text{m}, t-5\text{m}]$) and interictal ($\ge 4\text{h}$) windows.
3. Generate multi-channel signal traces and biomarker trajectory plots.
4. Execute Leave-One-Seizure-Out (LOSO) cross-validation across all model families.
5. Compute paired bootstrap confidence intervals ($\Delta \text{AUC}$) against baseline.
6. Verify statistical significance against an analytical Poisson random predictor.
7. Save all publication-ready figures to `results/figures/` and metrics to `results/tables/`.

---

## 8. Verification and Citations
All methods and hyperparameter choices are backed by peer-reviewed literature:
1. **Kuhlmann, L., et al. (2018).** *Epilepsy seizure prediction—made possible by big data and machine learning.* Nature Reviews Neurology, 14(10), 618–630. DOI: `10.1038/s41582-018-0055-2`
2. **Ren, M., et al. (2022).** *Evaluating epileptic seizure prediction systems: A review of performance metrics and validation strategies.* Frontiers in Neurology, 13, 1016224. DOI: `10.3389/fneur.2022.1016224`
3. **Lawhern, V. J., et al. (2018).** *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.* Journal of Neural Engineering, 15(5), 056013. DOI: `10.1088/1741-2552/aace8c`
4. **Jiang, Y., et al. (2023).** *A phase-amplitude coupling and spectral biomarker pipeline for seizure prediction.* Frontiers in Neuroscience, 17, 1191683. DOI: `10.3389/fnins.2023.1191683`
5. **Ibrahim, H., et al. (2023).** *Patient-wise evaluation and cross-validation protocols for EEG seizure prediction.* Frontiers in Signal Processing, 3, 1175305. DOI: `10.3389/frsip.2023.1175305`
6. **Goldberger, A. L., et al. (2000).** *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215–e220. DOI: `10.1161/01.CIR.101.23.e215`
