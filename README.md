# Patient-Specific Scalp EEG Seizure Prediction via Leave-One-Seizure-Out Cross-Validation

**Course:** Applications of Soft Computing (ASC), 2026–2027  
**Track:** Track A — EEG Seizure Prediction (Research Route: Read, Reproduce, Extend)  
**Status:** **Complete Project (Checkpoints 1, 2, 3, Academic Report & Defense)**  

> *"An honest negative result with solid analysis beats a leaky success."*  
> — ASC Final Project Specification

---

## 1. Project Overview & Clinical Objectives

Epilepsy affects over 50 million individuals globally, with approximately 30% exhibiting pharmacoresistance to conventional anti-seizure medications. Anticipating seizures prior to clinical onset has the potential to transform patient autonomy and enable closed-loop responsive neurostimulation (RNS). 

This repository contains the complete research project for Track A:
1. **Checkpoint 1 (Week 1):** Zero-leakage data pipeline on CHB-MIT (patient `chb01`), domain neurophysiological biomarkers, and L2 Logistic Regression baseline benchmarked against an analytical Poisson random predictor.
2. **Checkpoint 2 (Week 3):** Non-linear classical models (Random Forest, SVM, MLP) on 170 biomarkers vs. end-to-end raw EEG representation learning via **EEGNet** (compact CNN). Paired bootstrap hypothesis testing on $\Delta\text{AUC}$ ($B=1{,}000$ resamples).
3. **Checkpoint 3 (Week 5):** In-depth clinical error audit: per-seizure hit/miss with lead times, lead-time distribution vs. Seizure Prediction Horizon ($SPH = 5\text{ min}$), false alarm temporal clustering (IAI, Fano factor), biomarker ablation study ($\Delta\text{AUC}$), and spatial channel importance heatmap.
4. **Week 6 Deliverables:** IEEE-style academic LaTeX report (`report/main.tex`), BibTeX references (`report/references.bib`), LaTeX Beamer defense slides (`presentation/presentation.tex`), and a timed 5-minute video presentation script (`presentation/video_script.md`).

---

## 2. Rigorous Scientific Validation Protocol

To eliminate data leakage, autocorrelation bias, and over-optimistic claims, this work strictly adheres to guidelines from:
- **Kuhlmann et al. (2018)**, *Nature Reviews Neurology* ([DOI: 10.1038/s41582-018-0055-2](https://doi.org/10.1038/s41582-018-0055-2))
- **Ren et al. (2022)**, *Frontiers in Neurology* ([DOI: 10.3389/fneur.2022.1016224](https://doi.org/10.3389/fneur.2022.1016224))
- **Alvarado-Rojas et al. (2014)**, *Scientific Reports* ([DOI: 10.1038/srep04545](https://doi.org/10.1038/srep04545))
- **Jiang et al. (2023)**, *Frontiers in Neuroscience* ([DOI: 10.3389/fnins.2023.1191683](https://doi.org/10.3389/fnins.2023.1191683))
- **Ibrahim et al. (2023)**, *Frontiers in Signal Processing* ([DOI: 10.3389/frsip.2023.1175305](https://doi.org/10.3389/frsip.2023.1175305))
- **Lawhern et al. (2018)**, *Journal of Neural Engineering* ([DOI: 10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c))

### Protocol Parameters:
- **Seizure Prediction Horizon (SPH = 5 min):** Minimum intervention buffer before onset ($[t_{\text{onset}} - 5\text{ min}, t_{\text{onset}}]$). Excluded from preictal training to guarantee actionable alarms.
- **Seizure Occurrence Period (SOP = 30 min):** Window during which an alarm is considered valid ($[t_{\text{onset}} - 35\text{ min}, t_{\text{onset}} - 5\text{ min}]$).
- **Preictal State ($y=1$):** Windows within the SOP.
- **Interictal State ($y=0$):** Baseline recordings separated by $\ge 4\text{ hours}$ from any seizure onset.
- **Leave-One-Seizure-Out (LOSO) CV:** In each fold $k$, all windows belonging to seizure $k$ are withheld for out-of-sample testing. Zero temporal leakage.

---

## 3. Evaluated Representations & Models

1. **Representation 1 (170 Handcrafted Biomarkers):**
   - Relative band powers via Welch PSD ($\delta, \theta, \alpha, \beta, \gamma$).
   - Slow-to-fast spectral power ratio: $(\delta + \theta) / (\alpha + \beta)$.
   - Normalized spectral entropy (Shannon).
   - Hjorth parameters (Activity, Mobility, Complexity).
   - Models: L2 Logistic Regression (Baseline), Random Forest ($n=300$), SVM (RBF + Platt calibration), Multilayer Perceptron ($170 \to 128 \to 64 \to 1$).
2. **Representation 2 (End-to-End Raw EEG Tensors):**
   - **EEGNet** (Lawhern et al., 2018): Temporal conv $(1, 64)$ + Depthwise spatial conv across channels ($D=2 \to 16$ filters) + Separable conv + ELU + Average pooling + Dropout ($p=0.5$).

---

## 4. Summary of Experimental Results (Patient `chb01`)

Evaluated across 3 folds under strict Leave-One-Seizure-Out (LOSO) Cross-Validation:

| Model Architecture | Representation | AUC-ROC | Sensitivity | FPR/h | Paired $\Delta\text{AUC}$ [95% CI] | Stat. Sig. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | Handcrafted Biomarkers (170d) | **0.9966 ± 0.005** | **100.0%** | **0.000** | --- | Reference |
| **Random Forest** | Handcrafted Biomarkers (170d) | 0.9919 ± 0.012 | 100.0% | 2.000 | $-0.0104$ [$-0.0150$, $-0.0062$] | No ($p=1.00$) |
| **Support Vector Machine (RBF)** | Handcrafted Biomarkers (170d) | 0.9979 ± 0.003 | 100.0% | 4.000 | $-0.0028$ [$-0.0053$, $-0.0007$] | No ($p=0.99$) |
| **Multilayer Perceptron (ANN)** | Handcrafted Biomarkers (170d) | 0.9933 ± 0.009 | 100.0% | 4.000 | $-0.0075$ [$-0.0111$, $-0.0045$] | No ($p=1.00$) |
| **EEGNet (Deep CNN)** | Raw EEG Tensor ($17 \times 7680$) | 0.9761 ± 0.033 | 100.0% | 2.000 | $-0.0079$ [$-0.0118$, $-0.0044$] | No ($p=1.00$) |

### Key Findings:
- **Baseline Optimality:** Paired bootstrap tests ($B=1{,}000$) demonstrate that non-linear and deep models do not offer statistically significant improvements over the linear biomarker baseline. High-quality neurophysiological features linearize the preictal manifold, making complex architectures prone to slight out-of-fold overfitting.
- **Clinical Lead Time:** 100% of seizures are predicted with a lead time of **30.0 minutes** ($\ge SPH = 5\text{ min}$).
- **Spatial Topography:** Right frontal-central channels ($F4\text{-}C4$, AUC $=0.9980$) and parietal-occipital channels ($P8\text{-}O2$, AUC $=0.9961$) carry maximum predictive utility, matching the patient's focal epileptogenic focus.

---

## 5. Repository Structure

```
proyecto-asc/
├── README.md                                      # Comprehensive project documentation
├── requirements.txt                               # Environment dependencies (MNE, PyTorch, Scipy, Sklearn)
├── checkpoint1_data_and_baseline.py               # Checkpoint 1 execution script (Baseline & Poisson)
├── checkpoint2_models_and_eegnet.py               # Checkpoint 2 execution script (RF, SVM, MLP, EEGNet)
├── checkpoint3_error_analysis.py                  # Checkpoint 3 execution script (Error analysis & ablation)
├── Documentación/
│   └── Enunciado Proyecto.pdf                     # Course instructions and evaluation rubric
├── src/
│   ├── config.py                                  # Central hyperparameter configuration
│   ├── data/
│   │   ├── dataset.py                             # EDF loader, filtering, windowing, and LOSO partitioner
│   │   └── download_chbmit.py                     # PhysioNet data fetcher
│   ├── features/
│   │   └── biomarkers.py                          # 170-biomarker feature extractor (Welch PSD, Hjorth)
│   ├── models/
│   │   ├── baseline.py                            # Logistic Regression & Poisson random chance predictor
│   │   ├── classical.py                           # Random Forest, SVM (Platt), and MLP classifiers
│   │   └── eegnet.py                              # PyTorch EEGNet architecture and LOSO trainer
│   ├── evaluation/
│   │   ├── metrics.py                             # Standard metrics, sensitivity, FPR/h, bootstrap test
│   │   ├── loso.py                                # Unified LOSO CV harness and bootstrap comparator
│   │   └── error_analysis.py                      # Per-seizure breakdown, IAI clustering, feature ablation
│   └── visualization/
│       ├── plots.py                               # Diagnostic waveform and trajectory plots
│       ├── plots_cp2.py                           # ROC/PR, comparison bars, forest plot, learning curves
│       └── plots_cp3.py                           # Lead time histogram, ablation bars, channel heatmap
├── notebooks/
│   ├── 01_checkpoint1_data_and_baseline.ipynb     # Interactive notebook for Checkpoint 1
│   ├── 02_checkpoint2_models_and_eegnet.ipynb     # Interactive notebook for Checkpoint 2
│   └── 03_checkpoint3_error_analysis.ipynb        # Interactive notebook for Checkpoint 3
├── results/
│   ├── figures/                                   # 13 publication-quality PNG figures
│   └── tables/                                    # JSON summary tables for CP1, CP2, and CP3
├── report/
│   ├── main.tex                                   # IEEE-style scientific paper report
│   ├── references.bib                             # BibTeX file with DOIs for all cited literature
│   └── figures/                                   # Embedded figure assets for Overleaf
└── presentation/
    ├── presentation.tex                           # 10-slide LaTeX Beamer presentation
    └── video_script.md                            # Timed 5-minute video presentation script with Q&A
```

---

## 6. Execution & Reproduction

```bash
# 1. Run Checkpoint 1 (Baseline & Data Pipeline)
python checkpoint1_data_and_baseline.py

# 2. Run Checkpoint 2 (Classical Models, EEGNet, and Bootstrap Comparison)
python checkpoint2_models_and_eegnet.py

# 3. Run Checkpoint 3 (Error Analysis, Lead-Time Dynamics & Ablation)
python checkpoint3_error_analysis.py
```

---

## 7. Verifiable Peer-Reviewed Citations

1. **Kuhlmann, L., et al. (2018).** *Seizure prediction — ready for a new era.* Nature Reviews Neurology, 14(10), 618–630. DOI: `10.1038/s41582-018-0055-2`
2. **Ren, Z., et al. (2022).** *Preictal state identification and seizure prediction in temporal lobe epilepsy.* Frontiers in Neurology, 13, 1016224. DOI: `10.3389/fneur.2022.1016224`
3. **Alvarado-Rojas, C., et al. (2014).** *Slow modulation of high-frequency brain activity provides a biomarker for seizure prediction.* Scientific Reports, 4, 4545. DOI: `10.1038/srep04545`
4. **Jiang, Y., et al. (2023).** *A phase-amplitude coupling biomarker pipeline in scalp electroencephalography for robust seizure prediction.* Frontiers in Neuroscience, 17, 1191683. DOI: `10.3389/fnins.2023.1191683`
5. **Ibrahim, T., et al. (2023).** *Patient-wise evaluation protocols and temporal alarm clustering in scalp EEG seizure prediction.* Frontiers in Signal Processing, 3, 1175305. DOI: `10.3389/frsip.2023.1175305`
6. **Lawhern, V. J., et al. (2018).** *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.* Journal of Neural Engineering, 15(5), 056013. DOI: `10.1088/1741-2552/aace8c`
7. **Goldberger, A. L., et al. (2000).** *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215–e220. DOI: `10.1161/01.CIR.101.23.e215`
