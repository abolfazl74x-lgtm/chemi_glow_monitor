# chemi_glow_monitor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Android](https://img.shields.io/badge/Android-Kotlin%20%7C%20Lumin%20v1.0.0-green.svg)](https://github.com/abolfazl74x-lgtm/chemi_glow_monitor/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository provides the open-source Python analysis suite and standalone Android application (**Lumin**) developed via AI prompt engineering to support the manuscript:

> *"Synthesis of a natural-based hydrogel sensor integrated with AI prompt-engineered software for portable H₂O₂ and glucose detection"*

The toolset automates chemiluminescence video acquisition, region-of-interest (ROI) luminance extraction, temporal signal processing for split-plot Response Surface Methodology (RSM) optimization, phase-separated kinetic modeling, and on-device quantitative calibration.

---

## Repository Structure
- `chemi_glow_monitor.py` — OpenCV-based video ROI intensity extractor
- `data_analyser.py` — Signal smoothing and RSM response extraction GUI
- `kinetic_analyzer.PY` — Nonlinear kinetic modeling and model-selection GUI
- `Data.xlsx` & `saved_roi_coords.json` — Example data input template and saved ROI coordinates
- `Lumin_source_code.zip` — Full Kotlin/Android Studio source code for the **Lumin** mobile app
- **Releases (`v1.0.0`)** — Pre-built installable `.apk` file for **Lumin**

---

## Desktop Python Modules

### 1. Video Intensity Extractor (`chemi_glow_monitor.py`)
Converts smartphone chemiluminescence recordings into time-resolved quantitative intensity profiles using OpenCV.
- Interactive ROI selection with fixed coordinate reuse (`saved_roi_coords.json`) and adaptive thresholding.
- Frame-by-frame grayscale pixel intensity extraction (0–255 scale).
- Automated export of temporal luminance curves for downstream analysis.

### 2. Advanced Signal Analyzer (`data_analyser.py`)
Processes raw intensity-time data to compute analytical and optimization metrics.
- Savitzky–Golay smoothing (`SciPy`) to reduce measurement noise.
- Automated extraction of peak intensity, time to peak, cutoff-bounded total area under the curve (AUC) via trapezoidal integration, emission duration, and 20.0 s sliding-window max emission AUC.
- Structured outputs compatible with Design-Expert split-plot RSM optimization.

### 3. Kinetic Analyzer (`kinetic_analyzer.PY`)
Evaluates chemiluminescence reaction kinetics using `scipy.optimize.curve_fit`.
- Fits normalized kinetic profiles to **Zero-Order, First-Order, Second-Order, Avrami, Hill-Sigmoidal, and Bi-exponential** models.
- Automated model selection via Akaike Information Criterion (AIC) and $R^2$, along with extraction of kinetic parameters such as half-life response time ($t_{50}$).

### Python Prerequisites
Ensure Python 3.8+ is installed, then install the required packages:

```bash
pip install numpy pandas scipy matplotlib opencv-python scikit-learn openpyxl
```

---

## Standalone Android Application (`Lumin`)
To eliminate the need for an external computer during on-site measurements, **Lumin** replicates the desktop OpenCV/SciPy pipeline directly on Android smartphones.

- **Manual Capture & Import:** Camera2 API manual control over ISO, focus distance, and long-exposure durations, embedded FFmpegKit time-lapse encoding, and batch image/video import.
- **OpenCV-Consistent ROI Extraction:** Memory-efficient sRGB region decoding (`BitmapRegionDecoder`) with Rec. 601 grayscale conversion (`Y = 0.299R + 0.587G + 0.114B`) and optional channel-weight presets.
- **SciPy-Matched Signal Processing:** Third-order Savitzky–Golay smoothing with boundary polynomial interpolation, peak detection, trapezoidal AUC integration, high-intensity duration, and sliding-window max emission AUC.
- **On-Device Calibration & Storage:** Local SQLite (Room) database supporting multi-feature linear regression calibration (Peak Value, Total AUC, Max Emission AUC), unknown sample quantification, and CSV export.

### Download & Source Code
- **Installable APK:** Download the latest `.apk` directly from the [Releases](https://github.com/abolfazl74x-lgtm/chemi_glow_monitor/releases) page.
- **Source Code:** Available in `Lumin_source_code.zip` in the root directory.
