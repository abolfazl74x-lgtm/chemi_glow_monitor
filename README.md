# chemi_glow_monitor
Python tools for quantifying chemiluminescence in hydrogel sensors. Includes video data extraction, temporal signal processing, and phase-separated kinetic modeling.
# Hydrogel Chemiluminescence Kinetics & Signal Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a suite of custom-built Python software tools developed to automate the extraction, signal processing, and kinetic modeling of chemiluminescence data. Originally developed to support doctoral research on hydrogel-based chemical sensors for selective heavy metal detection (e.g., nickel ion binding using dimethylglyoxime-functionalized networks), this toolset bridges the gap between raw video data and advanced statistical optimization (such as Response Surface Methodology via Design Expert).

The suite consists of three interconnected modules designed to eliminate human error, enhance reproducibility, and extract deep kinetic insights from temporal luminescence profiles.

---

## 🛠️ Software Modules

### 1. Video Intensity Extractor (`chemi_glow_monitor.py`)
A computer vision tool built with OpenCV that converts qualitative video footage of chemiluminescence into quantitative data.
* **Features:**
  * Interactive Region of Interest (ROI) selection with adaptive thresholding for smart area detection.
  * Frame-by-frame extraction of average pixel intensity (0-255 scale).
  * Real-time temporal monitoring and automated export of time-resolved intensity profiles to Excel.

### 2. Advanced Signal Analyzer (`data_analyser.py`)
A comprehensive signal processing GUI that evaluates the raw temporal data to extract critical response variables for optimization algorithms.
* **Features:**
  * Savitzky-Golay filtering to mitigate instrumental noise.
  * Automated identification of Peak Intensity and Time-to-Peak.
  * Integration module utilizing trapezoidal numerical integration to calculate Total Area Under the Curve (AUC) and targeted high-emission windows.
  * Generates formatted output tables ready for integration with Design Expert (RSM split-plot designs).

### 3. Kinetic Model Fitter (`gui_app.py` & `kinetic_analyzer.py`)
A non-linear regression tool utilizing the SciPy optimization module to evaluate the reaction kinetics during both signal growth (build-up) and decay (recovery) phases.
* **Features:**
  * Fits data to multiple kinetic models: First-Order, Second-Order, Avrami, and Sigmoidal.
  * Model selection governed by the Akaike Information Criterion (AIC) and Coefficient of Determination ($R^2$).
  * Automatic calculation of specific kinetic parameters, including the half-life response time ($t_{1/2}$) and 95% confidence intervals derived from the covariance matrix.

---

## 💻 Installation & Prerequisites

To run these scripts locally, ensure you have Python 3.8 or higher installed. The required third-party scientific libraries can be installed via pip:

```bash
pip install numpy pandas scipy matplotlib opencv-python scikit-learn openpyxl
