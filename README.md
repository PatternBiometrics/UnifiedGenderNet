
# UnifiedGenderNet  
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE-CODE.md)  [![Status](https://img.shields.io/badge/status-under--review-yellow.svg)](#) 

A **unified** deep learning model for gender prediction that works with **either a hand or a face** image at inference time.  

It includes two architectural variants:
- **UMCC** (Unified Modality-Conditioned Classifier)  
- **MAG** (Modality-Aware Gated network)  

⚠️ **Disclaimer**: This repository accompanies a manuscript currently under peer review.  
To avoid premature disclosure of unpublished material, only baseline-level results are shown here.  
Full ablations, fairness analyses, and cross-dataset evaluations will be released upon publication.

---

## 1. Quick Start

### Clone the repository  

```bash
git clone https://github.com/PatternBiometrics/UnifiedGenderNet.git
cd UnifiedGenderNet
````

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # on Linux/macOS
# .venv\Scripts\activate     # on Windows PowerShell
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download pretrained artifacts

```bash
python scripts/download_checkpoints.py
```

```bash
python scripts/download_metrics_csv.py
```


---

## 2. Dataset

We release a derived subset of **HaGRID** with aligned face–hand crops.

* **Metadata**: `train.csv`, `val.csv`, `test.csv`
* **Cropped images**: `/faces/` and `/hands/`
* **License**: Inherits from HaGRID (attribution + conditions)

👉 See: *HaGRID-Derived Face–Hand Pairs*

---

## 3. Checkpoints

* Model weights (UMCC / MAG, EfficientNetV2-S) are hosted on Google Drive.
* Fetch with:

```bash
python scripts/download_checkpoints.py
```

* Tags encode backbone & configuration (e.g., `tf_efficientnetv2_s.in1k_F0_Af_Lf_Z0`).

---

## 4. Metrics

* CSV file: `metrics/final_metrics_with_full_configurations.csv`
* Contains config, backbone, modality, and evaluation metrics.
* Convert to Markdown table with:

```bash
python scripts/build_model_zoo.py
```

---

## 5. Results (Preview Only)

Baseline (UMCC, EfficientNetV2-S, full augmentation, BCE, no freezing):

* Validation balanced accuracy ≈ **0.91**
* Test accuracy ≈ **91%**
* Test ROC–AUC ≈ **0.97**

📌 Full results will be shared once the paper is accepted.

---

## 6. Ethics & Licensing

* Labels = **apparent gender**, used only for technical evaluation
* Demographics used strictly for fairness evaluation
* Dataset redistribution complies with HaGRID license
* Code: MIT license (see `LICENSE-CODE.md`)

---

## 7. Citation

```bibtex
@misc{UnifiedGenderNet,
  title        = {Unified Model for Gender Prediction from Either Hand or Face Images},
  author       = {Ait Abderrahmane, Mohamed and collaborators},
  year         = {2025},
  howpublished = {\url{https://github.com/PatternBiometrics/UnifiedGenderNet}}
}
```

---

## 8. Acknowledgements

Huge thanks to the original **HaGRID** authors for releasing their dataset and to the open-source community for the tools that enabled this derivative work.

```

Do you want me to also **add shields.io badges** (for Python version, license, repo status, etc.) at the top so the README looks more professional on GitHub?
```
