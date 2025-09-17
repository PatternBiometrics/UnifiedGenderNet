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
```

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
python scripts/download_metrics_csv.py
```

---

## 2. Dataset

We release a **derived subset of HaGRID** with aligned face–hand crops.  
Every record pairs a cropped hand with the same user’s face and comes with audited demographic labels:

| Column            | Description                               |
|-------------------|-------------------------------------------|
| `id`              | Unique sample ID                          |
| `user_id`         | Subject identifier (across samples)       |
| `age`             | Age bucket                                |
| `gender`          | Male / Female                             |
| `race`            | Skin-tone class                           |
| `labels`          | HaGRID gesture labels (stop, stop_inverted, palm) |
| `hand_image_name` | Filename of the cropped hand              |
| `face_image_name` | Filename of the cropped face              |

### Data distribution

* **`aligned_dataset.zip`** – cropped & aligned images  
* **`train.csv` · `val.csv` · `test.csv`** – metadata splits  

### Data structure

```
data/
└── Shared_Derived_HaGRID_unified_Model_For_Sex_Prediction/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── images/
        └── content/
            └── aligned_dataset/
                ├── faces/
                │   ├── F_<uuid>.jpg
                │   └── …
                └── hands/
                    ├── H_<uuid>.jpg
                    └── …
```

* **Face crops** in `faces/` start with `F_`  
* **Hand crops** in `hands/` start with `H_`  
* The UUID suffix is identical for the paired images  

> Example pair:  
> `faces/F_fe7d7a58-cc64-4755-849a-55970b08b75a.jpg`  
> `hands/H_fe7d7a58-cc64-4755-849a-55970b08b75a.jpg`

### Minimal usage example

```python
import pandas as pd
from pathlib import Path

root = Path("data/Shared_Derived_HaGRID_unified_Model_For_Sex_Prediction")
df   = pd.read_csv(root / "train.csv")
print(f"{len(df):,} training samples")
print(df.head())

img_root = root / "images" / "content" / "aligned_dataset"
hand_path = img_root / "hands" / df.loc[0, "hand_image_name"]
face_path = img_root / "faces" / df.loc[0, "face_image_name"]
print(hand_path, face_path, sep="\n")
```

---

## 3. Checkpoints

* Pretrained model weights (UMCC / MAG, EfficientNetV2-S) are hosted on Google Drive.  
* Fetch with:  
  ```bash
  python scripts/download_checkpoints.py
  ```  
* Tags encode backbone & configuration:  
  ```
  tf_efficientnetv2_s.in1k_F0_Af_Lf_Z0
  ```

---

## 4. Metrics

* CSV file: `metrics/final_metrics_with_full_configurations.csv`  
* Contains all configs and evaluation metrics.  


---

## 5. Results (Preview Only)

Baseline (UMCC, EfficientNetV2-S, full augmentation, BCE, no freezing):

* Validation balanced accuracy ≈ **0.91**  
* Test accuracy ≈ **91%**  
* Test ROC–AUC ≈ **0.97**  

📌 Full results (ablations, fairness, cross-dataset) will be added once the paper is accepted.

---

## 6. Ethics & Licensing

* Labels = **apparent gender**, used only for evaluation  
* Demographics used strictly for fairness assessment  
* **Source** – Derived from HaGRID (public licence with attribution & conditions)  
* **Derived data** – redistributed under same licence  
* **Code** – MIT (`LICENSE-CODE.md`)  

---

## 7. Dependencies

* Python ≥ 3.9  
* `torch`, `torchvision`, `timm`, `pandas`, `gdown`, `tabulate`, `typer`, `rich`

---


## 8. Acknowledgements

* Huge thanks to the original **HaGRID** authors for releasing their dataset.  
* Gratitude to the open-source community for providing tools that enabled this derivative work.
