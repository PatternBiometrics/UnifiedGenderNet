# HaGRID-Derived Face–Hand Pairs  
A derived dataset from HaGrid for demographic prediction (age, sex, skin tone) using aligned hand and face images.


---

## 1 Overview
This repository distributes a **derived subset of the original [HaGRID](https://github.com/hukenovs/hagrid) gesture-dataset**.  
Every record pairs a cropped hand with the same user’s face and comes with audited demographic labels:

| Column              | Description                              |
|---------------------|------------------------------------------|
| `id`                | Unique sample ID                         |
| `user_id`           | Subject identifier (across samples)      |
| `age`               | Age bucket                               |
| `gender`            | Male / Female                            |
| `race`              | skin-tone class        |
| `labels`            | HaGRID gesture labels (stop_inverted, stop, or palm)      |
| `hand_image_name`   | Filename of the cropped hand             |
| `face_image_name`   | Filename of the cropped face             |

The dataset ships as

* **`aligned_dataset.zip`** – cropped & aligned PNG/JPG images (https://drive.google.com/file/d/1-52CWGkVhs4k3uWtvAdSz7tvplSX0_84/view?usp=drive_link)
* **`train.csv` · `val.csv` · `test.csv`** – metadata splits  (https://drive.google.com/file/d/1W136bhwoVzT_ipjzABLie377wNNW8H6b/view?usp=drive_link, https://drive.google.com/file/d/1hilEWBh1GHsi469CQ5zckZnmqBOMo73q/view?usp=drive_link, https://drive.google.com/file/d/1dXKcOGVHGKRSERFoVN1HSSM-g7SlWnlf/view?usp=drive_link)
* *(optional)* `removed_faces.zip` / `removed_hands.zip` – raw removals during cleaning

---

## 2 Attribution & licensing
* **Source** – Derived from HaGRID, released under a *public licence with attribution and conditions reserved*.  
* **Derived data** – Redistributed **under the same licence**. See `HAGRID_DERIVED_LICENSE.pdf`.  
* **Code** – All scripts in `scripts/` are MIT-licensed (`LICENSE-CODE.md`).

---

## 3 Quick start

```bash
# 1 · Clone the repo
git clone https://github.com/<your-user>/hagrid-facehand-pairs.git
cd hagrid-facehand-pairs

# 2 · Install requirements
pip install -r scripts/requirements.txt

# 3 · Download & unpack (~3 GB)
python scripts/download_derived_hagrid.py
````

---

## 4 Data structure & naming convention

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
                │   ├── F_fe7d7a58-cc64-4755-849a-55970b08b75a.jpg
                │   └── …
                └── hands/
                    ├── H_fe7d7a58-cc64-4755-849a-55970b08b75a.jpg
                    └── …
```

* **Face crops** live in `…/faces/` and start with **`F_`**.
* **Hand crops** live in `…/hands/` and start with **`H_`**.
* The UUID suffix is identical for the paired images of the same sample.

> **Example pair**
> `faces/F_fe7d7a58-cc64-4755-849a-55970b08b75a.jpg`
> `hands/H_fe7d7a58-cc64-4755-849a-55970b08b75a.jpg`

---

## 5 Minimal usage example

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
## Checkpoints 
the studies are ablation study, cross modality , .....
        ...  tables

### Checkpoints 
 link to drive.
 
tag naming in config files ex.
\texttt{tf\_efficientnetv2\_s.in1k\_F0\_Af\_Lf\_Z0}

i.e., EfficientNetV2-S backbone, flag OFF (F0), \emph{Full} augmentation (Af), focal loss (Lf), and no layer freezing (Z0). Its selection is supported by the validation summary recorded in the run logs: mean accuracy \(0.9108\) (std \(0.0027\)) with per-seed accuracies \(0.9100\) (seed 42), \(0.9085\) (seed 123), \(0.9137\) (seed 999).


## 6 Dependencies

* Python ≥ 3.8
* [`gdown`] – Google-Drive download helper
* `pandas`

All pinned in `scripts/requirements.txt`.

---

## 7 Citation

```bibtex
@misc{hagrid_derived_2025,
  author       = {Mohamed Ait abderrahmane},
  title        = {Unified Deep Learning Model for Sex Prediction from Either Face or Hand Image},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-user>/hagrid-facehand-pairs}}
}
@misc{nuzhdin2024hagridv21mimagesstatic,
    title={HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition}, 
    author={Anton Nuzhdin and Alexander Nagaev and Alexander Sautin and Alexander Kapitanov and Karina Kvanchiani},
    year={2024},
    eprint={2412.01508},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.01508}, 
}

```

---

## 8 Contact

Open an issue or ping **@\Ait-abderrahmane** on GitHub.

---

## 9 Acknowledgements

Huge thanks to the original HaGRID authors for releasing their dataset and to the open-source community for the tools that enabled this derivative work.

