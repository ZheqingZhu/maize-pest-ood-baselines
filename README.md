# Maize Leaf Herbivory Dataset: PyTorch Baselines

PyTorch code for the paper: **A Spatio-temporal Image Dataset of Maize Leaf Herbivory by Four Major Pests** (*Scientific Data*).

## 1. Data Preparation

The image data and metadata are hosted on Zenodo. You must download them before running the scripts.

1. Download `Maize_Leaf_Herbivory_Dataset.zip` from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20327492.svg)](https://doi.org/10.5281/zenodo.20327492)
2. Extract the archive.
3. Move the `data/` directory and `metadata.csv` into the root folder of this repository.

The repository structure should be:
```text
.
├── data/                  # From Zenodo
├── metadata.csv           # From Zenodo
├── dataset.py
├── train.py
├── train_cross_instar.py
├── train_mtl.py
├── train_mtl_mixed.py
└── README.md
```

## 2. Dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas numpy
```

## 3. Scripts

Run the scripts directly from the root directory. Evaluation metrics and confusion matrices will be saved to the `results/` folder.

* **`train.py`**: Standard 4-class species classification.
* **`train_mtl.py`**: Multi-task learning (species classification + discrete instar classification).
* **`train_mtl_mixed.py`**: Mixed multi-task learning (species classification + continuous age regression).
* **`train_cross_instar.py`**: Out-of-distribution (OOD) evaluation. Trains on instars 1-3, tests on unseen instars 4-6.
